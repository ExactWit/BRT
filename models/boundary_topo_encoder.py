# boundary_topo_encoder.py
"""Scheme A: boundary-operator message passing on B-Rep cell complex."""

from __future__ import annotations

import torch
from torch import nn


def get_mask_from_length(length: torch.Tensor, max_length: int) -> torch.Tensor:
    return (
        torch.arange(max_length, device=length.device).unsqueeze(0).expand(length.shape[0], -1)
        < length.unsqueeze(-1)
    )


def build_face_edge_pairs(
    wire_index: torch.Tensor,
    edge_index: torch.Tensor,
    wire_index_length: torch.Tensor,
    edge_index_length: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (face_id, edge_id) pairs along oriented wire boundaries (+1 incidence)."""
    n_faces = wire_index.shape[0]
    n_wires = edge_index.shape[0]
    wire_to_face = torch.full((n_wires,), -1, dtype=torch.long, device=device)

    for face_id in range(n_faces):
        n_wires_on_face = int(wire_index_length[face_id].item())
        if n_wires_on_face <= 0:
            continue
        wires = wire_index[face_id, :n_wires_on_face].to(device)
        wire_to_face[wires] = face_id

    face_ids: list[int] = []
    edge_ids: list[int] = []
    for wire_id in range(n_wires):
        face_id = int(wire_to_face[wire_id].item())
        if face_id < 0:
            continue
        n_edges_on_wire = int(edge_index_length[wire_id].item())
        if n_edges_on_wire <= 0:
            continue
        edges = edge_index[wire_id, :n_edges_on_wire].tolist()
        face_ids.extend([face_id] * n_edges_on_wire)
        edge_ids.extend(edges)

    if not face_ids:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    return (
        torch.tensor(face_ids, dtype=torch.long, device=device),
        torch.tensor(edge_ids, dtype=torch.long, device=device),
    )


def scatter_boundary_to_faces(
    h_edge: torch.Tensor,
    face_ids: torch.Tensor,
    edge_ids: torch.Tensor,
    n_faces: int,
) -> torch.Tensor:
    """Apply ∂₂: aggregate oriented boundary edge features onto faces."""
    dim = h_edge.shape[-1]
    agg = torch.zeros(n_faces, dim, device=h_edge.device, dtype=h_edge.dtype)
    if face_ids.numel() == 0:
        return agg

    agg.index_add_(0, face_ids, h_edge[edge_ids])
    counts = torch.zeros(n_faces, 1, device=h_edge.device, dtype=h_edge.dtype)
    counts.index_add_(0, face_ids, torch.ones(face_ids.shape[0], 1, device=h_edge.device, dtype=h_edge.dtype))
    return agg / counts.clamp(min=1.0)


def scatter_incident_faces_to_edges(
    h_face: torch.Tensor,
    face_ids: torch.Tensor,
    edge_ids: torch.Tensor,
    n_edges: int,
) -> torch.Tensor:
    """Apply ∂₂ᵀ: aggregate incident face features onto edges."""
    dim = h_face.shape[-1]
    agg = torch.zeros(n_edges, dim, device=h_face.device, dtype=h_face.dtype)
    if edge_ids.numel() == 0:
        return agg

    agg.index_add_(0, edge_ids, h_face[face_ids])
    counts = torch.zeros(n_edges, 1, device=h_face.device, dtype=h_face.dtype)
    counts.index_add_(0, edge_ids, torch.ones(edge_ids.shape[0], 1, device=h_face.device, dtype=h_face.dtype))
    return agg / counts.clamp(min=1.0)


class BoundaryOperatorTopoEncoder(nn.Module):
  """Multi-layer ∂₂ / ∂₂ᵀ message passing replacing heuristic TopoEncoder."""

  def __init__(
      self,
      edge_dim: int = 128,
      face_dim: int | None = None,
      hidden_dim: int | None = None,
      num_layers: int = 3,
      dropout: float = 0.01,
  ):
      super().__init__()
      face_dim = edge_dim if face_dim is None else face_dim
      hidden_dim = 2 * edge_dim if hidden_dim is None else hidden_dim

      self.edge_dim = edge_dim
      self.face_dim = face_dim
      self.num_layers = num_layers

      self.face_update_layers = nn.ModuleList()
      self.edge_update_layers = nn.ModuleList()
      self.face_norms = nn.ModuleList()
      self.edge_norms = nn.ModuleList()

      for _ in range(num_layers):
          self.face_update_layers.append(
              nn.Sequential(
                  nn.Linear(face_dim + edge_dim + hidden_dim, hidden_dim, bias=False),
                  nn.LayerNorm(hidden_dim),
                  nn.ReLU(),
                  nn.Dropout(dropout),
                  nn.Linear(hidden_dim, face_dim, bias=False),
              )
          )
          self.edge_update_layers.append(
              nn.Sequential(
                  nn.Linear(edge_dim + face_dim, edge_dim, bias=False),
                  nn.LayerNorm(edge_dim),
                  nn.ReLU(),
                  nn.Dropout(dropout),
                  nn.Linear(edge_dim, edge_dim, bias=False),
              )
          )
          self.face_norms.append(nn.LayerNorm(face_dim))
          self.edge_norms.append(nn.LayerNorm(edge_dim))

      self.adj_face_layer = nn.Sequential(
          nn.Linear(2 * face_dim, hidden_dim, bias=False),
          nn.LayerNorm(hidden_dim),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(hidden_dim, hidden_dim, bias=False),
          nn.LayerNorm(hidden_dim),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(hidden_dim, hidden_dim, bias=False),
      )

      self.output_layer = nn.Sequential(
          nn.Linear(face_dim + hidden_dim, edge_dim, bias=False),
          nn.LayerNorm(edge_dim),
          nn.ReLU(),
          nn.Dropout(dropout),
      )

  def cpu_input(self):
      return ["edge_index_length"]

  def _aggregate_adj_faces(
      self,
      h_face: torch.Tensor,
      adj_face_index: torch.Tensor,
      adj_face_index_length: torch.Tensor,
  ) -> torch.Tensor:
      n_faces = h_face.shape[0]
      adj_mask = get_mask_from_length(adj_face_index_length, adj_face_index.shape[1])

      adj_faces = torch.gather(
          h_face.unsqueeze(0).expand(n_faces, -1, -1),
          1,
          adj_face_index.unsqueeze(-1).expand(-1, -1, h_face.shape[-1]),
      )
      adj_faces = self.adj_face_layer(
          torch.cat([h_face.unsqueeze(1).expand_as(adj_faces), adj_faces], dim=-1)
      )
      adj_faces = torch.masked_fill(adj_faces, ~adj_mask.unsqueeze(-1), 0.0)
      return adj_faces.sum(dim=1)

  def forward(
      self,
      edges,
      faces,
      edge_index,
      wire_index,
      face_index,
      edge_index_length,
      wire_index_length,
      adj_face_index_length,
  ):
      h_edge = edges
      h_face = faces
      device = h_edge.device

      face_ids, edge_ids = build_face_edge_pairs(
          wire_index,
          edge_index,
          wire_index_length,
          edge_index_length,
          device,
      )

      for layer_idx in range(self.num_layers):
          boundary_agg = scatter_boundary_to_faces(h_edge, face_ids, edge_ids, h_face.shape[0])
          adj_agg = self._aggregate_adj_faces(h_face, face_index, adj_face_index_length)

          face_delta = self.face_update_layers[layer_idx](
              torch.cat([h_face, boundary_agg, adj_agg], dim=-1)
          )
          h_face = self.face_norms[layer_idx](h_face + face_delta)

          incident_face_agg = scatter_incident_faces_to_edges(
              h_face, face_ids, edge_ids, h_edge.shape[0]
          )
          edge_delta = self.edge_update_layers[layer_idx](
              torch.cat([h_edge, incident_face_agg], dim=-1)
          )
          h_edge = self.edge_norms[layer_idx](h_edge + edge_delta)

      adj_agg = self._aggregate_adj_faces(h_face, face_index, adj_face_index_length)
      return self.output_layer(torch.cat([h_face, adj_agg], dim=-1))


def build_face_coedge_pairs(
    wire_index: torch.Tensor,
    coedge_index: torch.Tensor,
    wire_index_length: torch.Tensor,
    coedge_index_length: torch.Tensor,
    device: torch.device,
    coedge_sign: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (face_id, coedge_id, sign) along oriented wire boundaries.

    BRT stores one feature row per oriented half-edge (coedge). When ``coedge_sign``
    is absent, inner wires use -1 (first wire per face is treated as outer, +1).
    """
    n_faces = wire_index.shape[0]
    n_wires = coedge_index.shape[0]
    wire_to_face = torch.full((n_wires,), -1, dtype=torch.long, device=device)
    wire_slot_on_face = torch.zeros(n_wires, dtype=torch.long, device=device)

    for face_id in range(n_faces):
        n_wires_on_face = int(wire_index_length[face_id].item())
        if n_wires_on_face <= 0:
            continue
        wires = wire_index[face_id, :n_wires_on_face].to(device)
        wire_to_face[wires] = face_id
        for slot, wire_id in enumerate(wires.tolist()):
            wire_slot_on_face[wire_id] = slot

    face_ids: list[int] = []
    coedge_ids: list[int] = []
    signs: list[float] = []
    for wire_id in range(n_wires):
        face_id = int(wire_to_face[wire_id].item())
        if face_id < 0:
            continue
        n_coedges = int(coedge_index_length[wire_id].item())
        if n_coedges <= 0:
            continue
        coedges = coedge_index[wire_id, :n_coedges].tolist()
        default_sign = 1.0 if int(wire_slot_on_face[wire_id].item()) == 0 else -1.0
        for coedge_id in coedges:
            face_ids.append(face_id)
            coedge_ids.append(coedge_id)
            if coedge_sign is not None:
                signs.append(float(coedge_sign[coedge_id].item()))
            else:
                signs.append(default_sign)

    if not face_ids:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, torch.empty(0, device=device, dtype=torch.float32)

    return (
        torch.tensor(face_ids, dtype=torch.long, device=device),
        torch.tensor(coedge_ids, dtype=torch.long, device=device),
        torch.tensor(signs, dtype=torch.float32, device=device),
    )


def scatter_boundary_to_faces_signed(
    h_coedge: torch.Tensor,
    face_ids: torch.Tensor,
    coedge_ids: torch.Tensor,
    signs: torch.Tensor,
    n_faces: int,
) -> torch.Tensor:
    """∂₂ with signed incidence; linear sum without averaging."""
    dim = h_coedge.shape[-1]
    agg = torch.zeros(n_faces, dim, device=h_coedge.device, dtype=h_coedge.dtype)
    if face_ids.numel() == 0:
        return agg

    signed_messages = signs.unsqueeze(-1).to(h_coedge.dtype) * h_coedge[coedge_ids]
    agg.index_add_(0, face_ids, signed_messages)
    return agg


def scatter_incident_faces_to_coedges_signed(
    h_face: torch.Tensor,
    face_ids: torch.Tensor,
    coedge_ids: torch.Tensor,
    signs: torch.Tensor,
    n_coedges: int,
) -> torch.Tensor:
    """∂₂ᵀ with signed incidence; linear sum without averaging."""
    dim = h_face.shape[-1]
    agg = torch.zeros(n_coedges, dim, device=h_face.device, dtype=h_face.dtype)
    if coedge_ids.numel() == 0:
        return agg

    signed_messages = signs.unsqueeze(-1).to(h_face.dtype) * h_face[face_ids]
    agg.index_add_(0, coedge_ids, signed_messages)
    return agg


class BoundaryOperatorTopoEncoderA2(BoundaryOperatorTopoEncoder):
    """Scheme A2: signed ∂₂ / ∂₂ᵀ without boundary averaging."""

    def forward(
        self,
        edges,
        faces,
        edge_index,
        wire_index,
        face_index,
        edge_index_length,
        wire_index_length,
        adj_face_index_length,
        coedge_sign=None,
    ):
        h_coedge = edges
        h_face = faces
        device = h_coedge.device

        face_ids, coedge_ids, signs = build_face_coedge_pairs(
            wire_index,
            edge_index,
            wire_index_length,
            edge_index_length,
            device,
            coedge_sign=coedge_sign,
        )

        for layer_idx in range(self.num_layers):
            boundary_agg = scatter_boundary_to_faces_signed(
                h_coedge, face_ids, coedge_ids, signs, h_face.shape[0]
            )
            adj_agg = self._aggregate_adj_faces(h_face, face_index, adj_face_index_length)

            face_delta = self.face_update_layers[layer_idx](
                torch.cat([h_face, boundary_agg, adj_agg], dim=-1)
            )
            h_face = self.face_norms[layer_idx](h_face + face_delta)

            incident_face_agg = scatter_incident_faces_to_coedges_signed(
                h_face, face_ids, coedge_ids, signs, h_coedge.shape[0]
            )
            coedge_delta = self.edge_update_layers[layer_idx](
                torch.cat([h_coedge, incident_face_agg], dim=-1)
            )
            h_coedge = self.edge_norms[layer_idx](h_coedge + coedge_delta)

        adj_agg = self._aggregate_adj_faces(h_face, face_index, adj_face_index_length)
        return self.output_layer(torch.cat([h_face, adj_agg], dim=-1))
