#!/usr/bin/env python3
"""Forward + backward smoke test for Scheme A (BoundaryOperatorTopoEncoder)."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.brt import BRT
from models.brt_segmentation import BRTSegmentation
from models.boundary_topo_encoder import BoundaryOperatorTopoEncoder


def make_synthetic_batch(
    faces_per_solid=(4, 6),
    wires_per_face=2,
    edges_per_wire=4,
    facet_len=8,
    edge_segments=3,
    d_edge=11,
    device: str = "cpu",
):
    batch_size = len(faces_per_solid)
    total_faces = sum(faces_per_solid)
    total_wires = total_faces * wires_per_face
    total_edges = total_wires * edges_per_wire

    face = torch.randn(total_faces, facet_len, 28, 4, device=device)
    tri_normal = torch.randn(total_faces, facet_len, 7, device=device)
    face_vis_mask = torch.ones(total_faces, facet_len, dtype=torch.bool, device=device)
    face_padding_mask = torch.ones(total_faces, facet_len, dtype=torch.bool, device=device)

    edge = torch.randn(total_edges, edge_segments, d_edge, 4, device=device)
    edge_padding_mask = torch.ones(total_edges, edge_segments, dtype=torch.bool, device=device)

    edge_index = torch.arange(total_edges, device=device).view(total_wires, edges_per_wire)
    wire_index = (
        torch.arange(total_wires, device=device).view(total_faces, wires_per_face)
    )
    adj_face_index = torch.zeros(total_faces, 3, dtype=torch.long, device=device)
    for f in range(total_faces):
        neighbors = [j for j in range(total_faces) if j != f][:3]
        if neighbors:
            adj_face_index[f, : len(neighbors)] = torch.tensor(neighbors, device=device)

    edge_index_length = torch.full((total_wires,), edges_per_wire, dtype=torch.long, device=device)
    wire_index_length = torch.full((total_faces,), wires_per_face, dtype=torch.long, device=device)
    adj_face_index_length = torch.full((total_faces,), 2, dtype=torch.long, device=device)
    num_faces_per_solid = torch.tensor(faces_per_solid, dtype=torch.long, device=device)

    return {
        "face": face,
        "tri_normal": tri_normal,
        "face_vis_mask": face_vis_mask,
        "face_padding_mask": face_padding_mask,
        "edge": edge,
        "edge_padding_mask": edge_padding_mask,
        "edge_index": edge_index,
        "wire_index": wire_index,
        "adj_face_index": adj_face_index,
        "edge_index_length": edge_index_length,
        "wire_index_length": wire_index_length,
        "adj_face_index_length": adj_face_index_length,
        "num_faces_per_solid": num_faces_per_solid,
        "batch_size": batch_size,
        "total_faces": total_faces,
    }


def test_boundary_topo_encoder():
    device = "cpu"
    d_model = 32
    encoder = BoundaryOperatorTopoEncoder(edge_dim=d_model, face_dim=d_model, hidden_dim=64).to(device)

    n_faces, n_edges = 5, 12
    h_face = torch.randn(n_faces, d_model, device=device, requires_grad=True)
    h_edge = torch.randn(n_edges, d_model, device=device, requires_grad=True)

    edge_index = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        device=device,
    )
    wire_index = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2], [1, 0]], device=device)
    adj_face_index = torch.tensor(
        [[1, 2, 0], [0, 2, 0], [0, 1, 0], [1, 4, 0], [3, 2, 0]],
        device=device,
    )
    edge_index_length = torch.tensor([4, 4, 4], device=device)
    wire_index_length = torch.tensor([2, 2, 2, 2, 2], device=device)
    adj_face_index_length = torch.tensor([2, 2, 2, 2, 1], device=device)

    out = encoder(
        h_edge,
        h_face,
        edge_index,
        wire_index,
        adj_face_index,
        edge_index_length,
        wire_index_length,
        adj_face_index_length,
    )
    assert out.shape == (n_faces, d_model), f"unexpected shape: {out.shape}"

    loss = out.sum()
    loss.backward()
    assert h_face.grad is not None and h_face.grad.abs().sum() > 0
    assert h_edge.grad is not None and h_edge.grad.abs().sum() > 0
    print("[ok] BoundaryOperatorTopoEncoder forward/backward")


def test_brt_core():
    device = "cpu"
    model = BRT(dmodel=32, hidden_dim=256, n_layers=1, n_heads=4, max_face_length=10).to(device)
    batch = make_synthetic_batch(faces_per_solid=(4,))

    out, mask = model(
        edge=batch["edge"],
        face=batch["face"],
        tri_normal=batch["tri_normal"],
        face_vis_mask=batch["face_vis_mask"],
        face_padding_mask=batch["face_padding_mask"],
        edge_padding_mask=batch["edge_padding_mask"],
        edge_index=batch["edge_index"],
        wire_index=batch["wire_index"],
        adj_face_index=batch["adj_face_index"],
        edge_index_length=batch["edge_index_length"],
        wire_index_length=batch["wire_index_length"],
        adj_face_index_length=batch["adj_face_index_length"],
        num_faces_per_solid=batch["num_faces_per_solid"],
    )
    assert out.shape[0] == 1
    assert out.shape[-1] == 32

    loss = out.sum()
    loss.backward()
    print("[ok] BRT core forward/backward")


def test_brt_segmentation():
    device = "cpu"
    model = BRTSegmentation(num_classes=8, d_model=32, max_face_length=10).to(device)
    batch = make_synthetic_batch()

    labels = torch.randint(0, 8, (batch["total_faces"],), device=device)
    inputs = {k: batch[k] for k in batch if k not in ("batch_size", "total_faces")}

    logits = model(inputs)
    assert logits.shape == (batch["total_faces"], 8), f"unexpected logits shape: {logits.shape}"
    assert labels.shape == (batch["total_faces"],)

    loss = F.cross_entropy(logits, labels)
    loss.backward()

    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "no gradients flowing"
    print(f"[ok] BRTSegmentation forward/backward, loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")


if __name__ == "__main__":
    test_boundary_topo_encoder()
    test_brt_core()
    test_brt_segmentation()
    print("All Scheme A smoke tests passed.")
