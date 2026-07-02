#!/usr/bin/env python3
"""Forward + backward smoke tests for Scheme A2 (signed ∂₂, no averaging)."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.brt import BRT
from models.brt_segmentation import BRTSegmentation
from models.boundary_topo_encoder import (
    BoundaryOperatorTopoEncoderA2,
    build_face_coedge_pairs,
    scatter_boundary_to_faces_signed,
)


def test_build_face_coedge_pairs_inner_wire_sign():
    device = "cpu"
    edge_index = torch.tensor([[0, 1], [2, 3]], device=device)
    wire_index = torch.tensor([[0, 1]], device=device)
    edge_index_length = torch.tensor([2, 2], device=device)
    wire_index_length = torch.tensor([2], device=device)

    face_ids, coedge_ids, signs = build_face_coedge_pairs(
        wire_index, edge_index, wire_index_length, edge_index_length, device
    )
    assert face_ids.tolist() == [0, 0, 0, 0]
    assert coedge_ids.tolist() == [0, 1, 2, 3]
    assert signs.tolist() == [1.0, 1.0, -1.0, -1.0]


def test_signed_boundary_no_average():
    device = "cpu"
    h = torch.tensor([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]], device=device)
    face_ids = torch.tensor([0, 0, 0], device=device)
    coedge_ids = torch.tensor([0, 1, 2], device=device)
    signs = torch.tensor([1.0, 1.0, -1.0], device=device)
    agg = scatter_boundary_to_faces_signed(h, face_ids, coedge_ids, signs, 1)
    assert agg.shape == (1, 2)
    assert torch.allclose(agg[0, 0], torch.tensor(-1.0))


def test_boundary_topo_encoder_a2():
    device = "cpu"
    d_model = 32
    encoder = BoundaryOperatorTopoEncoderA2(edge_dim=d_model, face_dim=d_model, hidden_dim=64).to(device)

    n_faces, n_coedges = 5, 12
    h_face = torch.randn(n_faces, d_model, device=device, requires_grad=True)
    h_coedge = torch.randn(n_coedges, d_model, device=device, requires_grad=True)
    coedge_sign = torch.tensor([1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1], dtype=torch.float32, device=device)

    edge_index = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], device=device)
    wire_index = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2], [1, 0]], device=device)
    adj_face_index = torch.tensor(
        [[1, 2, 0], [0, 2, 0], [0, 1, 0], [1, 4, 0], [3, 2, 0]],
        device=device,
    )
    edge_index_length = torch.tensor([4, 4, 4], device=device)
    wire_index_length = torch.tensor([2, 2, 2, 2, 2], device=device)
    adj_face_index_length = torch.tensor([2, 2, 2, 2, 1], device=device)

    out = encoder(
        h_coedge,
        h_face,
        edge_index,
        wire_index,
        adj_face_index,
        edge_index_length,
        wire_index_length,
        adj_face_index_length,
        coedge_sign=coedge_sign,
    )
    assert out.shape == (n_faces, d_model)

    loss = out.sum()
    loss.backward()
    assert h_face.grad is not None and h_face.grad.abs().sum() > 0
    assert h_coedge.grad is not None and h_coedge.grad.abs().sum() > 0
    print("[ok] BoundaryOperatorTopoEncoderA2 forward/backward")


def make_synthetic_batch(**kwargs):
    from tests.test_scheme_a_forward import make_synthetic_batch as _make

    return _make(**kwargs)


def test_brt_segmentation_a2():
    device = "cpu"
    model = BRTSegmentation(num_classes=8, d_model=32, max_face_length=10).to(device)
    batch = make_synthetic_batch()
    batch["coedge_sign"] = torch.ones(batch["edge"].shape[0], dtype=torch.float32)

    labels = torch.randint(0, 8, (batch["total_faces"],), device=device)
    inputs = {k: batch[k] for k in batch if k not in ("batch_size", "total_faces")}

    logits = model(inputs)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    print(f"[ok] BRTSegmentation A2 forward/backward, loss={loss.item():.4f}")


if __name__ == "__main__":
    test_build_face_coedge_pairs_inner_wire_sign()
    test_signed_boundary_no_average()
    test_boundary_topo_encoder_a2()
    test_brt_segmentation_a2()
    print("All Scheme A2 smoke tests passed.")
