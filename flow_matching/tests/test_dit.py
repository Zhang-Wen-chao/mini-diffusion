"""DiT 结构单测（CPU）"""
import pytest
import torch

from flow_matching.config import ModelConfig
from flow_matching.dit import DiT

torch.manual_seed(0)


def make_cfg():
    return ModelConfig(hidden_size=128, num_heads=4, depth=2,
                       patch_size=4, image_size=32)


def test_dit_forward_shape():
    m = DiT(make_cfg())
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    out = m(x, t)
    assert out.shape == x.shape


def test_dit_time_conditioning():
    """adaLN-zero 下 t 不敏感（初始恒等）；调制置非零后 t 必须敏感"""
    m = DiT(make_cfg())
    x = torch.randn(1, 3, 32, 32)
    o1 = m(x, torch.tensor([0.0]))
    o2 = m(x, torch.tensor([1.0]))
    assert torch.allclose(o1, o2)      # adaLN-zero：初始输出与 t 无关

    for b in m.blocks:
        with torch.no_grad():
            b.adaLN_modulation[-1].weight.normal_(0, 0.1)
            b.adaLN_modulation[-1].bias.normal_(0, 0.1)
    with torch.no_grad():
        m.final.adaLN_modulation[-1].weight.normal_(0, 0.1)
        m.final.adaLN_modulation[-1].bias.normal_(0, 0.1)
    o1 = m(x, torch.tensor([0.0]))
    o2 = m(x, torch.tensor([1.0]))
    assert not torch.allclose(o1, o2)  # 调制生效后 t 影响输出


def test_dit_adaln_zero_init():
    """adaLN-zero：调制层零初始化 → 初始前向是恒等残差"""
    m = DiT(make_cfg())
    for b in m.blocks:
        w = b.adaLN_modulation[-1].weight
        assert w.abs().max().item() == 0.0
    w = m.final.adaLN_modulation[-1].weight
    assert w.abs().max().item() == 0.0


def test_dit_patchify_pos_embed_shapes():
    m = DiT(make_cfg())
    n = (32 // 4) ** 2
    assert m.n_patches == n == 64
    assert m.pos_embed.shape == (1, n, 128)


def test_dit_loss_backward():
    m = DiT(make_cfg())
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    v = torch.randn_like(x)
    loss = torch.nn.functional.mse_loss(m(x, t), v)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_dit_grad_flow_through_pos_embed():
    m = DiT(make_cfg())
    x = torch.randn(2, 3, 32, 32)
    m(x, torch.rand(2)).sum().backward()
    assert m.pos_embed.grad is not None
