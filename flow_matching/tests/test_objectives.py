"""CPU 单元测试：目标函数数学正确性 + 模型结构 + 采样基本性质"""
import pytest
import torch

from flow_matching.config import ModelConfig, TrainConfig, DDPMConfig
from flow_matching.model import UNet, sinusoidal_embedding
from flow_matching.objectives import DDPMObjective, FlowObjective
from flow_matching.sampler import sample_euler

torch.manual_seed(0)


def make_cfg(objective="flow"):
    return TrainConfig(objective=objective, max_steps=10,
                       model=ModelConfig(base_channels=32))


# ── timestep embedding ──────────────────────────────────────────
def test_sinusoidal_embedding_shape():
    t = torch.tensor([0.0, 0.5, 1.0])
    e = sinusoidal_embedding(t, 64)
    assert e.shape == (3, 64)


# ── model ───────────────────────────────────────────────────────
def test_unet_forward_shape():
    m = UNet(make_cfg().model)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    out = m(x, t)
    assert out.shape == x.shape


def test_unet_time_conditioning_changes_output():
    m = UNet(make_cfg().model)
    x = torch.randn(1, 3, 32, 32)
    o1 = m(x, torch.tensor([0]))
    o2 = m(x, torch.tensor([500]))
    assert not torch.allclose(o1, o2)


# ── DDPM objective ──────────────────────────────────────────────
def test_ddpm_endpoints():
    obj = DDPMObjective(make_cfg("ddpm"))
    x = torch.randn(4, 3, 32, 32)
    x0, eps0 = obj.add_noise(x, torch.zeros(4, dtype=torch.long))
    # t=0: 几乎数据。x0 = √ᾱ_0·x + √(1-ᾱ_0)·ε，噪声项 std≈0.01
    assert (x0 - x).std() < 0.02
    # 且与"正确公式"逐元素一致（确定性验证）
    a0 = obj.alpha_bar[0]
    assert torch.allclose(x0, a0.sqrt() * x + (1 - a0).sqrt() * eps0, atol=1e-5)
    xT, epsT = obj.add_noise(x, torch.full((4,), obj.num_timesteps - 1, dtype=torch.long))
    assert torch.allclose(xT, epsT, atol=5e-2)                 # t=T: 几乎纯噪声（√ᾱ_T·x 残余 ~0.013）
    assert (xT - epsT).std() < 0.02
    assert x0.shape == epsT.shape == x.shape


def test_ddpm_predict_x0_roundtrip():
    obj = DDPMObjective(make_cfg("ddpm"))
    x = torch.randn(2, 3, 8, 8)
    t = torch.tensor([100, 500])
    xt, eps = obj.add_noise(x, t)
    x0_rec = obj.predict_x0(xt, t, eps)
    assert torch.allclose(x0_rec, x, atol=1e-4)                # 预测正确则还原 x0


def test_ddpm_loss_scalar():
    m = UNet(make_cfg("ddpm").model)
    obj = DDPMObjective(make_cfg("ddpm"))
    loss = obj.loss(m, torch.randn(2, 3, 32, 32), torch.randint(0, 1000, (2,)))
    assert loss.ndim == 0 and torch.isfinite(loss)


# ── Flow objective ──────────────────────────────────────────────
def test_flow_endpoints():
    obj = FlowObjective(make_cfg("flow"))
    x = torch.randn(4, 3, 8, 8)
    x1, _ = obj.add_noise(x, torch.ones(4))                    # t=1: 数据
    assert torch.allclose(x1, x, atol=1e-5)
    x0, z0 = obj.add_noise(x, torch.zeros(4))                  # t=0: 噪声（同一次调用的 z）
    assert torch.allclose(x0, z0, atol=1e-5)


def test_flow_velocity_linear_path():
    obj = FlowObjective(make_cfg("flow"))
    x = torch.randn(4, 3, 8, 8)
    t = torch.rand(4)
    xt, z = obj.add_noise(x, t)
    v = obj.velocity(x, z)
    assert torch.allclose(xt + (1 - t[:, None, None, None]) * v, x, atol=1e-5)


def test_flow_loss_scalar():
    m = UNet(make_cfg("flow").model)
    obj = FlowObjective(make_cfg("flow"))
    loss = obj.loss(m, torch.randn(2, 3, 32, 32), torch.rand(2))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_flow_t_samplers():
    for name in ["uniform", "midpoint", "logit_normal"]:
        cfg = make_cfg("flow")
        cfg.flow_t_sampler = name
        obj = FlowObjective(cfg)
        t = obj.sample_t(64, "cpu")
        assert t.shape == (64,) and t.min() >= 0 and t.max() <= 1
        assert torch.isfinite(t).all()


# ── sampler ─────────────────────────────────────────────────────
def test_euler_recovers_straight_line():
    """常数速度场（指向 data）→ 1 步欧拉精确还原数据（直线=1步采样可行）"""
    torch.manual_seed(1)
    z = torch.randn(2, 3, 8, 8)
    data = torch.rand(2, 3, 8, 8) * 2 - 1

    # 完美的速度场：v(x) = data - x（沿直线匀速指向 data）
    class ConstantField(torch.nn.Module):
        def forward(self, x, t):
            return data - x

    out = sample_euler(ConstantField(), None, z.shape, steps=1, device="cpu")
    assert torch.allclose(out, data, atol=1e-4)
