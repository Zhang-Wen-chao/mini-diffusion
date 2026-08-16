#!/usr/bin/env python3
"""DeepSpeed 入口（deepspeed --num_gpus=N train_ds_entry.py --zero 3）"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# deepspeed 0.19.5 与 nvtx>=0.2.8 的 DummyDomain API 不兼容（push_range 签名变化）
# 直接 patch 掉 profiling 调用
try:
    import nvtx
    import nvtx._lib.lib as _nvtx_lib

    _orig = _nvtx_lib.DummyDomain.push_range

    def _patched(self, *args, **kwargs):
        try:
            return _orig(self, *args, **kwargs)
        except TypeError:
            return None

    _nvtx_lib.DummyDomain.push_range = _patched
except (ImportError, AttributeError):
    pass

from flow_matching.train_ds import main  # noqa: E402

if __name__ == "__main__":
    main()
