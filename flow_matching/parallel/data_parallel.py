"""数据并行：梯度 all-reduce + TP 未切分层梯度缩放"""
import torch.distributed as dist


def allreduce_grads(model_or_params, dp_group):
    if dp_group is None or dist.get_world_size(dp_group) <= 1:
        return
    params = (model_or_params if isinstance(model_or_params, list)
              else list(model_or_params.parameters()))
    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad, group=dp_group)
            param.grad.div_(dist.get_world_size(dp_group))


def scale_tp_untouched_grads(model, tp_size):
    """TP 下未切分层（adaLN/norm/patchify 等）的梯度重复计算了 tp 次，除以 tp_size

    Megatron 规则：跨 rank 复制的层，梯度需要缩放 1/tp_size。
    """
    if tp_size <= 1:
        return
    sharded = getattr(model, "_tp_sharded", set())
    for name, p in model.named_parameters():
        if p.grad is not None and name not in sharded:
            p.grad.div_(tp_size)
