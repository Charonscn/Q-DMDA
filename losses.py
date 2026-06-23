from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def _convert_to_onehot(labels, class_num=2):
    return np.eye(class_num)[labels]


def _class_weight(source_label, target_prob, class_num=2):
    batch_size = source_label.size(0)
    source_scalar = source_label.cpu().data.numpy()
    source_scalar = [x for row in source_scalar for x in row]
    source_vec = _convert_to_onehot(source_scalar, class_num=class_num)
    source_sum = np.sum(source_vec, axis=0).reshape(1, class_num)
    source_sum[source_sum == 0] = 100
    source_vec = source_vec / source_sum

    target_scalar = target_prob.cpu().data.max(1)[1].numpy()
    target_vec = target_prob.cpu().data.numpy()
    target_sum = np.sum(target_vec, axis=0).reshape(1, class_num)
    target_sum[target_sum == 0] = 100
    target_vec = target_vec / target_sum

    weight_ss = np.zeros((batch_size, batch_size))
    weight_tt = np.zeros((batch_size, batch_size))
    weight_st = np.zeros((batch_size, batch_size))

    source_classes = set(source_scalar)
    target_classes = set(target_scalar)
    count = 0
    for class_idx in range(class_num):
        if class_idx in source_classes and class_idx in target_classes:
            s_vec = source_vec[:, class_idx].reshape(batch_size, -1)
            t_vec = target_vec[:, class_idx].reshape(batch_size, -1)
            weight_ss += np.dot(s_vec, s_vec.T)
            weight_tt += np.dot(t_vec, t_vec.T)
            weight_st += np.dot(s_vec, t_vec.T)
            count += 1

    if count != 0:
        weight_ss /= count
        weight_tt /= count
        weight_st /= count
    else:
        weight_ss = np.array([0])
        weight_tt = np.array([0])
        weight_st = np.array([0])

    return (
        weight_ss.astype("float32"),
        weight_tt.astype("float32"),
        weight_st.astype("float32"),
    )


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    l2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    return sum(torch.exp(-l2_distance / bw) for bw in bandwidth_list)


def linear_mmd2(source, target):
    delta = source.float().mean(0) - target.float().mean(0)
    return delta.dot(delta.T)


def marginal(source, target, kernel_type="rbf", kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    if kernel_type == "linear":
        return linear_mmd2(source, target)
    if kernel_type != "rbf":
        raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    batch_size = int(source.size(0))
    kernels = guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    xx = torch.mean(kernels[:batch_size, :batch_size])
    yy = torch.mean(kernels[batch_size:, batch_size:])
    xy = torch.mean(kernels[:batch_size, batch_size:])
    yx = torch.mean(kernels[batch_size:, :batch_size])
    return torch.mean(xx + yy - xy - yx)


def conditional(source, target, source_label, target_prob,
                kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size(0)
    weight_ss, weight_tt, weight_st = _class_weight(source_label, target_prob)
    device = source.device
    weight_ss = torch.from_numpy(weight_ss).to(device=device, dtype=source.dtype)
    weight_tt = torch.from_numpy(weight_tt).to(device=device, dtype=source.dtype)
    weight_st = torch.from_numpy(weight_st).to(device=device, dtype=source.dtype)

    kernels = guassian_kernel(
        source,
        target,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )
    loss = source.new_tensor([0.0])
    if torch.sum(torch.isnan(sum(kernels))):
        return loss

    ss = kernels[:batch_size, :batch_size]
    tt = kernels[batch_size:, batch_size:]
    st = kernels[:batch_size, batch_size:]
    loss += torch.sum(weight_ss * ss + weight_tt * tt - 2 * weight_st * st)
    return loss


class LabelSmooth(nn.Module):
    def __init__(self, num_class: int, alpha: Optional[float] = 0.1,
                 device: Optional[str] = "cuda"):
        super().__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).to(self.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.alpha) * targets + self.alpha / self.num_class
        return (-targets * log_probs).sum(dim=1).mean()
