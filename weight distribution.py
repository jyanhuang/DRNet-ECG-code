import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import torch


def load_weights(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, torch.nn.Module):
        state_dict = ckpt.state_dict()
    else:
        state_dict = ckpt

    weights = []
    for name, tensor in state_dict.items():
        if "weight" in name:
            weights.append(tensor.detach().cpu().numpy().flatten())
    if not weights:
        raise RuntimeError(f"No weights found in {model_path}!")
    return np.concatenate(weights)


def plot_near_zero(w, title, save_path):
    eps = 1e-6
    zero_cnt = np.sum(np.abs(w) < eps)
    non_zero = np.count_nonzero(w)
    print(f"{title} |w|<{eps}: {zero_cnt:,}")
    print(f"{title} 非零权重数: {non_zero:,}")

    plt.figure(figsize=(5, 4))
    bins = np.linspace(-0.2, 0.2, 100)
    plt.hist(w, bins=bins, density=False, color="tab:blue", alpha=0.7)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)

    # 关键：标题直接取自 save_path 的文件名（去掉 .png）
    plt.title(os.path.basename(save_path).replace('.png', ''))
    plt.xlabel("Weight value")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}\n")

def main():
    before_path = r"D:\python\python项目\ECG\ecg-classification-master\experiments\HeartNet1D副本\checkpoints\00000012.pth"
    after_path = r"D:\python\python项目\ECG\ecg-classification-master\experiments（block2+0.6+0.99prune）\HeartNet1D\checkpoints\00000019.pth"

    w_before = load_weights(before_path)
    w_after = load_weights(after_path)

    plot_near_zero(w_before, "Before pruning", "Weights distribution before pruning.png")
    plot_near_zero(w_after, "After pruning", "Weights distribution after pruning.png")


if __name__ == "__main__":
    main()