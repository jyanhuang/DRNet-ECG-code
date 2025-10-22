# base_pipeline.py
import create_json
import os
import os.path as osp
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from thop import profile
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class BasePipeline:
    def __init__(self, config: dict):
        self.config = config
        self.exp_name = config.get("exp_name") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.res_dir = osp.join(config["exp_dir"], self.exp_name, "results")
        os.makedirs(self.res_dir, exist_ok=True)

        self.model = self._init_net()
        self.criterion = nn.CrossEntropyLoss().to(config["device"])
        self.writer = SummaryWriter(log_dir=self.res_dir)

        self.pipeline_loader = self._init_dataloader()

        with open(config["mapping_json"]) as f:
            self.mapper = {v: k for k, v in json.load(f).items()}

        self._load_pruned_weight()

    # ------------------------------------------------------------------
    def _load_pruned_weight(self):
        ckpt_path = self.config.get("model_path")
        if not ckpt_path:
            raise FileNotFoundError("config 必须给出 model_path")
        from utils.prune_utils import rebuild_pruned_model

        state = torch.load(ckpt_path, map_location=self.config["device"])["state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model = rebuild_pruned_model(self.model, state)

    # ------------------------------------------------------------------
    def _init_net(self):
        from models.models1d import HeartNet
        from utils.prune_utils import rebuild_pruned_model

        model = HeartNet(num_classes=8).to(self.config["device"])
        ckpt_path = self.config["model_path"]
        state = torch.load(ckpt_path, map_location=self.config["device"])["state_dict"]
        model = rebuild_pruned_model(model, state)
        return model

    # ------------------------------------------------------------------
    def _init_dataloader(self):
        raise NotImplementedError

    # ==================================================================
    #  计算 FLOPs：稠密参考 + 真实稀疏 MAC
    # ==================================================================
    def count_flops(self, input_shape=(1, 1, 2048)):
        device = next(self.model.parameters()).device
        dummy = torch.randn(*input_shape).to(device)

        # 1. thop 稠密估算
        flops_dense, _ = profile(self.model, inputs=(dummy,), verbose=False)

        # 2. 一遍 forward 收集输出长度 & 统计稀疏 MAC
        sparse_flops = 0
        out_len = {}

        def make_hook(name):
            def _hook(mod, inp, out):
                if isinstance(out, torch.Tensor):
                    out_len[name] = out.shape[-1]
            return _hook

        handles = []
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                handles.append(m.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            _ = self.model(dummy)

        for h in handles:
            h.remove()

        # 3. 逐层统计真实 MAC（按非零权重）
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                weight = m.weight
                nnz = (weight != 0).sum().item()  # 非零元素
                if nnz == weight.numel():
                    continue  # 未剪枝，跳过（已在 flops_dense 里）
                L = out_len.get(name, input_shape[-1])

                if isinstance(m, nn.Conv1d):
                    k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
                    s = m.stride[0] if isinstance(m.stride, tuple) else m.stride
                    p = m.padding[0] if isinstance(m.padding, tuple) else m.padding
                    d = m.dilation[0] if isinstance(m.dilation, tuple) else m.dilation
                    L_out = (L + 2 * p - d * (k - 1) - 1) // s + 1
                    mac = nnz * L_out
                    if m.bias is not None:
                        mac += L_out
                elif isinstance(m, nn.Linear):
                    mac = nnz
                    if m.bias is not None:
                        mac += m.out_features
                else:
                    mac = 0
                sparse_flops += mac

        # 若没剪枝，稀疏值=稠密值
        if sparse_flops == 0:
            sparse_flops = flops_dense

        return flops_dense, sparse_flops

    # ==================================================================
    #  主评估流程（无 ROC）
    # ==================================================================
    def run_pipeline(self):
        # 参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        nonzero_params = sum((p != 0).sum() for p in self.model.parameters())
        print("========== Parameters ==========")
        print(f"Total params     : {total_params:,}")
        print(f"Non-zero params  : {nonzero_params:,}")
        print(f"Sparsity         : {(1 - nonzero_params / total_params) * 100:.2f}%")
        print("================================")
        flops_dense, flops_sparse = self.count_flops()
        print("========== FLOPs ==========")
        print(f"Dense  estimate : {int(flops_dense):,} FLOPs")
        print(f"Sparse real MACs: {int(flops_sparse):,} MACs")
        print(f"Compression rate: {flops_dense / (flops_sparse + 1e-9):.2f}x")
        print("===========================")
        self.writer.add_scalar("FLOPs/Dense", flops_dense, 0)
        self.writer.add_scalar("FLOPs/Sparse", flops_sparse, 0)

        self.model.eval()
        total_loss, total_time = 0.0, 0.0
        gt_class = np.empty(0, dtype=np.int64)
        pd_class = np.empty(0, dtype=np.int64)

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.pipeline_loader), desc="Running pipeline"):
                inputs = batch["image"].to(self.config["device"])
                targets = batch["class"].to(self.config["device"])

                start = time.time()
                preds = self.model(inputs)
                total_time += time.time() - start

                loss = self.criterion(preds, targets)
                total_loss += loss.item()

                classes = preds.argmax(dim=1).cpu().numpy()
                gt_class = np.concatenate((gt_class, batch["class"].numpy()))
                pd_class = np.concatenate((pd_class, classes))

                if i == 0:  # 抽查
                    for jj in range(min(3, len(inputs))):
                        print(f"[DEBUG] sample {jj}: true={targets[jj].item()} pred={classes[jj]}")

        # ---------------- 指标 ----------------
        total_loss /= len(self.pipeline_loader)
        acc = (pd_class == gt_class).mean()
        precision = precision_score(gt_class, pd_class, average='weighted', zero_division=0)
        recall = recall_score(gt_class, pd_class, average='weighted', zero_division=0)
        f1 = f1_score(gt_class, pd_class, average='weighted', zero_division=0)
        avg_time = total_time / len(self.pipeline_loader)

        print(f"Test loss - {total_loss:.4f}")
        print(f"Test accuracy - {acc:.4f}")
        print(f"Precision - {precision:.4f}")
        print(f"Recall - {recall:.4f}")
        print(f"F1-score - {f1:.4f}")
        print(f"Avg inference time / batch - {avg_time:.4f} s")

        # ---------------- 泄漏自检 ----------------
        self._leak_sanity_check(gt_class, pd_class)

    # ------------------------------------------------------------------
    def _leak_sanity_check(self, gt, pd):
        print("\n------------------ Leak-Sanity Check ------------------")
        print("Total samples :", len(gt))
        print("Per-class counts :", np.bincount(gt, minlength=9))
        print("Per-class acc  :",
              [f'{(pd[gt == i] == i).mean():.1%}' for i in range(9)])
        print("Confusion matrix (%) – row=true, col=pred:\n",
              (confusion_matrix(gt, pd, normalize='true') * 100).round(2))
        print("---------------------------------------------------------")