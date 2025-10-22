import os
import os.path as osp
from datetime import datetime
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.utils.prune as prune
from utils.network_utils import load_checkpoint, save_checkpoint
from importlib import import_module
import copy
import io
from thop import profile
class BaseTrainer:
    # ===================================================================
    # 1. 构造 & 初始化
    # ===================================================================
    def __init__(self, config):
        self.config = config
        self.exp_name = self.config.get("exp_name", None)
        if self.exp_name is None:
            self.exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.log_dir = osp.join(self.config["exp_dir"], self.exp_name, "logs")
        self.pth_dir = osp.join(self.config["exp_dir"], self.exp_name, "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.pth_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        print("Initializing trainer...")
        self.model = self._init_model()
        print("Model initialized.")
        self.optimizer = self._init_optimizer()
        self.criterion = nn.CrossEntropyLoss().to(self.config["device"])

        # 真正剪枝
        if self.config.get('prune', True):
            self._prune_model()
            print("Model pruned and pruned model saved.")

        self.train_loader, self.val_loader = self._init_dataloaders()

        # 可选加载预训练
        pretrained_path = self.config.get("model_path", False)
        if pretrained_path:
            self.training_epoch, self.total_iter = load_checkpoint(
                pretrained_path, self.model, optimizer=self.optimizer,
            )
        else:
            self.training_epoch = 0
            self.total_iter = 0

        self.epochs = self.config.get("epochs", int(1e5))

    # ===================================================================
    # 2. 稀疏层定义（内部类）
    # ===================================================================
    class MaskedConv1d(nn.Module):
        def __init__(self, out_ch, in_ch, ks, weight1d, bias,
                     idx0, idx1, idx2,
                     stride=1, padding=0, dilation=1, groups=1):
            super().__init__()
            self.out_channels = out_ch
            self.in_channels = in_ch
            self.kernel_size = ks
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups

            self.register_parameter('weight_pruned', nn.Parameter(weight1d))
            self.register_parameter('bias', nn.Parameter(bias) if bias is not None else None)
            self.register_buffer('idx0', idx0)
            self.register_buffer('idx1', idx1)
            self.register_buffer('idx2', idx2)

        def forward(self, x):
            W = torch.zeros(self.out_channels, self.in_channels, self.kernel_size,
                            dtype=self.weight_pruned.dtype,
                            device=self.weight_pruned.device)
            W[self.idx0, self.idx1, self.idx2] = self.weight_pruned
            return nn.functional.conv1d(x, W, self.bias,
                                        stride=self.stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        groups=self.groups)

    class MaskedLinear(nn.Module):
        def __init__(self, in_f, out_f, weight1d, bias, row_idx, col_idx):
            super().__init__()
            self.in_features = in_f
            self.out_features = out_f
            self.register_parameter('weight_pruned', nn.Parameter(weight1d))
            self.register_parameter('bias', nn.Parameter(bias) if bias is not None else None)
            self.register_buffer('row_idx', row_idx)
            self.register_buffer('col_idx', col_idx)

        def forward(self, x):
            W = torch.zeros(self.out_features, self.in_features,
                            dtype=self.weight_pruned.dtype,
                            device=self.weight_pruned.device)
            W[self.row_idx, self.col_idx] = self.weight_pruned
            return nn.functional.linear(x, W, self.bias)

    # ===================================================================
    # 3. 工具：递归拿到父模块
    # ===================================================================
    @staticmethod
    def _get_parent_and_attr(root, name):
        names = name.split('.')
        parent = root
        for n in names[:-1]:
            parent = getattr(parent, n)
        return parent, names[-1]

    # ===================================================================
    # 4. 真正的剪枝函数
    # ===================================================================
    def _prune_model(self):
        print("Pruning model (real sparsity)...")

        def count_params(m):
            return sum(p.numel() for p in m.parameters()), \
                sum((p != 0).sum().item() for p in m.parameters())

        total_before, nz_before = count_params(self.model)
        print(f"Before - Total: {total_before}, Non-zero: {nz_before}")
        device = next(self.model.parameters()).device  # 自动取模型所在设备
        dummy_input = torch.randn(1, 1, 2048, device=device)
        flops, _ = profile(self.model, inputs=(dummy_input,), verbose=False)
        print("Total FLOPs:", flops)

        # 1. 先 magnitude pruning 并立即 remove，把 mask 写回原 weight
        for _, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                prune.l1_unstructured(m, name='weight', amount=0.9)
                prune.remove(m, 'weight')  # 关键：把 mask 永久化

        # 2. 现在 state_dict 和新建模型键名完全一致，可直接 deepcopy（或 state-dict 重建）
        try:
            pruned_model = copy.deepcopy(self.model)
        except Exception:  # 若仍怕 deepcopy 失败，用 state-dict 重建
            buffer = io.BytesIO()
            torch.save(self.model.state_dict(), buffer)
            buffer.seek(0)
            pruned_model = self._init_model()
            pruned_model.load_state_dict(torch.load(buffer))

        # 3. 逐层替换为稀疏层
        for name, m in pruned_model.named_modules():
            if isinstance(m, nn.Linear):
                mask = m.weight != 0
                nz_idx = mask.nonzero(as_tuple=False)
                rows, cols = nz_idx[:, 0], nz_idx[:, 1]
                w1d = m.weight[rows, cols].clone()
                b = m.bias.clone() if m.bias is not None else None
                new_layer = self.MaskedLinear(m.in_features, m.out_features,
                                              w1d, b, rows, cols)
                parent, attr = self._get_parent_and_attr(pruned_model, name)
                setattr(parent, attr, new_layer)

            elif isinstance(m, nn.Conv1d):
                mask = m.weight != 0
                nz_idx = mask.nonzero(as_tuple=False)
                idx0, idx1, idx2 = nz_idx[:, 0], nz_idx[:, 1], nz_idx[:, 2]
                w1d = m.weight[idx0, idx1, idx2].clone()
                b = m.bias.clone() if m.bias is not None else None
                new_layer = self.MaskedConv1d(
                    m.out_channels, m.in_channels, m.kernel_size[0],
                    w1d, b, idx0, idx1, idx2,
                    stride=m.stride[0],  # 关键
                    padding=m.padding[0],
                    dilation=m.dilation[0],
                    groups=m.groups
                )
                parent, attr = self._get_parent_and_attr(pruned_model, name)
                setattr(parent, attr, new_layer)

        total_after, nz_after = count_params(pruned_model)
        print(f"After  - Total: {total_after}, Non-zero: {nz_after}")
        device = next(pruned_model.parameters()).device  # 自动取模型所在设备
        dummy_input = torch.randn(1, 1, 2048, device=device)
        flops, _ = profile(pruned_model, inputs=(dummy_input,), verbose=False)
        print("Total FLOPs:", flops)

        ckpt_path = osp.join(self.pth_dir, "pruned_model_checkpoint.pth")
        torch.save({
            "state_dict": pruned_model.state_dict(),
            "optimizer": None,
            "epoch": 0,
            "total_iter": 0,
        }, ckpt_path)
        print(f"Pruned model saved to {ckpt_path}")

        self.model = pruned_model.to(self.config["device"])
        self.optimizer = self._init_optimizer()


    def _init_optimizer(self):
        optimizer = getattr(optim, self.config["optim"])(self.model.parameters(), **self.config["optim_params"])
        return optimizer

    def _init_model(self):
        print("Initializing model...")
        model_module = import_module("models.models1d")
        model_class = getattr(model_module, self.config["model"])
        model = model_class(num_classes=self.config["num_classes"])
        model = model.to(self.config["device"])

        return model

    def _init_dataloaders(self):
        raise NotImplemented

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        gt_class = np.empty(0)
        pd_class = np.empty(0)

        start_time = time.time()  # 开始计时
        for i, batch in enumerate(self.train_loader):
            inputs = batch["image"].to(self.config["device"])
            targets = batch["class"].to(self.config["device"])

            predictions = self.model(inputs)
            loss = self.criterion(predictions, targets)

            classes = predictions.topk(k=1)[1].view(-1).cpu().numpy()

            gt_class = np.concatenate((gt_class, batch["class"].numpy()))
            pd_class = np.concatenate((pd_class, classes))

            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 10 == 0:
                print("\tIter [%d/%d] Loss: %.4f" % (i + 1, len(self.train_loader), loss.item()))

            self.writer.add_scalar("Train loss (iterations)", loss.item(), self.total_iter)
            self.total_iter += 1

        end_time = time.time()  # 结束计时
        inference_time = end_time - start_time  # 计算推理时间

        total_loss /= len(self.train_loader)
        class_accuracy = sum(pd_class == gt_class) / pd_class.shape[0]

        print("Train loss - {:4f}".format(total_loss))
        print("Train CLASS accuracy - {:4f}".format(class_accuracy))
        print("Train inference time - {:.2f}s".format(inference_time))  # 打印推理时间

        self.writer.add_scalar("Train loss (epochs)", total_loss, self.training_epoch)
        self.writer.add_scalar("Train CLASS accuracy", class_accuracy, self.training_epoch)
        self.writer.add_scalar("Train inference time", inference_time, self.training_epoch)

    def val(self):
        self.model.eval()
        total_loss = 0
        gt_class = np.empty(0)
        pd_class = np.empty(0)

        start_time = time.time()  # 开始计时
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.val_loader)):
                inputs = batch["image"].to(self.config["device"])
                targets = batch["class"].to(self.config["device"])

                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)

                classes = predictions.topk(k=1)[1].view(-1).cpu().numpy()

                gt_class = np.concatenate((gt_class, batch["class"].numpy()))
                pd_class = np.concatenate((pd_class, classes))

                total_loss += loss.item()

        end_time = time.time()  # 结束计时
        inference_time = end_time - start_time  # 计算推理时间

        total_loss /= len(self.val_loader)
        class_accuracy = sum(pd_class == gt_class) / pd_class.shape[0]

        print("Validation loss - {:4f}".format(total_loss))
        print("Validation CLASS accuracy - {:4f}".format(class_accuracy))
        print("Validation inference time - {:.2f}s".format(inference_time))  # 打印推理时间

        self.writer.add_scalar("Validation loss", total_loss, self.training_epoch)
        self.writer.add_scalar("Validation CLASS accuracy", class_accuracy, self.training_epoch)
        self.writer.add_scalar("Validation inference time", inference_time, self.training_epoch)


    def loop(self):
        for epoch in range(self.training_epoch, self.epochs):
            print("Epoch - {}".format(self.training_epoch + 1))
            self.train_epoch()
            save_checkpoint(
                {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "total_iter": self.total_iter,
                },
                osp.join(self.pth_dir, "{:0>8}.pth".format(epoch)),
            )
            self.val()
            self.training_epoch += 1