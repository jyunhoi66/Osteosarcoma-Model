import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCam:
    def __init__(self, model, target):
        self.model = model.eval()
        self.feature = None
        self.gradient = None
        self.handlers = []
        self.target = target
        self._get_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = self.reshape_transform(output)

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = self.reshape_transform(output_grad)

        def _store_grad(grad):
            self.gradient = self.reshape_transform(grad)

        output_grad.register_hook(_store_grad)

    def _get_hook(self):
        h1 = self.target.register_forward_hook(self._get_features_hook)
        h2 = self.target.register_forward_hook(self._get_grads_hook)
        self.handlers.append(h1)
        self.handlers.append(h2)

    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()
        self.handlers = []

    def reshape_transform(self, tensor):
        # 1. 已经是 Spatial 格式 (B, C, H, W) 或 (B, C, D, H, W)
        if tensor.ndim >= 4:
            # 判断 Channel 是否在最后 (Swin 输出通常是 B, H, W, C)
            # 如果最后一维是 48, 96, 192, 384 等特征维度，或者比中间维度大
            if tensor.shape[-1] == max(tensor.shape) or tensor.shape[-1] > tensor.shape[1]:
                if tensor.ndim == 4: return tensor.permute(0, 3, 1, 2)
                if tensor.ndim == 5: return tensor.permute(0, 4, 1, 2, 3)
            return tensor

        # 2. Flatten 格式 (B, L, C)
        b, l, c = tensor.shape

        # 尝试 3D (立方体)
        side_3d = int(round(l ** (1 / 3)))
        if side_3d ** 3 == l:
            result = tensor.reshape(b, side_3d, side_3d, side_3d, c)
            result = result.permute(0, 4, 1, 2, 3)
            return result

        # 尝试 2D (正方形)
        side_2d = int(np.sqrt(l))
        if side_2d ** 2 == l:
            result = tensor.reshape(b, side_2d, side_2d, c)
            result = result.permute(0, 3, 1, 2)
            return result

        raise ValueError(f"Feature length {l} is neither square nor cube!")

    def __call__(self, inputs):
        self.model.zero_grad()
        output = self.model(inputs)

        probs = F.softmax(output, dim=1)
        index = np.argmax(probs.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()
        feature = self.feature[0].cpu().data.numpy()

        # --- 自动判断加权维度 ---
        if gradient.ndim == 4:  # 3D Feature: (C, D, H, W)
            weight = np.mean(gradient, axis=(1, 2, 3))
            cam = feature * weight[:, np.newaxis, np.newaxis, np.newaxis]
        elif gradient.ndim == 3:  # 2D Feature: (C, H, W)
            weight = np.mean(gradient, axis=(1, 2))
            cam = feature * weight[:, np.newaxis, np.newaxis]
        else:
            raise ValueError(f"Unknown gradient dimension: {gradient.ndim}")

        cam = np.sum(cam, axis=0)
        cam = np.maximum(cam, 0)

        # 归一化
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        # --- 自动 Resize 回输入尺寸 (修复了这里) ---

        # 情况 A: 输入是 4D (Batch, Channel, Height, Width) -> 2D 模式
        if inputs.ndim == 4:
            target_h, target_w = inputs.shape[2], inputs.shape[3]
            # cv2.resize 接受 (Width, Height)
            cam = cv2.resize(cam, (target_w, target_h))

        # 情况 B: 输入是 5D (Batch, Channel, Depth, Height, Width) -> 3D 模式
        elif inputs.ndim == 5:
            target_d, target_h, target_w = inputs.shape[2], inputs.shape[3], inputs.shape[4]
            # 使用 PyTorch interpolate 进行 3D Resize
            cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
            cam_tensor = F.interpolate(cam_tensor, size=(target_d, target_h, target_w), mode='trilinear',
                                       align_corners=False)
            cam = cam_tensor.squeeze().numpy()

        else:
            raise ValueError(f"Unsupported input dimension: {inputs.ndim}. Expected 4 or 5.")

        return cam