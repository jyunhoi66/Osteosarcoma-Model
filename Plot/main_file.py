import timm
import Combined_model
from swinMM import SSLHead, load_pretrained_model
from gradcam import GradCam  # 确保导入的是你的 gradcam_v1
import numpy as np
import cv2
import glob
import os
import nibabel as nib
import torch
import torch.nn.functional as F
import random
import gc
import traceback

# ================= 环境配置 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def standardize_image_data(img, file_id="unknown"):
    """
    清洗 NIfTI 数据，统一输出 (Depth, Height, Width)
    """
    try:
        data = img.get_fdata()
    except Exception as e:
        print(f"[{file_id}] get_fdata() failed, trying strict loading... Error: {e}")
        data = np.asanyarray(img.dataobj)

    # 维度清洗
    while data.ndim > 3:
        if data.shape[-1] == 1:
            data = data.squeeze(-1)
        else:
            data = data[..., 0]

    # 补全 2D -> 3D
    if data.ndim == 2:
        data = data[..., np.newaxis]

    # 转置: (H, W, D) -> (D, H, W)
    if data.ndim == 3:
        data = data.transpose(2, 0, 1)

    return data


def resize_volume_to_96(img_data):
    """
    将任意尺寸的 (D, H, W) 缩放到 (96, 96, 96)
    """
    # 转 Tensor: (1, 1, D, H, W)
    tensor = torch.tensor(img_data).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # 标准化
    if tensor.std() > 0:
        tensor = (tensor - tensor.mean()) / tensor.std()
    else:
        tensor = tensor - tensor.mean()

    # Resize 到 96x96x96
    # 这一步是为了满足你之前提到的特征提取维度要求
    tensor = F.interpolate(tensor, size=(96, 96, 96), mode='trilinear', align_corners=False)

    # 压缩回 numpy (96, 96, 96)
    return tensor.squeeze().numpy()


def gen_cam_slice(image_slice, mask_slice):
    """生成单张切片的热力图叠加"""
    # 归一化原图
    image_slice = image_slice - np.min(image_slice)
    image_slice = image_slice / (np.max(image_slice) + 1e-8)
    image_rgb = np.stack([image_slice] * 3, axis=-1)

    # 归一化 Mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask_slice), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0

    # 叠加
    mask_slice = mask_slice[..., np.newaxis]
    result = (1 - mask_slice) * image_rgb + mask_slice * heatmap
    return np.uint8(np.clip(result * 255, 0, 255))


if __name__ == '__main__':
    # ================= 路径配置 =================
    input_folder = "/home/dell/PycharmProjects/item1/湘雅二医院/train/"
    output_root = "/home/dell/图片/GradCam/"
    # ===========================================

    os.makedirs(output_root, exist_ok=True)

    print("Loading Model...")
    model = Combined_model.CombinedModel(n_class=2, input_size=768).to(device)

    load_pretrained_model(model=model.modules1,
                          pretrained_path='/home/dell/PycharmProjects/HCC-CPS/骨肉瘤/swim transformer权重/pretrained_ckpt .pt',
                          prefix='module.')
    pretrained_state_dict = torch.load(
        '/home/dell/PycharmProjects/HCC-CPS/骨肉瘤/swim transformer权重/patient feature/单模态MLP/全幅图像/对的/best_model_fold1.pth',
        map_location=device)
    model.modules2.load_state_dict(pretrained_state_dict)

    # 目标层
    target_layer = model.modules1.swinViT.layers2[0].blocks[-1].norm2
    model.eval()

    nii_files = glob.glob(os.path.join(input_folder, "*.nii.gz"))
    print(f"Total files: {len(nii_files)}")

    for file_idx, file_path in enumerate(nii_files):
        file_id = os.path.basename(file_path).replace('.nii.gz', '')
        save_dir = os.path.join(output_root, file_id)
        os.makedirs(save_dir, exist_ok=True)

        print(f"[{file_idx + 1}/{len(nii_files)}] Processing: {file_id} ...")

        try:
            # 1. 读取数据 (D, H, W)
            img = nib.load(file_path)
            raw_data = standardize_image_data(img, file_id)  # 原始高清数据
            orig_depth, orig_h, orig_w = raw_data.shape

            # 2. Resize 到 96x96x96 (中间层)
            # volume_96 的 shape 是 (96, 96, 96)
            volume_96 = resize_volume_to_96(raw_data)

            # 3. 准备结果容器 (96, 96, 96)
            mask_3d_96 = np.zeros((96, 96, 96), dtype=np.float32)

            # 4. 【核心修复】逐切片运行 Grad-CAM (适配 2D 模型)
            # 我们遍历这 96 层深度，每一层都是一张 96x96 的 2D 图片
            for i in range(96):
                # 取出第 i 张切片: (96, 96)
                slice_2d = volume_96[i, :, :]

                # 转为 Tensor 并增加 Batch/Channel 维度: (1, 1, 96, 96)
                # 这完全符合你的 2D 模型输入要求
                input_tensor = torch.tensor(slice_2d).float().unsqueeze(0).unsqueeze(0).to(device)

                # 运行 Grad-CAM
                grad_cam = GradCam(model, target_layer)
                mask_2d = grad_cam(input_tensor)  # 返回 (96, 96)

                # 存入容器
                mask_3d_96[i, :, :] = mask_2d

                # 及时清理钩子
                grad_cam.remove_hooks()
                del grad_cam
                del input_tensor

            # 5. 将生成的 96x96x96 Mask 放大回原始尺寸
            # Mask (96, 96, 96) -> Tensor (1, 1, 96, 96, 96)
            mask_tensor = torch.tensor(mask_3d_96).float().unsqueeze(0).unsqueeze(0)

            # Interpolate 回 (1, 1, Orig_D, Orig_H, Orig_W)
            mask_resized = F.interpolate(mask_tensor,
                                         size=(orig_depth, orig_h, orig_w),
                                         mode='trilinear',
                                         align_corners=False)

            # 转回 Numpy (D, H, W)
            mask_final = mask_resized.squeeze().cpu().numpy()

            # 维度保护
            if mask_final.ndim == 2: mask_final = mask_final[np.newaxis, ...]

            # 6. 保存所有切片
            for i in range(orig_depth):
                # 取原始切片和生成的 Mask 切片
                img_slice = raw_data[i]
                mask_slice = mask_final[i]

                result = gen_cam_slice(img_slice, mask_slice)
                cv2.imwrite(os.path.join(save_dir, f'slice_{i}.jpg'), result)

            # 垃圾回收
            del volume_96
            del mask_3d_96
            del mask_tensor
            del mask_resized
            del raw_data
            del mask_final
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n!!! Error processing {file_id} !!!")
            traceback.print_exc()
            if 'grad_cam' in locals():
                try:
                    grad_cam.remove_hooks()
                except:
                    pass
            torch.cuda.empty_cache()
            continue

    print("All Done!")
