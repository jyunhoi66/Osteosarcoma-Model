# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep

def load_pretrained_model(model, pretrained_path, prefix='encoder.'):
    pretrained_state_dict = torch.load(pretrained_path,map_location="cpu")

    model_state_dict = model.state_dict()
    missing_keys = []
    unexpected_keys = []
    load_keys = []

    for pre_key in pretrained_state_dict['state_dict']:
        model_key = pre_key.replace(prefix, '')
        if model_key in model_state_dict:
            if model_state_dict[model_key].shape == pretrained_state_dict['state_dict'][pre_key].shape:
                model_state_dict[model_key] = pretrained_state_dict['state_dict'][pre_key]
                load_keys.append(model_key)
            else:
                print(model_state_dict[model_key].shape, pretrained_state_dict['state_dict'][pre_key].shape)
                missing_keys.append(model_key)
        else:
            unexpected_keys.append(model_key)
    
    model.load_state_dict(model_state_dict)
    print(f"成功加载了 {len(load_keys)} 个ViT参数权重")
    print("以下keys被加载 :", load_keys)

    # 打印未加载的keys
    if missing_keys:
        print("以下keys未被加载 (形状不匹配):", missing_keys)
    if unexpected_keys:
        print("以下keys未被加载 (在模型中未找到):", unexpected_keys)
    #print("以下keys被加载 :", load_keys)

class SSLHead(nn.Module):
    def __init__(self, dim=768, n_class=2):
        super(SSLHead, self).__init__()
        patch_size = ensure_tuple_rep(2, 3)
        window_size = ensure_tuple_rep(7, 3)
        self.swinViT = SwinViT(
            in_chans=1,
            embed_dim=48,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3,
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.layernorm = nn.LayerNorm(768)
        self.classify = nn.Linear(dim, n_class)

         # 冻结self.swinViT的权重
        for param in self.swinViT.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_out = self.swinViT(x.contiguous())[4]

        x = self.avgpool(x_out)
        x = x.view(x.size(0), -1)
        
        return  x

# if __name__ == "__main__":
#     model = SSLHead()
#     # model_dict = torch.load('./pretrained_ckpt.pt',map_location="cpu")
#     pretrained_path = '/home/dell/PycharmProjects/HCC-CPS/骨肉瘤/swim transformer权重/pretrained_ckpt .pt'
#     load_pretrained_model(model, pretrained_path, prefix='module.')
#
#     a = torch.rand(1,1,256,256,24)
#     #model.state_dict()['swinViT.layers3.0.blocks.1.attn.qkv.weight'].mean()
#     #tensor(-3.9093e-05)
#     out = model(a)
#     print(out.shape)
