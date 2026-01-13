import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import BaseRoIExtractor
from mmcv.cnn import ConvModule, Linear, normal_init

def str2reg(input_str):
    """문자열에서 <bbox> 태그 내의 좌표를 추출"""
    bbox_regex = r'<bbox>\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)\s*</bbox>'
    results = []
    matches = re.findall(bbox_regex, input_str)
    for match in matches:
        results.append([float(match[0]), float(match[1]), float(match[2]), float(match[3])])
    return results

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MLVLFuseModule(nn.Module):
    def __init__(self, input_dims=1024, embed_dims=1024, num_levels=3, num_fuse=4):
        super(MLVLFuseModule, self).__init__()
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_fuse = num_fuse
        self.input_dims = input_dims
        self.shuffle_channles = embed_dims // 4

        self.fuse_lvl_list = []
        for lvl in range(num_levels):
            top_lvl = min(lvl + 1, num_levels - 1)
            dow_lvl = max(lvl - 1, 0)
            self.fuse_lvl_list.append((lvl, top_lvl, dow_lvl))

        self.remain_chs = self.embed_dims - self.shuffle_channles * 2
        self._init_layers()

    def _init_layers(self):
        self.input_conv = nn.ModuleList([nn.Conv2d(self.input_dims + 2, self.embed_dims, 1)
                                         for _ in range(self.num_levels)])
        self.fuse_convs = nn.ModuleList([
            ConvModule(self.embed_dims, self.embed_dims, 3, stride=1, padding=1,
                       conv_cfg=None, norm_cfg=dict(type='GN', num_groups=32, requires_grad=True))
            for _ in range(self.num_fuse)
        ])

    def generate_coordinate(self, featmap_sizes, device='cuda'):
        x_range = torch.linspace(-1, 1, featmap_sizes[-1], device=device)
        y_range = torch.linspace(-1, 1, featmap_sizes[-2], device=device)
        y, x = torch.meshgrid(y_range, x_range, indexing='ij')
        y = y.expand([featmap_sizes[0], 1, -1, -1])
        x = x.expand([featmap_sizes[0], 1, -1, -1])
        return torch.cat([x, y], 1)

    def _single_shuffle(self, inputs, conv_module):
        for single_conv_m in conv_module:
            fused_inputs = []
            for tar_lvl, top_lvl, dow_lvl in self.fuse_lvl_list:
                tar_input = inputs[tar_lvl]
                remain = tar_input[:, :self.remain_chs]
                
                # BFloat16 대응: Interpolate 시 float32로 변환
                from_top = inputs[top_lvl][:, self.remain_chs:][:, self.shuffle_channles:]
                from_top = F.interpolate(from_top.float(), size=tar_input.shape[-2:], 
                                         mode='bilinear', align_corners=True).to(tar_input.dtype)
                
                from_down = inputs[dow_lvl][:, self.remain_chs:][:, :self.shuffle_channles]
                from_down = F.interpolate(from_down.float(), size=tar_input.shape[-2:], 
                                          mode='bilinear', align_corners=True).to(tar_input.dtype)
                
                fused_inputs.append(torch.cat([remain, from_top, from_down], dim=1))
            inputs = [single_conv_m(item) for item in fused_inputs]
        return inputs

    def forward(self, inputs):
        new_inputs = []
        for feat in inputs:
            coord_feat = self.generate_coordinate(feat.shape, device=feat.device).to(feat.dtype)
            new_inputs.append(torch.cat([feat, coord_feat], dim=1))
        
        inputs = [self.input_conv[lvl](item) for lvl, item in enumerate(new_inputs)]
        for conv_m in self.fuse_convs:
            inputs = self._single_shuffle(inputs, [conv_m])
        return inputs

class MlvlRoIExtractor(BaseRoIExtractor):
    def __init__(self, roi_layer, out_channels, featmap_strides, embed_dims=1024, 
                 out_dims=4096, image_size=224):
        super(MlvlRoIExtractor, self).__init__(roi_layer, out_channels, featmap_strides)
        self.embed_dims = embed_dims
        self.image_size = image_size
        
        self.pconvs = nn.ModuleList([nn.Conv2d(self.embed_dims, self.embed_dims, 3, stride=1, padding=1)
                                     for _ in range(len(featmap_strides))])
        
        self.pos_embedd = nn.Sequential(
            nn.Linear(4, 256), nn.ReLU(inplace=True), nn.LayerNorm(256),
            nn.Linear(256, 1024), nn.ReLU(inplace=True), nn.LayerNorm(1024),
        )
        self.flatten_linear = nn.Linear(self.embed_dims * self.roi_layers[0].output_size[0] ** 2, 1024)
        self.updims = nn.Linear(1024, out_dims)

    def forward(self, feats, rois):
        num_imgs = len(rois)
        target_dtype = feats[0].dtype
        
        # 좌표 스케일링 및 배치 ID 부여
        new_rois = []
        for img_id, single_img_roi in enumerate(rois):
            single_img_roi = single_img_roi * self.image_size
            roi_img_id = single_img_roi.new_ones(len(single_img_roi)) * img_id
            new_rois.append(torch.cat([roi_img_id[:, None], single_img_roi], dim=1))
        
        rois_tensor = torch.cat(new_rois).float()
        pos_embedd = self.pos_embedd(torch.cat(rois).to(target_dtype))

        # Multi-level RoI Pooling
        roi_feats_list = []
        for i, feat in enumerate(feats):
            # RoIAlign은 float32에서 가장 안정적임
            roi_feat = self.roi_layers[i](feat.float(), rois_tensor).to(target_dtype)
            roi_feats_list.append(self.pconvs[i](roi_feat))

        # 특징 융합 및 차원 투영
        fuse_roi_feats = F.relu(sum(roi_feats_list)).flatten(1)
        fuse_roi_feats = self.flatten_linear(fuse_roi_feats)
        fuse_roi_feats = self.updims(fuse_roi_feats + pos_embedd)

        query_feats = []
        curr = 0
        for img_rois in rois:
            n = len(img_rois)
            query_feats.append(fuse_roi_feats[curr:curr+n])
            curr += n
        return query_feats

class MLVLROIQueryModule(nn.Module):
    def __init__(self, embed_dims=1024, out_dims=4096, num_levels=3, image_size=224):
        super(MLVLROIQueryModule, self).__init__()
        self.mlvl_fuse = MLVLFuseModule(input_dims=embed_dims, embed_dims=embed_dims, 
                                        num_levels=num_levels, num_fuse=5)
        
        # CLIP-ViT 특징맵 비율에 맞춘 스트라이드 설정
        strides = [14 / (2**i) for i in range(num_levels)][::-1]
        
        bbox_roi_extractor = dict(
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=2),
            out_channels=embed_dims,
            featmap_strides=strides,
            embed_dims=embed_dims,
            out_dims=out_dims,
            image_size=image_size
        )
        self.roi_align = MlvlRoIExtractor(**bbox_roi_extractor)

    def forward(self, mlvl_feats, bboxes):
        # 1. 시퀀스 형태([B, L, C])의 입력을 공간 형태([B, C, H, W])로 변환
        processed_feats = []
        for feat in mlvl_feats:
            if feat.dim() == 3:
                b, l, c = feat.shape
                h = w = int(math.sqrt(l))
                feat = feat.transpose(1, 2).reshape(b, c, h, w)
            processed_feats.append(feat)

        # 2. 멀티스케일 보간 (BFloat16 안전 처리)
        base_h, base_w = processed_feats[0].shape[-2:]
        interp_feats = []
        for i, feat in enumerate(processed_feats):
            target_size = (base_h * (2**i), base_w * (2**i))
            # todo: BF16 interpolation fix
            feat_fp32 = F.interpolate(feat.float(), size=target_size, 
                                      mode='bilinear', align_corners=True)
            interp_feats.append(feat_fp32.to(feat.dtype))

        # 3. 특징 융합 및 영역 쿼리 추출
        fused_feats = self.mlvl_fuse(interp_feats)
        return self.roi_align(fused_feats, bboxes)