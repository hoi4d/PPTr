import random

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from point_4d_convolution import *
from transformer import *
from pst_convolutions import *
from pointnet import pointnet




class PrimitiveTransformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, num_classes):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.emb_relu1 = nn.ReLU()
        self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.emb_relu2 = nn.ReLU()
        self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim)


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, input):


        # 4d BACKBONE
        # [B, L, N, 3]
        xyzs, features = self.tube_embedding(input)  # [B, L, n, 3], [B, L, C, n]

        features = features.transpose(2, 3)  # B ,L , n, C
        B, L, N, C = features.size()

        raw_feat = features

        point_feat = torch.reshape(input=raw_feat, shape=(B * L * 8, -1, C))  # [B*L*4, n', C]
        point_feat = self.emb_relu1(point_feat)
        point_feat = self.transformer1(point_feat)  # [B*L*4, n', C]

        primitive_feature = point_feat.permute(0, 2, 1)
        primitive_feature = F.adaptive_max_pool1d(primitive_feature, (1))  # B*l*4, C, 1
        primitive_feature = torch.reshape(input=primitive_feature, shape=(B, L * 8, C))  # [B, L*4, C]



        primitive_feature = self.emb_relu2(primitive_feature)
        primitive_feature = self.transformer2(primitive_feature) # B. L*4, C

        output = torch.max(input=primitive_feature, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)

        return output



class LongPrimitiveTransformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, num_classes):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.pointnet = pointnet()

        self.emb_relu1 = nn.ReLU()
        self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.emb_relu2 = nn.ReLU()
        self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim)


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, input):
        device = input.get_device()
       # 3d BACKBONE
        input_copy = input.clone()
        B1, L1, N1, C1 = input_copy.size()

        S = 20
        index = torch.tensor(random.sample(range(L1), S), device=device)
        index = index.long()
        input_copy = torch.index_select(input_copy, 1, index)


        B1, L1, N1, C1 = input_copy.size()

        pointnet_input = torch.reshape(input=input_copy, shape=(B1 * L1, N1, C1))


        pointnet_output = self.pointnet(pointnet_input.transpose(1, 2))

        BL, N2, C2 = pointnet_output.size()

        pointnet_output = torch.reshape(input=pointnet_output, shape=(BL*4, -1, C2))  # [B*L*4, n', C]

        # pointnet_output = self.emb_relu1(pointnet_output)
        # pointnet_output = self.transformer1(pointnet_output)  # [B*L*4, n', C]

        pointnet_output = pointnet_output.permute(0, 2, 1)
        pointnet_output = F.adaptive_max_pool1d(pointnet_output, (1))  # B*l*4, C, 1
        pointnet_output = torch.reshape(input=pointnet_output, shape=(B1, L1 * 4, C2))  # [B, L*4, C]



       # 4d BACKBONE
        # [B, L, N, 3]
        xyzs, features = self.tube_embedding(input)  # [B, L, n, 3], [B, L, C, n]

        features = features.transpose(2, 3)  # B ,L , n, C
        B, L, N, C = features.size()

        raw_feat = features

        point_feat = torch.reshape(input=raw_feat, shape=(B * L * 4, -1, C))  # [B*L*4, n', C]
        point_feat = self.emb_relu1(point_feat)
        point_feat = self.transformer1(point_feat)  # [B*L*4, n', C]

        primitive_feature = point_feat.permute(0, 2, 1)
        primitive_feature = F.adaptive_max_pool1d(primitive_feature, (1))  # B*l*4, C, 1
        primitive_feature = torch.reshape(input=primitive_feature, shape=(B, L * 4, C))  # [B, L*4, C]


        # Integrate
        primitive_feature = torch.cat((primitive_feature, pointnet_output), dim=1)


        primitive_feature = self.emb_relu2(primitive_feature)
        output = self.transformer2(primitive_feature)[:,:L*4,:]  # B. L*4, C

        output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
        output = self.mlp_head(output)

        return output
