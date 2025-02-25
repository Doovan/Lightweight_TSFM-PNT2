import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel + 3  # 关键修复：包含坐标差特征
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
            
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        
        # 维度修正关键步骤
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, S]
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list,
                with_attn=False, with_edge=False, attn_heads=4):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.with_attn = with_attn
        self.with_edge = with_edge

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        # 修复1：确保每个MLP分支的输入通道正确
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            # 输入通道 = 原始特征通道 + 坐标差(3)
            branch_in = in_channel + 3
            
            # 修复2：正确构建每层的输入输出通道
            for j, out_channel in enumerate(mlp_list[i]):
                convs.append(nn.Conv2d(branch_in if j==0 else mlp_list[i][j-1], 
                                     out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                branch_in = out_channel
            
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)
            
            if with_edge:
                # 修复3：确保边缘卷积通道对齐
                setattr(self, f'edge_conv_{i}', EdgeConvBlock(mlp_list[i][-1], growth_rate=mlp_list[i][-1]))

        # 修复4：正确计算注意力总维度
        if with_attn:
            total_dim = 0
            for i in range(len(mlp_list)):
                branch_dim = mlp_list[i][-1]
                if with_edge:
                    branch_dim += mlp_list[i][-1]  # 边缘特征与原特征拼接
                total_dim += branch_dim
            self.attn = LocalAttention(total_dim, num_heads=attn_heads)

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
        new_points_list = []

        for i in range(len(self.radius_list)):
            radius = self.radius_list[i]
            nsample = self.nsample_list[i]  # 当前分支的 nsample
            
            # 原始特征提取
            group_idx = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, self.npoint, 1, C)
            
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            
            # 边缘卷积增强（修正维度顺序）
            if self.with_edge:
                edge_conv = getattr(self, f'edge_conv_{i}')
                edge_feat = edge_conv(
                    new_xyz.permute(0, 2, 1), 
                    grouped_points.max(dim=2)[0],  # [B, C, S]
                    k=nsample
                )  # [B, growth_rate, S, K]
                
                # 调整维度顺序为 [B, growth_rate, K, S]
                edge_feat = edge_feat.permute(0, 1, 3, 2)  # 关键修改
                grouped_points = torch.cat([grouped_points, edge_feat], dim=1)

            new_points = torch.max(grouped_points, 2)[0]
            new_points_list.append(new_points)

        new_points_concat = torch.cat(new_points_list, dim=1)
        
        # 多尺度注意力
        if self.with_attn:
            new_points_concat = self.attn(new_xyz.permute(0,2,1), new_points_concat)

        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class RelativePositionEncoder(nn.Module):
    """几何位置编码器（修复输入维度）"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 32),  # 输入维度修正为4 (3坐标差 + 1距离)
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )
        self.channel_align = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, pos, feat):
        feat = self.channel_align(feat.permute(0, 2, 1))  # [B, N, C']
        rel_pos = pos.unsqueeze(2) - pos.unsqueeze(1)      # [B, N, N, 3]
        rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # [B, N, N, 1]
        pos_enc = self.mlp(torch.cat([rel_pos, rel_dist], dim=-1))  # [B, N, N, 4]
        return pos_enc.permute(0, 3, 1, 2)  # [B, out_dim, N, N]

class LocalAttention(nn.Module):
    def __init__(self, channels, num_heads=8, radius=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = channels // num_heads
        self.radius = radius
        self.qkv = nn.Conv1d(channels, channels*3, 1)
        self.pos_encoder = RelativePositionEncoder(channels, self.d_k)
        assert channels % num_heads == 0, "channels必须能被num_heads整除"

    def forward(self, xyz, features):
        B, C, N = features.shape
        
        # ==== 维度对齐修复关键点 ====
        # 生成QKV [B, 3*C, N] -> [3, B, H, N, d_k]
        qkv = self.qkv(features).reshape(B, 3, self.num_heads, self.d_k, N).permute(1,0,2,4,3)
        q, k, v = qkv.unbind(0)  # 每个 [B, H, N, d_k]

        # 位置编码处理 [B, d_k, N, N] -> [B, H, N, N, 1]
        pos_code = self.pos_encoder(xyz.permute(0,2,1), features)  # [B, d_k, N, N]
        pos_code = pos_code.view(B, self.num_heads, self.d_k//self.num_heads, N, N)
        pos_code = pos_code.mean(dim=2).unsqueeze(-1)  # 关键修复：合并子维度 [B, H, N, N, 1]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)).unsqueeze(-1)  # [B, H, N, N, 1]
        attn = (attn + pos_code).squeeze(-1)  # [B, H, N, N]

        # 掩码处理
        dist_mat = torch.cdist(xyz.permute(0,2,1), xyz.permute(0,2,1))
        mask = (dist_mat < self.radius).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn = attn.masked_fill(~mask, float('-inf'))

        # Softmax和输出
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        return x

class EdgeConvBlock(nn.Module):
    # "动态边缘卷积"
    def __init__(self, in_channels, growth_rate=64):  # growth_rate 需等于对应分支的 mlp[-1]
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels*2, growth_rate, 1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU()
        )

    def forward(self, xyz, features, k):
        """
        xyz: [B, 3, N]
        features: [B, C, N]
        k: 邻域数（从 nsample_list 传入）
        """
        B, C, N = features.shape
        # KNN搜索
        dist = torch.cdist(xyz.permute(0, 2, 1), xyz.permute(0, 2, 1))  # [B, N, N]
        knn_idx = dist.topk(k, dim=-1, largest=False)[1]  # [B, N, K]
        
        # 构建边缘特征
        central_feat = features.unsqueeze(-1).repeat(1, 1, 1, k)  # [B, C, N, K]
        knn_feat = torch.gather(
            features.unsqueeze(2).expand(-1, -1, N, -1), 
            dim=3, 
            index=knn_idx.unsqueeze(1).expand(-1, C, -1, -1)
        )  # [B, C, N, K]
        edge_feat = torch.cat([central_feat, knn_feat - central_feat], dim=1)  # [B, 2C, N, K]
        
        return self.mlp(edge_feat)  # [B, growth_rate, N, K]