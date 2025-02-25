import torch.nn as nn
import torch
import torch.nn.functional as F
#from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation
from pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super().__init__()
        self.normal_channel = normal_channel
        self.num_classes = num_classes
        # 新增输出层定义
        self.conv_seg = nn.Conv1d(128, num_classes, 1)  # 关键修复：添加分割输出层
        self.conv_norm = nn.Conv1d(128, 3, 1)          # 新增法向量预测层

        # 修正各层通道数对应关系
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[16, 32, 128],
            in_channel=0,
            mlp_list=[
                [32, 32, 64],
                [64, 64, 128],
                [64, 96, 128]
            ]
        )
        
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.2, 0.4, 0.8],
            nsample_list=[32, 64, 128],
            in_channel=64+128+128,  # 320
            mlp_list=[
                [128, 128, 256],
                [256, 256, 512],
                [256, 384, 512]
            ]
        )
        
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256+512+512,  # 1280
            mlp=[1024, 1024, 2048],
            group_all=True
        )
        
        self.fp3 = PointNetFeaturePropagation(1280+2048, [1024, 512])
        self.fp2 = PointNetFeaturePropagation(320+512, [512, 256])
        self.fp1 = PointNetFeaturePropagation(256+3+1, [128, 128])  # 输入坐标+标签

    def forward(self, xyz, cls_label):
        B, C, N = xyz.shape
        
        # 输入处理修正
        l0_xyz = xyz[:, :3, :] if self.normal_channel else xyz
        l0_points = xyz if self.normal_channel else None
        
        # 前向传播修正
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # 特征传播维度对齐
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        # 最终特征拼接修正
        cls_label_one_hot = cls_label.view(B, 1, 1).repeat(1, 1, N)
        l0_points = self.fp1(l0_xyz, l1_xyz, 
                           torch.cat([cls_label_one_hot, l0_xyz], 1),
                           l1_points)
        
        # 输出层
        seg_logits = F.log_softmax(self.conv_seg(l0_points), dim=1)
        norm_pred = self.conv_norm(l0_points).transpose(1,2)  # 新增法向量输出
        # 修改返回值匹配训练代码的输入参数
        return seg_logits.permute(0, 2, 1), l3_points  # 仅返回分割结果和特征矩阵
    def get_loss(self):
        return get_loss(num_classes=self.num_classes)  # 正确传递参数

class get_loss(nn.Module):
    def __init__(self, num_classes, alpha=0.7, mat_diff_loss_scale=0.001):  # 添加正则化系数参数
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.mat_diff_loss_scale = mat_diff_loss_scale  # 初始化权重系数
        self.seg_loss = nn.NLLLoss()
        self.norm_loss = nn.MSELoss()
        self.class_weights = torch.ones(num_classes)
        self.class_weights[0] = 0.2

    def forward(self, preds, target, trans_feat):
        seg_pred = preds  # 直接接收分割预测结果
        seg_gt = target
        
        # 分割损失计算
        seg_loss = F.nll_loss(
            seg_pred, seg_gt,
            weight=self.class_weights.to(seg_pred.device)
        )
        
        # 特征矩阵正则化
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        
        return seg_loss + mat_diff_loss * self.mat_diff_loss_scale

# 新增特征变换正则化函数
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss
