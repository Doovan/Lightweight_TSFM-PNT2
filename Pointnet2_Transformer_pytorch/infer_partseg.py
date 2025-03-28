"""
点云部件分割推理脚本（完整增强版）
"""
import argparse
import os
import sys
import logging
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from models.pointnet2_part_seg_msg import get_model  # 根据实际路径修改

# -------------------- 参数解析 --------------------
def parse_args():
    parser = argparse.ArgumentParser('点云部件分割推理')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='训练好的模型路径')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='ASC文件输入目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='PLY结果输出目录')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='推理批大小')
    parser.add_argument('--gpu', type=str, default='0',
                       help='使用的GPU编号')
    parser.add_argument('--num_point', type=int, default=2048,
                       help='每个点云采样点数')
    parser.add_argument('--num_classes', type=int, default=3,
                       help='分类类别数')
    return parser.parse_args()

# -------------------- 全局常量 --------------------
LABEL_COLORS = np.array([
    [255, 0, 0],    # 类别0 - 红色
    [0, 255, 0],    # 类别1 - 绿色
    [0, 0, 255],    # 类别2 - 蓝色
    [255, 255, 0],  # 其他类别颜色
    [255, 0, 255],
    [0, 255, 255]
])

# -------------------- 预处理模块 --------------------
class ASCPreprocessor:
    def __init__(self, num_points=2048):
        self.num_points = num_points
        
    def farthest_point_sample(self, points):
        """ 改进版最远点采样 """
        points = np.array(points)
        n_points = points.shape[0]
        
        if n_points <= self.num_points:
            return points
        
        start_idx = np.random.randint(n_points)
        sampled = np.zeros((self.num_points, 3), dtype=np.float32)
        sampled[0] = points[start_idx]
        distance = np.full(n_points, np.inf)
        
        for i in range(1, self.num_points):
            dist = np.sum((points - sampled[i-1])**2, axis=1)
            np.minimum(distance, dist, out=distance)
            farthest = np.argmax(distance)
            sampled[i] = points[farthest]
        return sampled

    def normalize(self, pc):
        """ 与训练一致的归一化 """
        centroid = np.mean(pc, axis=0)
        pc -= centroid
        max_dist = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc /= max_dist + 1e-8  # 防止零除
        return pc

    def process(self, file_path):
        """ ASC文件处理流水线 """
        try:
            # 读取数据（只取前3列）
            points = np.loadtxt(file_path, usecols=(0,1,2)).astype(np.float32)
            
            # 采样处理
            if len(points) > self.num_points:
                points = self.farthest_point_sample(points)
            else:
                # 重复填充不足点数
                choice = np.random.choice(len(points), self.num_points, replace=True)
                points = points[choice]
            
            # 归一化处理
            return self.normalize(points)
        except Exception as e:
            logging.error(f"处理 {file_path} 失败: {str(e)}")
            raise

# -------------------- 数据集类 --------------------
class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, preprocessor):
        self.input_files = list(Path(input_dir).glob("*.asc"))
        if not self.input_files:
            raise FileNotFoundError(f"目录中未找到ASC文件: {input_dir}")
        self.preprocessor = preprocessor
        logging.info(f"加载 {len(self.input_files)} 个点云文件")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        file_path = self.input_files[idx]
        try:
            points = self.preprocessor.process(file_path)
            return torch.from_numpy(points), file_path.name
        except Exception as e:
            logging.error(f"读取 {file_path} 失败: {str(e)}")
            raise

# -------------------- 模型加载 --------------------
def load_model(model_path, num_classes, device):
    """ 增强模型加载 """
    from models.pointnet2_part_seg_msg import get_model  # 动态导入防止路径问题
    
    model = get_model(num_classes=num_classes).to(device)
    
    try:
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        
        # 处理多GPU训练保存的模型
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        logging.error(f"模型加载失败: {str(e)}")
        raise

# -------------------- 结果保存 --------------------
def save_ply(points, labels, path):
    """ 带颜色编码的PLY保存 """
    colors = LABEL_COLORS[labels % len(LABEL_COLORS)]
    vertex = np.array(
        [(x, y, z, r, g, b) for (x, y, z), (r, g, b) in zip(points, colors)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    )
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(str(path))

# -------------------- 主函数 --------------------
def main(args):
    # 初始化设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 验证输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化预处理
    preprocessor = ASCPreprocessor(args.num_point)
    
    # 加载模型
    model = load_model(args.model_path, args.num_classes, device)
    
    # 数据加载
    dataset = InferenceDataset(args.input_dir, preprocessor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False
    )
    
    # 推理循环
    with torch.no_grad():
        for batch_idx, (points, filenames) in enumerate(loader):
            points = points.transpose(2, 1).to(device)
            cls_label = torch.zeros(points.size(0), dtype=torch.long).to(device)
            
            # 模型推理
            seg_pred, _ = model(points, cls_label)
            labels = seg_pred.argmax(-1).cpu().numpy()
            
            # 保存结果
            for b in range(points.size(0)):
                pts = points[b].transpose(0, 1).cpu().numpy()
                save_path = output_dir / f"{Path(filenames[b]).stem}_pred.ply"
                save_ply(pts, labels[b], save_path)
                logging.info(f"已保存: {save_path}")

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        logging.exception("推理过程发生致命错误:")
        sys.exit(1)