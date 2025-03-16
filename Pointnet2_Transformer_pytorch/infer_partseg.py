""" 修复版推理脚本（含完整验证）"""
import argparse
import os
import sys
import logging
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData, PlyElement

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
from models.pointnet2_part_seg_msg import get_model  # 根据实际路径修改

LABEL_COLORS = np.array([[255,0,0], [0,255,0], [0,0,255]])

def parse_args():
    parser = argparse.ArgumentParser('推理脚本')
    parser.add_argument('--model_path', required=True, help='模型路径')
    parser.add_argument('--input_dir', required=True, help='输入目录')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_point', type=int, default=2048)
    parser.add_argument('--num_classes', type=int, default=3)
    return parser.parse_args()

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, num_points=2048):
        self.input_files = list(Path(input_dir).glob('*.ply'))
        if not self.input_files:
            raise FileNotFoundError(f"无输入文件: {input_dir}")
        print(f"找到 {len(self.input_files)} 个输入文件")
        self.num_points = num_points

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        file_path = self.input_files[idx]
        points = np.loadtxt(file_path).astype(np.float32)
        
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"文件 {file_path} 格式错误，形状: {points.shape}")
            
        choice = np.random.choice(
            points.shape[0], self.num_points, 
            replace=(points.shape[0] < self.num_points)
        )
        return torch.from_numpy(points[choice, :3]), file_path.name

def save_ply(points, labels, path):
    colors = LABEL_COLORS[labels]
    vertex = np.array([(x,y,z,*rgb) for (x,y,z), rgb in zip(points, colors)],
                     dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    PlyData([PlyElement.describe(vertex,'vertex')]).write(str(path))

def main(args):
    # 设备设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # 输出目录验证
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    test_file = output_dir / "test_write.tmp"
    try:
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise PermissionError(f"无法写入 {output_dir}: {e}")

    # 模型加载
    model = get_model(num_classes=args.num_classes).to(device).eval()
    ckpt = torch.load(args.model_path, map_location=device)
    
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
    else:
        model = ckpt.to(device)
    
    # 数据加载
    dataset = InferenceDataset(args.input_dir, args.num_point)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 推理
    with torch.no_grad():
        for i, (points, names) in enumerate(loader):
            print(f"处理批次 {i+1}/{len(loader)}")
            points = points.transpose(2, 1).to(device)
            cls_label = torch.zeros(points.size(0), dtype=torch.long).to(device)
            
            seg_pred, _ = model(points, cls_label)
            labels = seg_pred.argmax(-1).cpu().numpy()
            
            for b in range(points.size(0)):
                pts = points[b].transpose(0,1).cpu().numpy()[:,:3]
                save_path = output_dir / f"{Path(names[b]).stem}.ply"
                save_ply(pts, labels[b], save_path)
                print(f"已保存: {save_path}")

if __name__ == '__main__':
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"错误发生: {str(e)}")
        sys.exit(1)