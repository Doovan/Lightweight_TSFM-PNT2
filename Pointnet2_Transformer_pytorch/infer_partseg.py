"""显存优化版推理代码"""
import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement

class InferenceConfig:
    CHUNK_SIZE = 2048    # 减小分块尺寸
    OVERLAP_RATIO = 0.2  # 增大重叠比例保证精度
    BATCH_SIZE = 1       # 单批次处理
    LABEL_COLORS = np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255]
    ], dtype=np.uint8)  # 根据实际类别数调整为3类

class MemoryOptimizedProcessor:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def process_file(self, input_path, output_path):
        # 读取数据
        points = np.loadtxt(input_path, dtype=np.float32)[:, :3]
        
        # 分块处理
        chunks = self._chunk_points(points)
        preds = self._safe_inference(chunks)
        
        # 合并结果
        full_pred = self._merge_predictions(points, preds)
        self._save_ply(points, full_pred, output_path)
    
    def _chunk_points(self, points):
        """安全分块策略"""
        chunk_size = self.config.CHUNK_SIZE
        overlap = int(chunk_size * self.config.OVERLAP_RATIO)
        stride = chunk_size - overlap
        
        chunks = []
        for i in range(0, len(points), stride):
            end = min(i + chunk_size, len(points))
            chunk = points[i:end]
            
            if len(chunk) < chunk_size:
                pad = np.zeros((chunk_size - len(chunk), 3), dtype=points.dtype)
                chunk = np.concatenate([chunk, pad])
            
            chunks.append(chunk)
        return chunks
    
    def _safe_inference(self, chunks):
        """显存安全推理"""
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for chunk in tqdm(chunks, desc="处理分块"):
                # 单块处理
                tensor = torch.from_numpy(chunk).float().cuda()
                tensor = tensor.transpose(1, 0).unsqueeze(0)  # (1, 3, N)
                
                # 使用半精度减少显存占用
                with torch.cuda.amp.autocast():
                    cls_label = torch.zeros(1, dtype=torch.long).cuda()
                    seg_logits, _ = self.model(tensor, cls_label)
                    pred = seg_logits.argmax(dim=-1).squeeze()
                
                # 立即转移数据到CPU
                all_preds.append(pred.cpu().numpy())
                del tensor, seg_logits, pred  # 主动释放显存
                torch.cuda.empty_cache()      # 清空缓存
        
        return np.concatenate(all_preds)
    
    def _merge_predictions(self, points, preds):
        """带权重合并"""
        merged = np.zeros(len(points), dtype=np.float32)
        weights = np.zeros_like(merged)
        
        chunk_size = self.config.CHUNK_SIZE
        overlap = int(chunk_size * self.config.OVERLAP_RATIO)
        stride = chunk_size - overlap
        
        for i, pred in enumerate(preds):
            start = i * stride
            end = start + chunk_size
            valid_end = min(end, len(points))
            
            # 创建高斯权重
            x = np.linspace(-1, 1, chunk_size)
            weight = np.exp(-x**2 / 0.5)
            weight = weight[:valid_end-start]
            
            merged[start:valid_end] += pred[:valid_end-start] * weight
            weights[start:valid_end] += weight
        
        return np.round(merged / weights).astype(np.int32)
    
    def _save_ply(self, points, labels, path):
        """优化存储"""
        valid_mask = (points != 0).any(axis=1)
        valid_points = points[valid_mask]
        valid_labels = labels[valid_mask]
        
        vertex_data = np.zeros(valid_points.shape[0], dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        
        colors = self.config.LABEL_COLORS[valid_labels % len(self.config.LABEL_COLORS)]
        vertex_data['x'] = valid_points[:, 0]
        vertex_data['y'] = valid_points[:, 1]
        vertex_data['z'] = valid_points[:, 2]
        vertex_data['red'] = colors[:, 0]
        vertex_data['green'] = colors[:, 1]
        vertex_data['blue'] = colors[:, 2]
        
        PlyData([PlyElement.describe(vertex_data, 'vertex')], 
               text=False).write(str(path))

class PointCloudSegmentor:
    def __init__(self, model):
        self.model = model
        
    def process_file(self, input_path, output_path):
        """处理单个点云文件（无分块）"""
        # 读取完整点云
        points = np.loadtxt(input_path, dtype=np.float32)[:, :3]
        
        # 转换为张量
        tensor_points = torch.from_numpy(points).float().cuda()
        tensor_points = tensor_points.transpose(1, 0).unsqueeze(0)  # (1, 3, N)
        
        # 模型推理
        with torch.no_grad():
            cls_label = torch.zeros(1, dtype=torch.long).cuda()
            seg_logits, _ = self.model(tensor_points, cls_label)
            preds = seg_logits.argmax(dim=-1).squeeze().cpu().numpy()
        
        # 保存优化后的PLY
        self._save_binary_ply(points, preds, output_path)

    def _save_binary_ply(self, points, labels, path):
        """二进制格式存储（体积更小）"""
        # 生成颜色数据
        colors = InferenceConfig.LABEL_COLORS[labels % len(InferenceConfig.LABEL_COLORS)]
        
        # 创建结构化数组
        vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        vertex_data = np.zeros(points.shape[0], dtype=vertex_dtype)
        
        vertex_data['x'] = points[:, 0]
        vertex_data['y'] = points[:, 1]
        vertex_data['z'] = points[:, 2]
        vertex_data['red'] = colors[:, 0]
        vertex_data['green'] = colors[:, 1]
        vertex_data['blue'] = colors[:, 2]
        
        # 保存为二进制PLY
        PlyData([PlyElement.describe(vertex_data, 'vertex')], 
               text=False).write(str(path))

class InferenceEngine:
    def __init__(self, model_path, num_classes, device):
        self.device = device
        self.model = self._load_model(model_path, num_classes)
        self.processor = PointCloudSegmentor(self.model)
    
    def _load_model(self, model_path, num_classes):
        """模型加载优化"""
        from models.pointnet2_part_seg_msg import get_model
        
        model = get_model(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 兼容多卡训练参数
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(state_dict, strict=False)
        return model.to(self.device).eval()
    
    def process(self, input_dir, output_dir):
        """批量处理"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for asc_file in Path(input_dir).glob("*.asc"):
            ply_path = output_dir / f"{asc_file.stem}_seg.ply"
            self.processor.process_file(str(asc_file), str(ply_path))

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    engine = InferenceEngine(args.model_path, args.num_classes, device)
    engine.process(args.input_dir, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="模型权重路径")
    parser.add_argument("--input_dir", required=True, help="输入ASC目录")
    parser.add_argument("--output_dir", required=True, help="输出PLY目录")
    parser.add_argument("--gpu", default="0", help="GPU编号")
    parser.add_argument("--num_classes", type=int, required=True, help="类别数")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)