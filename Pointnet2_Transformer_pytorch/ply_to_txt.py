"""
PLY点云转TXT格式工具
功能特性: 
- 批量转换目录下所有.ply文件
- 保留坐标(x,y,z)和颜色(r,g,b)信息
- 自动处理不同格式的PLY文件
- 进度条显示转换进度
"""

import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    from plyfile import PlyData
except ImportError:
    print("错误: 需要安装plyfile库, 请执行: pip install plyfile")
    sys.exit(1)

def ply_to_txt(input_path, output_path, delimiter=' ', with_color=True):
    """
    转换单个PLY文件到TXT
    参数: 
    input_path: 输入PLY文件路径
    output_path: 输出TXT文件路径
    delimiter: 分隔符, 默认为空格
    with_color: 是否包含颜色信息, 默认为True
    """
    try:
        # 读取PLY文件
        ply_data = PlyData.read(input_path)
        vertices = ply_data['vertex'].data
        
        # 提取数据
        x = vertices['x'].astype(float)
        y = vertices['y'].astype(float)
        z = vertices['z'].astype(float)
        
        # 初始化数据矩阵
        data = np.column_stack((x, y, z))
        
        # 添加颜色信息( 如果存在且需要)
        if with_color:
            color_columns = []
            for color in ['red', 'green', 'blue']:
                if color in vertices.dtype.names:
                    color_columns.append(vertices[color].astype(float)/255.0)  # 归一化到[0,1]
                else:
                    print(f"警告: {input_path} 缺少颜色通道 {color}, 已跳过颜色转换")
                    color_columns = []
                    break
            
            if color_columns:
                data = np.column_stack((data, *color_columns))

        # 保存为TXT
        np.savetxt(output_path, data, 
                 delimiter=delimiter,
                 fmt='%.6f' if with_color else '%.3f')  # 颜色信息使用更高精度
                 
    except Exception as e:
        print(f"转换失败: {input_path} - {str(e)}")

def batch_convert(input_dir, output_dir, delimiter=' ', with_color=True):
    """
    批量转换函数
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有PLY文件
    ply_files = list(input_dir.glob("*.ply"))
    if not ply_files:
        print(f"错误: 输入目录中没有找到PLY文件 - {input_dir}")
        return
    
    print(f"找到 {len(ply_files)} 个PLY文件, 开始转换...")
    
    # 使用进度条
    with tqdm(total=len(ply_files), unit='file') as pbar:
        for ply_file in ply_files:
            output_file = output_dir / f"{ply_file.stem}.txt"
            ply_to_txt(ply_file, output_file, delimiter, with_color)
            pbar.update(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PLY点云转TXT工具')
    parser.add_argument('-i', '--input', required=True, help='输入目录路径( 包含PLY文件)')
    parser.add_argument('-o', '--output', required=True, help='输出目录路径')
    parser.add_argument('-d', '--delimiter', default=' ', choices=[' ', ',', ';', '\t'], 
                       help='分隔符( 默认: 空格)')
    parser.add_argument('-nc', '--no-color', action='store_true',
                       help='不包含颜色信息')
    args = parser.parse_args()
    
    batch_convert(
        input_dir=args.input,
        output_dir=args.output,
        delimiter=args.delimiter,
        with_color=not args.no_color
    )
    
    print(f"\n转换完成! 输出目录: {args.output}")