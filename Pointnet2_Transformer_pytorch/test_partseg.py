"""
Author: Benny
Date: Nov 2019
Modified by: [Your Name]
Date: [Current Date]
"""
import argparse
import os
from pathlib import Path
from data_utils.ShapeNetDataLoader import DealMyDataset  # 修改为你的数据集加载器
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# 修改seg_classes与训练代码一致
seg_classes = {'zhoukong': [0,1,2]}  # 你的新类别定义
seg_label_to_cat = {}  # {0:zhoukong, 1:zhoukong, 2:zhoukong}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=1, help='禁用投票或设置为1')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    '''CREATE DIRS'''
    experiment_dir = Path('log/part_seg') / args.log_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)  # 修复路径创建问题
    
    # 创建visual目录（如果需要）
    visual_dir = experiment_dir / 'visual'
    visual_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 确保日志目录存在
    log_file = experiment_dir / 'eval.txt'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(str(args))

    '''DATA LOADING'''
    TEST_DATASET = DealMyDataset(root="/home/kong-vb/data_set/test_data")
    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    log_string(f"The number of test data is: {len(TEST_DATASET)}")

    '''MODEL CONFIG'''
    num_classes = 1
    num_part = 3

    '''MODEL LOADING'''
    model_name = 'pointnet2_part_seg_msg'
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    
    checkpoint_path = experiment_dir / 'checkpoints' / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    log_string(f"Loaded best model from {checkpoint_path}")

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}

        classifier = classifier.eval()
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), 
                                                     total=len(testDataLoader),
                                                     smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points = points.float().cuda()
            points = points.transpose(2, 1)
            label = label.long().cuda()
            target = target.long().cuda()

            seg_pred, _ = classifier(points, to_categorical(label, num_classes))
            cur_pred_val = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
            target_np = target.cpu().numpy()

            correct = np.sum(cur_pred_val == target_np)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target_np == l)
                total_correct_class[l] += np.sum((cur_pred_val == l) & (target_np == l))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target_np[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = []
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):
                        iou = 1.0
                    else:
                        iou = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                    part_ious.append(iou)
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        log_string('Test Accuracy: %f' % test_metrics['accuracy'])
        log_string('Class Avg Accuracy: %f' % test_metrics['class_avg_accuracy'])
        log_string('Class Avg IoU: %f' % test_metrics['class_avg_iou'])
        log_string('Instance Avg IoU: %f' % test_metrics['inctance_avg_iou'])

if __name__ == '__main__':
    args = parse_args()
    main(args)