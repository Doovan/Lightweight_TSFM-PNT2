2025-2-25 \
非常感谢基线源码贡献作者 我们在此基础上进行了改进  \
https://github.com/yanx27/Pointnet_Pointnet2_pytorch    \
\
\
Pointnet2_Transformer_pytorch 为修正迭代的模型版本
Pointnet2_base 为基线版本
\
\
\
训练命令（部件分割）：\
python train_partseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_part_seg_msg \
\
\
验证命令（部件分割）：\
python test_partseg.py --log_dir pointnet2_part_seg_msg
