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
ply转txt预处理命令：\
python ply_to_txt.py -i /home/kong-vb/data_set/raw_data -o /home/kong-vb/data_set/raw_data2/ \
\
\
训练命令（部件分割）：\
python train_partseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_part_seg_msg \
\
\
验证命令（部件分割）：\
python test_partseg.py --log_dir pointnet2_part_seg_msg
\
\
推理命令（部件分割）:
python infer_partseg.py --model_path log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth --input_dir home/kong-vb/data_set/raw_data2 --output_dir /home/kong-vb/save_ply_files --batch_size 8 --gpu 0