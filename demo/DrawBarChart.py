# import matplotlib.pyplot as plt
# import numpy as np

# categories = ['3DMutil-PoseNet', 'MuPoTS', 'CMU-Panoptic', '3DPW']
# bbox_loU = [38, 3.8, 2, 3.3]
# CrowdIndex = [49, 13, 11.3, 4.3]

# # 设置柱状图的宽度
# bar_width = 0.35

# # 设置每个柱状图的位置
# index = np.arange(len(categories))

# # 创建柱状图
# plt.figure(figsize=(10, 6))
# bar1 = plt.bar(index, bbox_loU, bar_width, color='skyblue', label='bbox_loU')
# bar2 = plt.bar(index + bar_width, CrowdIndex, bar_width, color='pink', label='CrowdIndex')

# plt.xticks(index + bar_width / 2, categories)

# # 添加图例
# plt.legend(loc='upper right')

# # 显示图表
# plt.show()

import os.path as osp
import sys

# 插入路径
sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))

# 打印当前 sys.path
print("sys.path:", sys.path)

from config import cfg
from model import get_model
from utils.preprocessing import process_bbox, generate_patch_image, get_bbox
from utils.transforms import pixel2cam, cam2pixel, transform_joint_to_other_db
from utils.vis import vis_mesh, save_obj, render_mesh, vis_coco_skeleton

# 插入 cfg.smpl_path 路径
sys.path.insert(0, cfg.smpl_path)

# 打印当前 sys.path
print("sys.path after cfg.smpl_path:", sys.path)

from utils.smpl import SMPL