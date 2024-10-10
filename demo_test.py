import sys
import os
import os.path as osp
import argparse
import time
import numpy as np
import cv2
import colorsys
import json
import random
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt


sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import process_bbox, generate_patch_image, get_bbox
from utils.transforms import pixel2cam, cam2pixel, transform_joint_to_other_db
from utils.vis import vis_mesh, save_obj, render_mesh, vis_coco_skeleton
sys.path.insert(0, cfg.smpl_path)
from utils.smpl import SMPL



def add_pelvis(joint_coord, joints_name):
    lhip_idx = joints_name.index('L_Hip')
    rhip_idx = joints_name.index('R_Hip')
    pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
    pelvis[2] = joint_coord[lhip_idx, 2] * joint_coord[rhip_idx, 2]  # confidence for openpose
    pelvis = pelvis.reshape(1, 3)

    joint_coord = np.concatenate((joint_coord, pelvis))

    return joint_coord

def add_neck(joint_coord, joints_name):
    lshoulder_idx = joints_name.index('L_Shoulder')
    rshoulder_idx = joints_name.index('R_Shoulder')
    neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
    neck[2] = joint_coord[lshoulder_idx, 2] * joint_coord[rshoulder_idx, 2]
    neck = neck.reshape(1,3)

    joint_coord = np.concatenate((joint_coord, neck))

    return joint_coord

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--model_path', type=str, default='demo_checkpoint.pth.tar')
    # parser.add_argument('--img_idx', type=str, default='101570')
    parser.add_argument('--img_idx', type=str, default='100005')

    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids, is_test=True)
cfg.render = True
cudnn.benchmark = True

# SMPL joint set
joint_num = 30  # original: 24. manually add nose, L/R eye, L/R ear, head top，手动添加了鼻子、左右眼、左右耳和头顶
joints_name = (
'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe',
'Neck', 'L_Thorax', 'R_Thorax',
'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye',
'R_Eye', 'L_Ear', 'R_Ear', 'Head_top')
#flip_pairs 是左右对称的关节对，用于数据增强
flip_pairs = (
(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
#skeleton 是关节连接关系，用于绘制骨架
skeleton = (
(0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19),
(19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15), (24, 25), (24, 26),
(25, 27), (26, 28), (24, 29))

# SMPl mesh
vertex_num = 6890
smpl = SMPL()
face = smpl.face

# other joint set，定义了 COCO 数据集的关节名称和骨架连接关系
coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
#显示关节连接关系，用于绘制骨架
coco_skeleton = (
(1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6),
(11, 17), (12,17), (17,18))

#定义了用于可视化的关节名称和骨架连接关系
vis_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Thorax', 'Pelvis')
vis_skeleton = ((0, 1), (0, 2), (2, 4), (1, 3), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 17), (6, 17), (11, 18), (12, 18), (17, 18), (17, 0), (6, 8), (8, 10),)

# snapshot load
model_path = args.model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path 
print('Load checkpoint from {}'.format(model_path))
model = get_model(vertex_num, joint_num, 'test')

loadModel_time = time.time() 
#将模型封装到 DataParallel 中，以便在多 GPU 上进行并行计算
model = DataParallel(model).cuda()
#加载模型检查点 ckpt。将检查点中的网络参数加载到模型中，strict=False 表示允许部分参数不匹配。最后，将模型设置为评估模式 eval()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
#图像转换器 transform)，将图像转换为张量
transform = transforms.ToTensor()
pose2d_result_path = './input/2d_pose_result.json'
with open(pose2d_result_path) as f:
    pose2d_result = json.load(f)

img_dir = './input/images'
# img_dir = './input'
#生成每个图像的完整路径 img_path = ./input/images/xxx.jpg
for img_name in sorted(pose2d_result.keys()):
    img_path = osp.join(img_dir, img_name)
# for img_name in sorted(os.listdir(img_dir)):
#     img_path = osp.join(img_dir, img_name)

    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Failed to load image: {img_path}")
        continue  # 跳过该图像并处理下一个

    input = original_img.copy()
    input2 = original_img.copy()
    original_img_height, original_img_width = original_img.shape[:2]
    #获取当前图像对应的 COCO 关节列表
    coco_joint_list = pose2d_result[img_name]

    if args.img_idx not in img_name:
        continue

    drawn_joints = []
    c = coco_joint_list
    # manually assign the order of output meshes
    # coco_joint_list = [c[2], c[0], c[1], c[4], c[3]]

    for idx in range(len(coco_joint_list)):
        """ 2D pose input setting & hard-coding for filtering """
        pose_thr = 0.1
        #将当前关节点列表转换为 NumPy 数组，并只保留前三列（x 坐标、y 坐标和置信度）存进 coco_joint_img
        coco_joint_img = np.asarray(coco_joint_list[idx])[:, :3]
        coco_joint_img = add_pelvis(coco_joint_img, coco_joints_name)
        coco_joint_img = add_neck(coco_joint_img, coco_joints_name)
        coco_joint_valid = (coco_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)

        # filter inaccurate inputs
        det_score = sum(coco_joint_img[:, 2])
        if det_score < 1.0:
            continue
        if len(coco_joint_img[:, 2:].nonzero()[0]) < 1:
            continue
        # filter the same targets
        tmp_joint_img = coco_joint_img.copy()
        continue_check = False
        for ddx in range(len(drawn_joints)):
            drawn_joint_img = drawn_joints[ddx]
            drawn_joint_val = (drawn_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)
            diff = np.abs(tmp_joint_img[:, :2] - drawn_joint_img[:, :2]) * coco_joint_valid * drawn_joint_val
            diff = diff[diff != 0]
            if diff.size == 0:
                continue_check = True
            elif diff.mean() < 20:
                continue_check = True 
        if continue_check:
            continue
        drawn_joints.append(tmp_joint_img)

        """ Prepare model input """
        # prepare bbox
        bbox = get_bbox(coco_joint_img, coco_joint_valid[:, 0]) # xmin, ymin, width, height
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        if bbox is None:
            continue
        img, img2bb_trans, bb2img_trans = generate_patch_image(input2[:,:,::-1], bbox, 1.0, 0.0, False, cfg.input_img_shape)
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]

        coco_joint_img_xy1 = np.concatenate((coco_joint_img[:, :2], np.ones_like(coco_joint_img[:, :1])), 1)
        coco_joint_img[:, :2] = np.dot(img2bb_trans, coco_joint_img_xy1.transpose(1, 0)).transpose(1, 0)
        coco_joint_img[:, 0] = coco_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        coco_joint_img[:, 1] = coco_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

        coco_joint_img = transform_joint_to_other_db(coco_joint_img, coco_joints_name, joints_name)
        coco_joint_valid = transform_joint_to_other_db(coco_joint_valid, coco_joints_name, joints_name)
        coco_joint_valid[coco_joint_img[:, 2] <= pose_thr] = 0

        # check truncation
        coco_joint_trunc = coco_joint_valid * ((coco_joint_img[:, 0] >= 0) * (coco_joint_img[:, 0] < cfg.output_hm_shape[2]) * (coco_joint_img[:, 1] >= 0) * (coco_joint_img[:, 1] < cfg.output_hm_shape[1])).reshape(
            -1, 1).astype(np.float32)
        coco_joint_img, coco_joint_trunc, bbox = torch.from_numpy(coco_joint_img).cuda()[None, :, :], torch.from_numpy(coco_joint_trunc).cuda()[None, :, :], torch.from_numpy(bbox).cuda()[None, :]

        """ Model forward """
        inputs = {'img': img, 'joints': coco_joint_img, 'joints_mask': coco_joint_trunc}
        targets = {}
        meta_info = {'bbox': bbox}

        image_count = 1
        model_time = time.time()
        #调用模型进行前向推理，输出3D mesh的预测结果 (mesh_cam_render)，包括每个顶点的3D坐标
        with torch.no_grad():
            image_time = time.time()    
            out = model(inputs, targets, meta_info, 'test')
            print(f'{image_count} Image time: {time.time() - image_time} Sec')
            image_count += 1
        print(f'Model time: {time.time() - model_time} Sec')

        # draw output mesh
        mesh_cam_render = out['mesh_cam_render'][0].cpu().numpy()

        # for coord in mesh_cam_render:
        #     print(f'3D Coordinate: X={coord[0]}, Y={coord[1]}, Z={coord[2]}')
        # 将3D坐标保存为json
        output_dir = 'output'
        output_data = {"3D_coordinates": mesh_cam_render.tolist()}
        with open(f'{output_dir}/3D_coordinates_{img_name}.json', 'w') as f:
            json.dump(output_data, f)

print(f'Last model time: {time.time() - loadModel_time}')
