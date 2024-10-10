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
from ultralytics import YOLO


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
    parser.add_argument('--img_idx', type=str, default='101570')

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
joint_num = 30  # original: 24. manually add nose, L/R eye, L/R ear, head top
joints_name = (
'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe',
'Neck', 'L_Thorax', 'R_Thorax',
'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye',
'R_Eye', 'L_Ear', 'R_Ear', 'Head_top')
flip_pairs = (
(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
skeleton = (
(0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19),
(19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15), (24, 25), (24, 26),
(25, 27), (26, 28), (24, 29))

# SMPl mesh
vertex_num = 6890
smpl = SMPL()
face = smpl.face

# other joint set
coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'Neck')
coco_skeleton = (
(1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6),
(11, 17), (12,17), (17,18))

vis_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Thorax', 'Pelvis')
vis_skeleton = ((0, 1), (0, 2), (2, 4), (1, 3), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 17), (6, 17), (11, 18), (12, 18), (17, 18), (17, 0), (6, 8), (8, 10),)

# snapshot load
model_path = args.model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path 
print('Load checkpoint from {}'.format(model_path))
model = get_model(vertex_num, joint_num, 'test')

loadModel_time = time.time() 
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()
pose2d_result_path = './input/2d_pose_result.json'
with open(pose2d_result_path) as f:
    pose2d_result = json.load(f)

# 导入YOLO模型
model_yolo = YOLO("./best.pt")

# cap = cv2.VideoCapture("./input/12.mp4")
cap = cv2.VideoCapture("./input/1.jpg")
time_count = 0

img_dir = './input/images'
while cap.isOpened():
    # 读取摄像头图像
    success, image = cap.read()
    if not success:
        print("忽略空视频帧")
        time_count += 1
        if time_count > 20:
            break
        continue

    # 使用YOLOv8检测人物, 提高YOLOv8的检测置信度
    # results_tr = model.track(source=image, conf=0.5, persist=True, show=True)
    results_tr = model.track(source=image, conf=0.3, persist=True, show=False)

    # 存储每个人物的初始边界框大小和最大边界框
    initial_bbox_sizes = {}
    max_bbox_sizes = {}

    # 处理跟踪结果
    tracked_objects = results_tr[0].boxes
    for obj in tracked_objects:
        if int(obj.cls[0]) != 0:  # 类别0表示“person”
            continue

        # 获取边界框坐标
        x1, y1, x2, y2 = map(int, obj.xyxy[0])  
        width = x2 - x1
        height = y2 - y1

        # 获取跟踪ID
        track_id = int(obj.id[0]) if obj.id is not None else 'N/A'  

        # # 裁剪出人物区域
        person_roi = image[y1:y2, x1:x2]
        original_img = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

        img_name = str(time.time())
        img_path = osp.join(img_dir, img_name)

        input = original_img.copy()
        input2 = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        coco_joint_list = pose2d_result[img_name]

        if args.img_idx not in img_name:
            print(f'Ignore {img_name}')
            continue

        drawn_joints = []
        c = coco_joint_list
        # manually assign the order of output meshes
        # coco_joint_list = [c[2], c[0], c[1], c[4], c[3]]

        
        for idx in range(len(coco_joint_list)):
            """ 2D pose input setting & hard-coding for filtering """
            pose_thr = 0.1
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
            with torch.no_grad():
                image_time = time.time()    
                out = model(inputs, targets, meta_info, 'test')
                print(f'{image_count} Image time: {time.time() - image_time} Sec')
                image_count += 1
            print(f'Model time: {time.time() - model_time} Sec')

            # draw output mesh
            mesh_cam_render = out['mesh_cam_render'][0].cpu().numpy()
            bbox = out['bbox'][0].cpu().numpy()
            princpt = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
            # original_img = vis_bbox(original_img, bbox, alpha=1)  # for debug

            # for coord in mesh_cam_render:
            #     print(f'3D Coordinate: X={coord[0]}, Y={coord[1]}, Z={coord[2]}')
            # 将3D坐标保存为json
            output_dir = 'output'
            output_data = {"3D_coordinates": mesh_cam_render.tolist()}
            with open(f'{output_dir}/3D_coordinates_{img_name}.json', 'w') as f:
                json.dump(output_data, f)

            # generate random color
            color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
            original_img = render_mesh(original_img, mesh_cam_render, face, {'focal': cfg.focal, 'princpt': princpt}, color=color)

            # Save output mesh
            output_dir = 'output'
            file_name = f'{output_dir}/{img_path.split("/")[-1][:-4]}_{idx}.jpg'
            print("file name: ", file_name)
            save_obj(mesh_cam_render, face, file_name=f'{output_dir}/{img_path.split("/")[-1][:-4]}_{idx}.obj')
            cv2.imwrite(file_name, original_img)

            # Draw input 2d pose
            tmp_joint_img[-1], tmp_joint_img[-2] = tmp_joint_img[-2].copy(), tmp_joint_img[-1].copy()
            input = vis_coco_skeleton(input, tmp_joint_img.T, vis_skeleton)
            cv2.imwrite(file_name[:-4] + '_2dpose.jpg', input)

            # 按下ESC键退出
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 释放视频资源
cap.release()
# out.release()
cv2.destroyAllWindows()