from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from queue import Queue
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

# 定义处理任务的函数
def process_person_keypoints(model, inputs, targets, meta_info, person_id):
    # print(f"Processing person {person_id} on device: {next(model.parameters()).device}")
    with torch.no_grad():
        output = model(inputs, targets, meta_info, 'test')
    
    keypoints_3d = output['mesh_cam_render'][0].cpu().numpy()  # 这只是一个示例，具体取决于模型输出
    return {
        'person_id': person_id,
        'keypoints_3d': keypoints_3d.tolist()  # 将3D关键点转换为列表
    }

# 初始化多个模型实例，并分配到CPU或GPU
def init_models(num_models, device='cuda'):
    models = []
    model_lode_time = time.time()
    for i in range(num_models):
        model = get_model(vertex_num, joint_num, 'test')  # 从 demo_jiehe.py 获取模型
        model = model.to(device)
        model.eval()
        models.append(model)
    
    print(f'所有模型加载时间: {time.time() - model_lode_time}')
    return models


# 定义多线程并行处理关键点数据的调度器
def scheduler(models, output_file_path):
    results = []
    num_models = len(models)
        
    #图像转换器 transform)，将图像转换为张量
    transform = transforms.ToTensor()
    # pose2d_result_path = './input/2d_pose_result.json'
    pose2d_result_path = './input/output_keypoints.json'
    with open(pose2d_result_path) as f:
        pose2d_result = json.load(f)


    img_dir = './input/images'
    # img_dir = './input/'
    #生成每个图像的完整路径 img_path = ./input/images/xxx.jpg
    for img_name in sorted(pose2d_result.keys()):
        img_path = osp.join(img_dir, img_name)
    # for img_name in sorted(os.listdir(img_dir)):
    #     img_path = osp.join(img_dir, img_name)

        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Failed to load image: {img_path}")
            continue  # 跳过该图像并处理下一个

        # print(f'获取到图片: {img_path}')
        input = original_img.copy()
        input2 = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        #获取当前图像对应的 COCO 关节列表
        coco_joint_list = pose2d_result[img_name]
        # print(f'获取到 COCO 关节列表: {coco_joint_list}')

        # if args.img_idx not in img_name:
        #     print(f'跳过图片: {img_name}')
        #     continue
        

        drawn_joints = []
        c = coco_joint_list

    task_times = []
    start_time = time.time()

    # 创建线程池
    with ThreadPoolExecutor(max_workers=num_models) as executor:
        futures = []
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

            model = models[i % num_models]  # 轮流使用每个模型
            task_start_time = time.time()
            # 提交任务到线程池
            futures.append(executor.submit(process_person_keypoints, model, inputs, targets, meta_info, i))
            # print(f'提交任务到线程池: {i}, 用时: {time.time() - task_start_time}')

        # 获取所有线程的执行结果
        for future in as_completed(futures):
            task_end_time = time.time()
            task_times.append(task_end_time - task_start_time)
            results.append(future.result())


    #将task_times的结果从后面的时间减去前面的时间，得到每个任务的用时
    task_times = [task_times[i] - task_times[i-1] for i in range(1, len(task_times))]

    # 保存所有结果到JSON文件
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)  # 保存为JSON文件

    all_model_time = time.time()

    print(f'每个任务用时: {task_times}')
    print(f'所有任务用时: {sum(task_times)}')
    print(f'所有任务已完成，总用时: {all_model_time - start_time}')
    print(f'从加载模型到处理所有任务的总时间: {all_model_time - init_time}')

    return results


# def pose_model_run():
# 导入YOLO模型
model_yolo = YOLO("./yolov8m-pose.pt")
# image_path = "C:/Users/Liu/Desktop/100707.jpg"
image_path = "./input/100083.jpg"
cap = cv2.VideoCapture(image_path)
# image = cv2.imread(image_path)
time_count = 0

# 获取文件名
file_name = os.path.basename(image_path)

# 存储字典形式的关键点数据
keypoints_dict = {}

while cap.isOpened():
    # 读取摄像头图像
    success, image = cap.read()
    if not success:
        print("忽略空视频帧")
        time_count += 1
        if time_count > 20:
            break
        continue

    img_height, img_width = image.shape[:2]

    # 存储每个人的关键点数据
    person_keypoints_list = []    

    # Run inference
    results = model_yolo(image, conf=0.6)

        # 对每一帧结果进行处理
    for result in results:
        for person in result.keypoints:
            frame_keypoints = []
            # 提取关键点的x, y坐标和置信度
            keypoints_xy = person.xy  # 提取关键点坐标
            keypoints_conf = person.conf  # 提取关键点置信度

            # 遍历每个关键点，组合 [x, y, confidence]
            for kp, conf in zip(keypoints_xy[0], keypoints_conf[0]):
                x, y = kp.tolist()  # 将tensor转化为list
                x_normalized = (x / img_width) * 2 - 1  # 将x归一化为[-1, 1]
                y_normalized = (y / img_height) * 2 - 1  # 将y归一化为[-1, 1]
                frame_keypoints.append([x, y, conf.item(), x_normalized, y_normalized])
                # frame_keypoints.append([x, y, conf.item()])  # 组合 [x, y, confidence]

            # 将这一帧的关键点存储到结果列表中
            person_keypoints_list.append(frame_keypoints)

        # 将此帧的所有人物关键点加入字典，使用文件名作为键

    if person_keypoints_list:
        keypoints_dict[file_name] = person_keypoints_list

    # 由于只是处理图片，可以在这里直接退出循环
    break

output_json_path = './input/output_keypoints.json'
with open(output_json_path, 'w') as f:
    f.write('{\n')
    key_count = len(keypoints_dict)
    for i, (key, value) in enumerate(keypoints_dict.items()):
        f.write(f'  "{key}": [\n')
        person_count = len(value)
        for j, person in enumerate(value):
            f.write('    [\n')
            keypoint_count = len(person)
            for k, keypoint in enumerate(person):
                if k < keypoint_count - 1:
                    f.write(f'      {keypoint},\n')  # 每个关键点占一行
                else:
                    f.write(f'      {keypoint}\n')  # 最后一个关键点不加逗号
            if j < person_count - 1:
                f.write('    ],\n')
            else:
                f.write('    ]\n')  # 最后一个人不加逗号
        if i < key_count - 1:
            f.write('  ],\n')
        else:
            f.write('  ]\n')  # 最后一个键值对不加逗号
    f.write('}\n')
# print(keypoints_dict)

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

# snapshot load
model_path = args.model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path 
# print('Load checkpoint from {}'.format(model_path))
# model = get_model(vertex_num, joint_num, 'test')


# 假设有4个GPU，初始化4个模型
num_models = 4
device = 'cuda'  # 可以替换为'cpu'或者具体的GPU，比如'cuda:0'

init_time = time.time()
models = init_models(num_models, device)

# 获取YOLO识别到的人物关键点数据
# person_keypoints_data = yolo_detected_data()

output_file_path = 'output_results1.json'  # 指定输出文件路径

# 调度并行处理关键点数据
results = scheduler(models, output_file_path)
print(f'结果已保存到 {output_file_path}')
