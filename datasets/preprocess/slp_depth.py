#!/usr/bin/python
import os
from os.path import join
import sys
sys.path.append('./')
import argparse
import numpy as np
import cv2
import scipy.io as sio
from read_openpose import read_openpose
import constants
from utils.imutils import transform
from utils.visualize import Debugger

h36m_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], 
              [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
              [8, 12], [9, 12],
              [2, 8], [3, 9]]

def slp_single_mod(dataset_path, out_path, out_name, img_types, sub_list):

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_, openposes_  = [], [], [], [], [], []

    global_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14]
    global_idx_17 =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]

    # go over all the images
    for sub_ind in sub_list:
        sub_folder = "%05d" % (sub_ind)
        for img_type in img_types:
            openpose_folder = join(dataset_path, sub_folder, 'openpose')

            # 2D annotation files
            annot_file = os.path.join(dataset_path, sub_folder, 'joints_gt_RGB.mat')
            joints = sio.loadmat(annot_file)['joints_gt']

            imgs = range(45)
            for img_i in imgs:
                # image name
                imgname = 'image_%06d.png' % (img_i+1)
                imgname_full = join(sub_folder, img_type, imgname)
                # read 2d keypoints
                part14 = joints[:2,:,img_i].T
                # scale and center
                bbox = [min(part14[:,0]), min(part14[:,1]),
                        max(part14[:,0]), max(part14[:,1])]
                center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
                # update keypoints
                part = np.zeros([24,3])
                part[:14] = np.hstack([part14, np.ones([14,1])])
                # read openpose detections
                json_file = os.path.join(openpose_folder, imgname.replace('.png', '_keypoints.json'))
                openpose = read_openpose(json_file, part, 'lsp')

                # 3D annotation files
                annot_file_3d = os.path.join(dataset_path, sub_folder, ("gt_3d/"+imgname[:-4]+'.mat'))
                poses_3d = sio.loadmat(annot_file_3d)['joint_gt_3d']

                # change 2d pose
                # print(part)
                part[2:4, :2] = poses_3d[2:4, :2] # hip
                part[8:10, :2] = poses_3d[8:10, :2] # shoulder
                part[1, :2] = poses_3d[1, :2] # knee
                part[4, :2] = poses_3d[4, :2] # knee
                # print(part)

                # print("a",poses_3d)
                ## center norm
                c = np.array([1024 / 2., 1024 / 2.], dtype=np.float32)
                poses_3d[:,:2] = poses_3d[:,:2] / c - 1.0

                # read GT 3D pose (17 points)
                S15 = np.reshape(poses_3d, [-1,3])
                S15[14,:] = np.mean((S15[2,:], S15[3,:]), 0)
                S17 = np.zeros([17,3])
                S17[:15,:] = S15
                
                S17[15,:] = np.mean((S15[2,:], S15[3,:], S15[8,:], S15[9,:]), 0)
                S17[16,:] = np.mean((S15[12,:], S15[13,:]), 0)

                S17 -= S17[14] # root-centered
                S24 = np.zeros([24,4])
                S24[global_idx_17, :3] = S17
                S24[global_idx_17, 3] = 1

                # store data
                imgnames_.append(imgname_full)
                centers_.append(center)
                scales_.append(scale)
                parts_.append(part)
                Ss_.append(S24)
                openposes_.append(openpose)

            # print(imgname_full)
            # img_ori = cv2.imread(os.path.join(SLP_ROOT,imgname_full)).copy()
            # debugger = Debugger(edges=h36m_edges)
            # debugger.add_img(img_ori)
            # debugger.add_point_2d(part, (255, 0, 0))
            # debugger.add_point_3d(S24[:,:3], 'b')
            # debugger.show_all_imgs(pause=True)
            # debugger.show_3d()

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, out_name)
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       S=Ss_,
                       openpose=openposes_
                       )


def slp_multi_mod(dataset_path, out_path, out_name, cover_types, sub_list):

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we use
    imgnames_,  scales_, centers_, parts_, Ss_, openposes_, gender_  = [], [], [], [], [], [], []
    irimgnames_, depthnames_, pmnames_ = [], [], []

    global_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14]
    global_idx_17 =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]

    # load gender
    gender_all = np.loadtxt(os.path.join(os.path.dirname(dataset_path), "danaLab_data_gender.csv"))

    # go over all the images
    for sub_ind in sub_list:#range(102):
        sub_folder = "%05d" % (sub_ind)
        for cover_type in cover_types:
            openpose_folder = join(dataset_path, sub_folder, 'openpose')

            # 2D annotation files
            annot_file = os.path.join(dataset_path, sub_folder, 'joints_gt_RGB.mat')
            joints = sio.loadmat(annot_file)['joints_gt']

            imgs = range(45)
            for img_i in imgs:
                # image name
                imgname = '%06d.png' % (img_i+1)
                imgname_full = join(sub_folder, "RGB/" + cover_type, 'image_' + imgname)
                ir_imgname_full = join(sub_folder, "IR_aligned/" + cover_type, imgname)
                depth_name_full = join(sub_folder, "depth_aligned/" + cover_type, imgname)
                pm_name_full = join(sub_folder, "PM_aligned/" + cover_type, imgname)
                # read 2d keypoints
                part14 = joints[:2,:,img_i].T
                # scale and center
                bbox = [min(part14[:,0]), min(part14[:,1]),
                        max(part14[:,0]), max(part14[:,1])]
                center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
                # update keypoints
                part = np.zeros([24,3])
                # part[:14] = np.hstack([joints[:3,:,img_i].T])
                part[:14] = np.hstack([part14, np.ones([14,1])])
                # read openpose detections
                json_file = os.path.join(openpose_folder, 'image_' + imgname.replace('.png', '_keypoints.json'))
                openpose = read_openpose(json_file, part, 'lsp')

                # 3D annotation files
                # annot_file_3d = os.path.join(dataset_path, sub_folder, ("gt_3d/image_"+imgname[:-4]+'.mat'))
                # poses_3d = sio.loadmat(annot_file_3d)['joint_gt_3d']
                # print("--->",np.min(poses_3d[:, 2]), np.max(poses_3d[:, 2]))
                poses_3d = np.zeros([15,3])
                poses_3d[:14, :2] = part14

                depth_uncover_full = join(sub_folder, "depth_aligned/uncover", imgname)
                depth = cv2.imread(os.path.join(dataset_path, depth_uncover_full), 0)
                bed_depth = [178, 180]
                for i in range(part14.shape[0]):
                    poses_3d[i, 2] = depth[int(part14[i, 1]), int(part14[i, 0])] / 255.0
                    if joints[2, i, img_i] == 0:
                        if i < 6:
                            poses_3d[i, 2] = bed_depth[0] / 255.0
                        else:
                            poses_3d[i, 2] = bed_depth[1] / 255.0
                    poses_3d[i, 2] = 1 - poses_3d[i, 2]
                poses_3d[14, :2] = (part14[2, :2] + part14[3, :2])/2
                poses_3d[14, 2] = (poses_3d[2, 2] + poses_3d[3, 2]) /2
                # poses_3d[14, 2] = (depth[int(part14[2, 1]), int(part14[2, 0])] / 255.0 \
                #                 + depth[int(part14[3, 1]), int(part14[3, 0])] / 255.0) / 2
                # poses_3d[:, 2] =(poses_3d[:, 2] - 0.3 )*10 #20 too large ## 2.5 small
                # print(np.min(poses_3d[:, 2]), np.max(poses_3d[:, 2]))

                ## center norm
                c = np.array([1024 / 2., 1024 / 2.], dtype=np.float32)
                poses_3d[:,:2] = poses_3d[:,:2] / c - 1.0

                ## norm with openpose bounding box
                # for i, pt in enumerate(poses_3d):
                #     poses_3d[i, :2] = np.array(transform(poses_3d[i, :2]+1, center, scale, (1024, 1024), invert=0))-1
                # poses_3d[:, :2] = poses_3d[:, :2] / 1024.0 

                # # read GT 3D pose (15 points)
                # S15 = np.reshape(poses_3d, [-1,3])
                # S15 -= S15[14] # root-centered
                # S24 = np.zeros([24,4])
                # S24[global_idx, :3] = S15
                # S24[global_idx, 3] = 1

                # read GT 3D pose (17 points)
                S15 = np.reshape(poses_3d, [-1,3])
                S15[14,:] = np.mean((S15[2,:], S15[3,:]), 0)
                S17 = np.zeros([17,3])
                S17[:15,:] = S15
                
                # S17[15,:] = np.mean((S15[2,:], S15[3,:], S15[8,:], S15[9,:]), 0)
                S17[16,:] = np.mean((S15[12,:], S15[13,:]), 0)

                S17 -= S17[14] # root-centered
                S24 = np.zeros([24,4])
                S24[global_idx_17, :3] = S17
                S24[global_idx_17, 3] = 1
                S24[global_idx_17[15], 3] = 0
                # print(S17)

                # store data
                imgnames_.append(imgname_full)
                irimgnames_.append(ir_imgname_full)
                depthnames_.append(depth_name_full)
                pmnames_.append(pm_name_full)
                centers_.append(center)
                scales_.append(scale)
                parts_.append(part)
                Ss_.append(S24)
                openposes_.append(openpose)
                gender_.append(int(gender_all[sub_ind-1]))

                # print(imgname_full)
                # print(ir_imgname_full)
                img_ori = cv2.imread(os.path.join(SLP_ROOT,imgname_full)).copy()
                # img_ir = cv2.imread(os.path.join(SLP_ROOT,ir_imgname_full)).copy()
                # img_depth = cv2.imread(os.path.join(SLP_ROOT,depth_name_full)).copy()
                # img_pm = cv2.imread(os.path.join(SLP_ROOT,pm_name_full)).copy()
                debugger = Debugger(edges=h36m_edges)
                debugger.add_img(img_ori)
                debugger.add_point_2d(part, (255, 0, 0))
                debugger.add_point_3d(S24[:,:3], 'b')
                debugger.show_all_imgs(pause=False)
                # debugger.add_img(img_ir)
                # debugger.add_img(img_depth)
                # debugger.add_img(img_pm)
                # debugger.show_all_imgs(pause=True)
                debugger.show_3d()
                    
    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, out_name)
    np.savez(out_file, imgname=imgnames_,
                       irimgname=irimgnames_,
                       depthname=depthnames_,
                       pmname=pmnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       S=Ss_,
                       openpose=openposes_,
                       gender=gender_,
                       )



if __name__ == '__main__':
    # SLP_ROOT = '/media/10THD/Dataset/pose/SLP/SLP/danaLab'
    SLP_ROOT = '/media/yuyin/10THD1/Dataset/pose/SLP/SLP/danaLab'
    out_path = './'
    
    ## for generating GT
    # out_name = 'slp_rgb_uncover_train.npz'
    # train_sub_list = range(21,41)
    # # train_sub_list = [1,2,3,4,5,6]
    # # train_sub_list.extend(range(21,41)) 
    # slp_single_mod(SLP_ROOT, out_path, out_name, ['RGB/uncover'], train_sub_list)

    # # ------------ single modality ------------ #
    # ## test data for single modality
    # out_name_single_mod = ['slp_rgb_uncover_test.npz', 'slp_rgb_cover1_test.npz', 'slp_rgb_cover2_test.npz', 
    #             'slp_ir_uncover_test.npz', 'slp_ir_cover1_test.npz', 'slp_ir_cover2_test.npz']
    # img_type = [['RGB/uncover'], ['RGB/cover1'], ['RGB/cover2'],
    #             ['IR_align/uncover'], ['IR_align/cover1'], ['IR_align/cover2']]
    # test_sub_list = range(71, 102) 
    # for i in range(len(out_name_single_mod)):
    #     slp_single_mod(SLP_ROOT, out_path, out_name_single_mod[i], img_type[i], test_sub_list)

    
    # ## training data for single modality
    # out_name_single_mod = ['slp_rgb_train.npz', 'slp_ir_train.npz']
    # img_type = [['RGB/uncover', 'RGB/cover1', 'RGB/cover2'],
    #             ['IR_align/uncover', 'IR_align/cover1', 'IR_align/cover2']]
    # train_sub_list = range(1, 71) 
    # for i in range(len(out_name_single_mod)):
    #     slp_single_mod(SLP_ROOT, out_path, out_name_single_mod[i], img_type[i], train_sub_list)


    # ------- multi-modalities (RGB & IR & depth & PM) ------- #
    ## test data for multi-modalities
    out_name_multi_mod = ['slp_4mod_uncover.npz', 'slp_4mod_cover1.npz', 'slp_4mod_cover2.npz']
    img_type = [['uncover'], ['cover1'], ['cover2']]
    test_sub_list = range(85, 102) 
    for i in range(len(out_name_multi_mod)):
        slp_multi_mod(SLP_ROOT, out_path, out_name_multi_mod[i], img_type[i], test_sub_list)


    ## train data for multi-modalities
    out_name_multi_mod = 'slp_4mod_train.npz'
    img_type = ['uncover', 'cover1', 'cover2']
    train_sub_list = range(1, 85) 
    slp_multi_mod(SLP_ROOT, out_path, out_name_multi_mod, img_type, train_sub_list)

