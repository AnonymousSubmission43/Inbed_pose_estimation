import os
import sys
import cv2
import glob
import h5py
import numpy as np
import argparse
os.environ["CDF_LIB"] = "/media/yuyin/10THD1/3d_recovery/SPIN-master/cdf37_0-dist/lib"
from spacepy import pycdf
from read_openpose import read_openpose
from utils.visualize import Debugger


h36m_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], 
              [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
              [8, 12], [9, 12],
              [2, 8], [3, 9]]

# Illustrative script for training data extraction
# No SMPL parameters will be included in the .npz file.
def h36m_train_extract(dataset_path, openpose_path, out_path, extract_img=False):
    protocol = 1
    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # structs we use
    imgnames_, scales_, centers_, parts_, Ss_, openposes_  = [], [], [], [], [], []

    # users in validation set
    user_list = [1, 5, 6, 7, 8]

    # go over each user
    for user_i in user_list:
        user_name = 'S%d' % user_i
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat', 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D2_Positions')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for seq_i in seq_list:
            print(seq_i)
            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = pycdf.CDF(seq_i)['Pose'][0]

            # 2D pose file
            pose2d_file = os.path.join(pose2d_path, seq_name)
            poses_2d = pycdf.CDF(pose2d_file)['Pose'][0]
            print(pose2d_file)

            # bbox file
            bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            bbox_h5py = h5py.File(bbox_file)

            # video file
            if extract_img:
                vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
                imgs_path = os.path.join(dataset_path, 'images')
                vidcap = cv2.VideoCapture(vid_file)

            # go over each frame of the sequence
            for frame_i in range(poses_3d.shape[0]):
                # read video frame
                if extract_img:
                    success, image = vidcap.read()
                    if not success:
                        break

                # check if you can keep this frame
                if frame_i % 5 == 0 and (protocol == 1 or camera == '60457274'):
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)
                    
                    # save image
                    if extract_img:
                        img_out = os.path.join(imgs_path, imgname)
                        cv2.imwrite(img_out, image)

                    # read GT bounding box
                    mask = bbox_h5py[bbox_h5py['Masks'][frame_i,0]].value.T
                    ys, xs = np.where(mask==1)
                    bbox = np.array([np.min(xs), np.min(ys), np.max(xs)+1, np.max(ys)+1])
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = 0.9*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200.

                    # read GT 3D pose
                    partall = np.reshape(poses_2d[frame_i,:], [-1,2])
                    part17 = partall[h36m_idx]
                    part = np.zeros([24,3])
                    part[global_idx, :2] = part17
                    part[global_idx, 2] = 1

                    # read GT 3D pose
                    Sall = np.reshape(poses_3d[frame_i,:], [-1,3])/1000.
                    S17 = Sall[h36m_idx]
                    S17 -= S17[0] # root-centered
                    S24 = np.zeros([24,4])
                    S24[global_idx, :3] = S17
                    S24[global_idx, 3] = 1
                    
                    # read openpose detections
                    json_file = os.path.join(openpose_path,
                        imgname.replace('.jpg', '_keypoints.json'))
                    openpose = read_openpose(json_file, part, 'h36m')

                    # store data
                    imgnames_.append(os.path.join('images', imgname))
                    centers_.append(center)
                    scales_.append(scale)
                    parts_.append(part)
                    Ss_.append(S24)
                    openposes_.append(openpose)

                    img_out = os.path.join(dataset_path, 'images', imgname)
                    img = cv2.imread(img_out)
                    # aa = np.reshape(poses_3d[frame_i,:], [-1,3])
                    # for i in range(17):
                    #     print(1000*aa[i,0],1000*aa[i,1])
                    #     img = cv2.circle(img, (int(1000*S24[i,0]),int(1000*S24[i,1])), radius=2, color=(0, 0, 255), thickness=-1)
                    # img = cv2.circle(img, (int(1000*S24[1,0]),int(1000*S24[1,1])), radius=2, color=(0, 0, 255), thickness=-1)
                    # cv2.imshow('Example',img)
                    # cv2.waitKey(0)

                    # print("+"*60)
                    # print(S17)
                    # print("+"*60)
                    # print("-"*60)
                    # print(S24)
                    # print("-"*60)

                    print(img_out)
                    img_ori = cv2.imread(os.path.join(H36M_ROOT,dataset_path, 'images', imgname)).copy()
                    debugger = Debugger(edges=h36m_edges)
                    debugger.add_img(img_ori)
                    debugger.add_point_2d(part, (255, 0, 0))
                    debugger.add_point_3d(S24[:,:3], 'b')
                    debugger.show_all_imgs(pause=True)
                    debugger.show_3d()

    # # store the data struct
    # if not os.path.isdir(out_path):
    #     os.makedirs(out_path)
    # out_file = os.path.join(out_path, 'h36m_train_nosmpl.npz')
    # np.savez(out_file, imgname=imgnames_,
    #                    center=centers_,
    #                    scale=scales_,
    #                    part=parts_,
    #                    S=Ss_,
    #                    openpose=openposes_)


if __name__ == '__main__':
    H36M_ROOT = '/media/yuyin/10THD1/Dataset/pose/human36m'
    openpose_path = '/media/yuyin/10THD1/Dataset/pose/human36m/openpose_results'
    out_path = './'
    h36m_train_extract(H36M_ROOT, openpose_path, out_path, extract_img=False)