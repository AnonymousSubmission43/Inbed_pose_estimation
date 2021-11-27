"""
This script can be used to evaluate a trained model on 3D pose/shape and masks/part segmentation. You first need to download the datasets and preprocess them.
Example usage:
```
python3 eval.py --checkpoint=data/model_checkpoint.pt --dataset=h36m-p1 --log_freq=20
python eval.py --checkpoint logs/train_slp_3d_26sub/checkpoints/2020_04_22-21_35_20.pt  --dataset slp-rgb-cover1
```
Running the above command will compute the MPJPE and Reconstruction Error on the Human3.6M dataset (Protocol I). The ```--dataset``` option can take different values based on the type of evaluation you want to perform:
1. Human3.6M Protocol 1 ```--dataset=h36m-p1```
2. Human3.6M Protocol 2 ```--dataset=h36m-p2```
3. 3DPW ```--dataset=3dpw```
4. LSP ```--dataset=lsp```
5. MPI-INF-3DHP ```--dataset=mpi-inf-3dhp```
6. SLP
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import datetime
import argparse
import json
from collections import namedtuple
from tqdm import tqdm
import torchgeometry as tgm

import config
import constants
from models import hmr
from models.smpl import SMPL
from datasets import BaseDataset
from utils.imutils import uncrop
from utils.pose_utils import reconstruction_error
from utils.part_utils import PartRenderer
from utils.geometry import perspective_projection

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='hmr', help='hmr, mulhmr, featcat')
parser.add_argument('--checkpoint', default=None, help='Path to network checkpoint')
parser.add_argument('--dataset', default='slp-4mod-uncover', help="Choose evaluation dataset:['slp-rgb-uncover', 'h36m','h36m-p1', 'h36m-p2', 'lsp', '3dpw', 'mpi-inf-3dhp']")
parser.add_argument('--log_freq', default=50, type=int, help='Frequency of printing intermediate results')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for testing')
parser.add_argument('--shuffle', default=False, action='store_true', help='Shuffle data')
parser.add_argument('--num_workers', default=8, type=int, help='Number of processes for data loading')
parser.add_argument('--result_file', default=None, help='If set, save detections to a .npz file')

def run_evaluation(model, dataset_name, dataset, result_file, checkpoint_dir=None, epoch=0, 
    batch_idx=None, batch_size=32, img_res=224, num_workers=32, shuffle=False, log_freq=50, 
    model_name='mulhmr', no_render=False, num_cas_iters=2, mod1_epoch=50, pretrained_ir_depth_model=None):
    """Run evaluation on the datasets and metrics we report in the paper. """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    focal_length=constants.FOCAL_LENGTH

    # Transfer model to the GPU
    model.to(device)

    # Load SMPL model
    smpl_neutral = SMPL(config.SMPL_MODEL_DIR,
                        create_transl=False).to(device)
    smpl_male = SMPL(config.SMPL_MODEL_DIR,
                     gender='male',
                     create_transl=False).to(device)
    smpl_female = SMPL(config.SMPL_MODEL_DIR,
                       gender='female',
                       create_transl=False).to(device)
    
    part_renderer = PartRenderer() 
    if no_render == False:
        from utils.renderer import Renderer
        renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl_neutral.faces)


    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    
    save_results = result_file is not None
    # Disable shuffling if you want to save the results
    if save_results:
        shuffle=False
        if not os.path.exists(result_file):
            os.mkdir(result_file)
        img_savepath = os.path.join(result_file, dataset_name)
        if not os.path.exists(img_savepath):
            os.mkdir(img_savepath)
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    # Pose metrics
    # MPJPE and Reconstruction error for the non-parametric and parametric shapes
    mpjpe = np.zeros(len(dataset))
    recon_err = np.zeros(len(dataset))
    mpjpe_smpl = np.zeros(len(dataset))
    recon_err_smpl = np.zeros(len(dataset))

    # Shape metrics
    # Mean per-vertex error
    shape_err = np.zeros(len(dataset))
    shape_err_smpl = np.zeros(len(dataset))

    # Mask and part metrics
    # Accuracy
    accuracy = 0.
    parts_accuracy = 0.
    # True positive, false positive and false negative
    tp = np.zeros((2,1))
    fp = np.zeros((2,1))
    fn = np.zeros((2,1))
    parts_tp = np.zeros((7,1))
    parts_fp = np.zeros((7,1))
    parts_fn = np.zeros((7,1))
    # Pixel count accumulators
    pixel_count = 0
    parts_pixel_count = 0

    # Store SMPL parameters
    smpl_pose = np.zeros((len(dataset), 72))
    smpl_betas = np.zeros((len(dataset), 10))
    smpl_camera = np.zeros((len(dataset), 3))
    pred_joints = np.zeros((len(dataset), 17, 3))

    eval_pose = False
    eval_masks = False
    eval_parts = False
    save_recovered_img = False
    # Choose appropriate evaluation for each dataset
    if dataset_name.find('slp') == 0:
        eval_pose = True
        eval_masks = True
    if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp':
        eval_pose = True
    elif dataset_name == 'lsp':
        eval_masks = True
        eval_parts = True
        annot_path = config.DATASET_FOLDERS['upi-s1h']

    # joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    # joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    joint_mapper_h36m = constants.H36M_TO_J17 #if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
    joint_mapper_gt = constants.J24_TO_J17 #if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
        # Get ground truth annotations from the batch
        gt_pose = batch['pose'].to(device)
        gt_betas = batch['betas'].to(device)
        gt_vertices = smpl_neutral(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]).vertices
        images = batch['img'].to(device)
        gender = batch['gender'] # 0 male, 1 female
        # smpl_neutral
        curr_batch_size = images.shape[0]

        with torch.no_grad():
            ir_img = batch['ir_img'].to(device)
            depth_img = batch['depth_img'].to(device)
            pm_img = batch['pm_img'].to(device)
            if model_name == "rechmr":
                pred_rotmat, pred_betas, pred_camera, pred_depth = model([images, ir_img, depth_img, pm_img])
            elif model_name == "rec3hmr":
                pred_rotmat, pred_betas, pred_camera, pred_depth, pred_ir, pred_pm  = model([images, ir_img, depth_img, pm_img])
            elif model_name in ["cashmr", "featcat_cashmr","featatt_cashmr"]:
                pred_rotmat_1, pred_betas_1, pred_camera_1, pred_depth_1 = model([images, ir_img, depth_img, pm_img])
                pred_rotmat, pred_betas, pred_camera, pred_depth = model([images, ir_img, pred_depth_1, pm_img])

            elif model_name in ["cashmrV2"]:
                pred_rotmat_temp, pred_betas_temp, pred_camera_temp, pred_depth_temp = model([images, ir_img, depth_img, pm_img])
                for cas_iter in range(num_cas_iters-1):
                    pred_pose_1 = pred_rotmat_temp[:,:,:,1:].contiguous().view(curr_batch_size,144)
                    pred_rotmat, pred_betas, pred_camera, pred_depth = model(
                        [images, ir_img, pred_depth_temp, pm_img])
                        # [images, ir_img, pred_depth_temp, pm_img], pred_pose_1, pred_betas_temp, pred_camera_temp)
                    pred_depth_temp = pred_depth
            elif model_name in ["ir_depth_featatt_cashmrV2"]:
                pred_rotmat_temp, pred_betas_temp, pred_camera_temp, pred_depth_temp, pred_ir_temp = model([ir_img, depth_img])
                for cas_iter in range(num_cas_iters-1):
                    pred_pose_1 = pred_rotmat_temp[:,:,:,1:].contiguous().view(curr_batch_size,144)
                    pred_rotmat, pred_betas, pred_camera, pred_depth, pred_ir = model([pred_ir_temp, pred_depth_temp])
                    pred_depth_temp = pred_depth
                    pred_ir_temp = pred_ir
            elif model_name == "mulhmr":
                # pred_rotmat, pred_betas, pred_camera = model([ir_img, depth_img])
                pred_rotmat, pred_betas, pred_camera = model([ir_img, depth_img, pm_img])
                # pred_rotmat, pred_betas, pred_camera = model([ir_img, depth_img, pm_img, images])
            elif model_name == "hmr":
                pred_rotmat, pred_betas, pred_camera = model(images)
            elif model_name == "hmr4mod":
                pred_rotmat, pred_betas, pred_camera = model(torch.cat([images, ir_img, depth_img, pm_img], 1))
            elif model_name == "irhmr":
                pred_rotmat, pred_betas, pred_camera = model(ir_img)
            elif model_name == "depthhmr":
                pred_rotmat, pred_betas, pred_camera = model(depth_img)
            elif model_name == "pmhmr":
                pred_rotmat, pred_betas, pred_camera = model(pm_img)
            elif model_name == "ir_depth_fusion":
                _, _, _, pred_rotmat, pred_betas, pred_camera, ir_out, depth_out, mask_l = model([ir_img, depth_img], smpl_neutral)
                # pred_rotmat_1, pred_betas_1, pred_camera_1, pred_depth_1 = model([ ir_img, depth_img])
                # pred_rotmat, pred_betas, pred_camera, pred_depth = model([ir_img, pred_depth_1])
                save_recovered_img = True
            elif model_name == "ir_pm_fusion":
                _, _, _, pred_rotmat, pred_betas, pred_camera, ir_out, depth_out, mask_l = model([ir_img, pm_img], smpl_neutral)
                save_recovered_img = True

            elif model_name in ["rgb_depth_fusion", "rgb_pm_fusion"]:
                if model_name == "rgb_depth_fusion":
                    input_2 = depth_img
                elif model_name == "rgb_pm_fusion":
                    input_2 = pm_img
                _, _, _, pred_rotmat, pred_betas, pred_camera, depth_out, mask_l = model(
                    [images, input_2], smpl_neutral)
                save_recovered_img = False


            elif model_name == "ir_depth_pm_fusion":
                _,_,_,pred_pose_ori, pred_betas_ori, pred_camera_ori, ir_out_ori, depth_out_ori,_ = \
                pretrained_ir_depth_model([ir_img, depth_img], smpl_neutral, return_pose=True)
                # _, _, _, pred_rotmat, pred_betas, pred_camera, ir_out, depth_out, pm_out, mask_l = model(
                _, _, _, pred_rotmat, pred_betas, pred_camera, ir_out, depth_out, mask_l = model(
                    [ir_out_ori, depth_out_ori, pm_img, ir_img, depth_img], smpl_neutral, )
                    # init_pose=pred_pose_ori, init_shape=pred_betas_ori, init_cam=pred_camera_ori)
                save_recovered_img = True

            # ''' bodiesAtRest
            # '''
            elif model_name in ["bodiesAtRest", "bodiesAtRest4mod"]:
                pm_contact = batch['pm_contact'].to(device)
                if model_name == "bodiesAtRest":
                    input_batch = torch.cat([pm_img, pm_contact], 1)
                else:
                    input_batch = torch.cat([images, ir_img, depth_img, pm_img, pm_contact], 1)
                pred_rotmat, pred_betas, pred_camera, pred_pose = model(images=input_batch)
                
                if model_name in ["bodiesAtRest4mod"]:
                # if epoch >= mod1_epoch:
                    pred_output_mod1 = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
                    pred_joints_mod1 = pred_output_mod1.joints

                    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
                    # This camera translation can be used in a full perspective projection
                    pred_cam_t = torch.stack([pred_camera[:,1],
                                              pred_camera[:,2],
                                              2*focal_length/(img_res * pred_camera[:,0] +1e-9)],dim=-1)


                    camera_center = torch.zeros(curr_batch_size, 2, device=device)
                    pred_keypoints_3d = perspective_projection(pred_joints_mod1,
                                                               rotation=torch.eye(3, device=device).unsqueeze(0).expand(curr_batch_size, -1, -1),
                                                               translation=pred_cam_t,
                                                               focal_length=focal_length,
                                                               camera_center=camera_center, out_3d=True)
                    # Normalize keypoints to [-1,1]
                    pred_keypoints_3d[:,:,:-1] = pred_keypoints_3d[:,:,:-1] + 0.5 * img_res


                    '''
                    project vertices to plane
                    '''
                    padding_x = 10 
                    padding_y = 100 
                    masks = torch.zeros([curr_batch_size, 1, img_res+padding_y*2, img_res+padding_x*2]).to(device)
                    for i, mask in enumerate(masks): 
                        x = pred_keypoints_3d[i,:,0].type(torch.LongTensor) + padding_x
                        y = pred_keypoints_3d[i,:,1].type(torch.LongTensor) + padding_y
                        z = pred_keypoints_3d[i,:,2]
                        masks[i, 0, y, x] = 1 # z
                    masks[:,:,2:-2,2:-2] = (masks[:,:,:-4,:-4] + masks[:,:,1:-3,:-4] + masks[:,:,2:-2,:-4] + masks[:,:,3:-1,:-4] + masks[:,:,4:,:-4] \
                                        + masks[:,:,:-4,1:-3] + masks[:,:,1:-3,1:-3] + masks[:,:,2:-2,1:-3] + masks[:,:,3:-1,1:-3] + masks[:,:,4:,1:-3] \
                                        + masks[:,:,:-4,2:-2] + masks[:,:,1:-3,2:-2] + masks[:,:,2:-2,2:-2] + masks[:,:,3:-1,2:-2] + masks[:,:,4:,2:-2] \
                                        + masks[:,:,:-4,3:-1] + masks[:,:,1:-3,3:-1] + masks[:,:,2:-2,3:-1] + masks[:,:,3:-1,3:-1] + masks[:,:,4:,3:-1] \
                                        + masks[:,:,:-4,4:]   + masks[:,:,1:-3,4:]   + masks[:,:,2:-2,4:]   + masks[:,:,3:-1,4:]   + masks[:,:,4:,4:])/25.0
                    masks = masks[:,:,padding_y:img_res+padding_y,padding_x:img_res+padding_x]
                    masks[masks > 0] = 1

                    # plt.figure()
                    # masks_show = masks.clone()
                    # plt.imshow(masks_show.cpu().detach().numpy()[0][0])
                    # plt.show()
                    if model_name == "bodiesAtRest":
                        all_input = torch.cat([pm_img, pm_contact, masks], 1)
                    else:
                        all_input = torch.cat([images, ir_img, depth_img, pm_img, pm_contact, masks], 1)
                    pred_rotmat, pred_betas, pred_camera = model(
                        images=all_input, mode="2", init_pose=pred_pose, init_shape=pred_betas, init_cam=pred_camera)

            pred_output = smpl_neutral(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices


        if save_results:
            ## for large render
            renderer_scale = 4
            renderer_large = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES*renderer_scale, faces=smpl_neutral.faces)
            camera_translation_all_large = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES* renderer_scale * pred_camera[:,0] +1e-9)],dim=-1) #torch.Size([Batch, 3])

            # Calculate camera parameters for rendering
            camera_translation_all = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1) #torch.Size([Batch, 3])

            # Calculate smpl parameters
            rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
            rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
            pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
            smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
            smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
            smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()
           
            pred_vertices_all = pred_vertices.clone() #torch.Size([Batch, 6890, 3])            
            for img_ind in range(len(batch)):
                camera_translation = camera_translation_all[img_ind,:].cpu().numpy()
                pred_vertices_save = pred_vertices_all[img_ind, :, :].cpu().numpy()
                img = batch['img'][img_ind].permute(1,2,0).cpu().numpy()
                img = img * constants.IMG_NORM_STD + constants.IMG_NORM_MEAN
                img_uncover = batch['img_uncover'][img_ind].permute(1,2,0).cpu().numpy()

                if no_render == False:
                    # Render parametric shape
                    img_shape = renderer(pred_vertices_save, camera_translation, img)
                    
                    # Render side views
                    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0]
                    center_r = pred_vertices_save.mean(axis=0)

                    rot_vertices = np.dot((pred_vertices_save - center_r), aroundy) + center_r
                    # Render non-parametric shape
                    img_shape_side = renderer(rot_vertices, camera_translation)

                    # Render top views
                    aroundy2 = cv2.Rodrigues(np.array([-np.radians(90.), 0, 0]))[0]
                    rot_vertices2 = np.dot((pred_vertices_save - center_r), aroundy2) + center_r
                    # Render non-parametric shape
                    img_shape_top = renderer(rot_vertices2, camera_translation)

                    # Save reconstructions
                    name_list = batch['imgname'][img_ind].split('/')
                    outfile = '%s/%s_%s' % (img_savepath, name_list[-4], name_list[-1][:-4])
                    cv2.imwrite(outfile + '_shape.png', 255 * img_shape[:,:,::-1])
                    # cv2.imwrite(outfile + '_shape_side.png', 255 * img_shape_side[:,:,::-1])
                    # cv2.imwrite(outfile + '_shape_top.png', 255 * img_shape_top[:,:,::-1])

                    # # Save ref image 
                    # cv2.imwrite(outfile + '_ref.png', 255 * img_uncover[:,:,::-1])

                    # ## large mesh
                    # camera_translation_large = camera_translation_all_large[img_ind,:].cpu().numpy()
                    # img_shape_large = renderer_large(pred_vertices_save, camera_translation_large)
                    # cv2.imwrite(outfile + '_large.png', 255 * img_shape_large[:,:,::-1])


                    if save_recovered_img:
                        black = (img[:,:,0]<0.0001)
                        ir_save = ir_out[img_ind,0,:,:].cpu().numpy()
                        ir_save = 255 * (ir_save * constants.IR_NORM_STD[0] + constants.IR_NORM_MEAN[0])
                        ir_save = cv2.applyColorMap(ir_save.astype(np.uint8), cv2.COLORMAP_HOT)
                        ir_save[black] = 0

                        depth_save = depth_out[img_ind,0,:,:].cpu().numpy()
                        depth_save = (255 * (depth_save * constants.DEPTH_NORM_STD[0] + constants.DEPTH_NORM_MEAN[0])).astype(np.uint8)
                        bg = (depth_save>220)
                        bed = (depth_save<220)
                        depth_save[bg] = (depth_save[bg] )
                        depth_save[bed] = (depth_save[bed] - 150) * 3
                        depth_save[black] = (depth_save[black]) * 0

                        depth_save2 = depth_save.copy()
                        depth_save2 = cv2.applyColorMap(depth_save2, cv2.COLORMAP_OCEAN) #COLORMAP_OCEAN, COLORMAP_BONE, COLORMAP_RAINBOW
                        depth_save2[black] =0

                        cv2.imwrite(outfile + '_irout.png', ir_save)
                        cv2.imwrite(outfile + '_depthoutori.png', depth_save)
                        cv2.imwrite(outfile + '_depthout.png', depth_save2)

                        # print(mask_l.size())
                        mask_save = mask_l[img_ind,0,:,:].cpu().numpy()
                        cv2.imwrite(outfile + '_mask.png', mask_save)

        # 3D pose evaluation
        if eval_pose:
            # Regressor broadcasting
            J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
            # Get 14 ground truth joints
            if 'h36m' in dataset_name or 'mpi-inf' in dataset_name or 'slp' in dataset_name:
                gt_keypoints_3d = batch['pose_3d'].cuda()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            # For 3DPW get the 14 common joints from the rendered shape
            else:
                gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis 


            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
            if save_results:
                pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :]  = pred_keypoints_3d.cpu().numpy()
            pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis 

            # Absolute error (MPJPE)
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

            # Reconstuction_error
            r_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
            recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error


        # If mask or part evaluation, render the mask and part images
        if eval_masks or eval_parts:
            mask, parts = part_renderer(pred_vertices, pred_camera)

        # Mask evaluation (for LSP)
        if eval_masks:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            # Dimensions of original image
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                # After rendering, convert imate back to original resolution
                pred_mask = uncrop(mask[i].cpu().numpy(), center[i], scale[i], orig_shape[i]) > 0
                # Load gt mask
                maskname = batch['imgname'][i].replace("RGB", "masks").replace("cover1", "uncover").replace("cover2", "uncover").replace("image_", "")
                gt_mask = cv2.imread(os.path.join(config.DATA_ROOT, maskname), 0) > 0
                # gt_mask = batch['mask_uncover'][i][0].cpu().numpy() > 0
                # print((gt_mask == pred_mask).shape)
                # Evaluation consistent with the original UP-3D code
                accuracy += (gt_mask == pred_mask).sum()
                pixel_count += np.prod(np.array(gt_mask.shape))
                for c in range(2):
                    cgt = gt_mask == c
                    cpred = pred_mask == c
                    tp[c] += (cgt & cpred).sum()
                    fp[c] +=  (~cgt & cpred).sum()
                    fn[c] +=  (cgt & ~cpred).sum()
                f1 = 2 * tp / (2 * tp + fp + fn)

        # Part evaluation (for LSP)
        if eval_parts:
            center = batch['center'].cpu().numpy()
            scale = batch['scale'].cpu().numpy()
            orig_shape = batch['orig_shape'].cpu().numpy()
            for i in range(curr_batch_size):
                pred_parts = uncrop(parts[i].cpu().numpy().astype(np.uint8), center[i], scale[i], orig_shape[i])
                # Load gt part segmentation
                gt_parts = cv2.imread(os.path.join(annot_path, batch['partname'][i]), 0)
                # Evaluation consistent with the original UP-3D code
                # 6 parts + background
                for c in range(7):
                   cgt = gt_parts == c
                   cpred = pred_parts == c
                   cpred[gt_parts == 255] = 0
                   parts_tp[c] += (cgt & cpred).sum()
                   parts_fp[c] +=  (~cgt & cpred).sum()
                   parts_fn[c] +=  (cgt & ~cpred).sum()
                gt_parts[gt_parts == 255] = 0
                pred_parts[pred_parts == 255] = 0
                parts_f1 = 2 * parts_tp / (2 * parts_tp + parts_fp + parts_fn)
                parts_accuracy += (gt_parts == pred_parts).sum()
                parts_pixel_count += np.prod(np.array(gt_parts.shape))

        # Print intermediate results during evaluation
        if step % log_freq == log_freq - 1:
            if eval_pose:
                print('MPJPE: ' + str(1000 * mpjpe[:step * batch_size].mean()))
                print('Reconstruction Error: ' + str(1000 * recon_err[:step * batch_size].mean()))
                print()
            if eval_masks:
                print('Accuracy: ', accuracy / pixel_count)
                print('F1: ', f1.mean())
                print()
            if eval_parts:
                print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
                print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
                print()

    # Save reconstructions to a file for further processing
    if save_results:
        # np.save('data/static_fits/{}_fits.npy'.format(dataset_name),  np.concatenate((smpl_pose, smpl_betas), axis=1))
        smpl_savepath = f"{result_file}/smpl_fits"
        if not os.path.exists(smpl_savepath):
            os.mkdir(smpl_savepath)
        np.savez('{}/{}_fits.npy'.format(smpl_savepath, dataset_name), pred_joints=pred_joints, pose=smpl_pose, betas=smpl_betas, camera=smpl_camera)

    if eval_pose:
        print(f'{dataset_name}: MPJPE: ' + str(1000 * mpjpe.mean()))
        print('\tReconstruction Error: ' + str(1000 * recon_err.mean()))
        if dataset_name.find('cover2'):
            print()

        # save log to file
        if checkpoint_dir:
            log_dir = os.path.join(checkpoint_dir, 'log.txt')
            open_type = 'a' if os.path.exists(log_dir) else 'w'

            with open(log_dir, open_type) as f:
                f.write(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + 
                    '\t[epoch: ' + str(epoch) + '], batch_idx: ' + str(batch_idx) + '\n')
                f.write(f'{dataset_name}\tMPJPE: ' + str(1000 * mpjpe.mean()))
                f.write('\tReconstruction Error: ' + str(1000 * recon_err.mean()))
                if eval_masks:
                    f.write('\tFB Accuracy: ' + str(accuracy / pixel_count))
                    f.write('\tFB F1: ' + str(f1.mean()))
                f.write('\n')
                if dataset_name.find('cover2'):
                    f.write('\n')

    if eval_masks:
        print('Accuracy: ', accuracy / pixel_count)
        print('F1: ', f1.mean())
        print()

    if eval_parts:
        print('Parts Accuracy: ', parts_accuracy / parts_pixel_count)
        print('Parts F1 (BG): ', parts_f1[[0,1,2,3,4,5,6]].mean())
        print()

if __name__ == '__main__':
    args = parser.parse_args()
    model = hmr(config.SMPL_MEAN_PARAMS, model_name=args.model)
    # model = model.cuda()
    model = nn.DataParallel(model).cuda()

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if args.model in ["ir_depth_pm_fusion", "ir_depth_pm_rgb_fusion"]:
        pretrained_ir_depth_model = hmr(config.SMPL_MEAN_PARAMS, model_name="ir_depth_fusion")
        checkpoint_ir_depth = torch.load("./logs/IR_DEPTH_FUSION_11/checkpoints/epoch_83_0.pt")
        pretrained_ir_depth_model = nn.DataParallel(pretrained_ir_depth_model).to('cuda')
        pretrained_ir_depth_model.load_state_dict(checkpoint_ir_depth['model'], strict=True)
        
        # if args.model == "ir_depth_pm_rgb_fusion":
        #     pretrained_ir_depth_pm_model = hmr(config.SMPL_MEAN_PARAMS, model_name="ir_depth_pm_fusion")
        #     checkpoint_ir_depth_pm = torch.load("./logs/irdepthpm_test/checkpoints/epoch_127_0.pt")
        #     pretrained_ir_depth_pm_model.load_state_dict(checkpoint_ir_depth_pm['model'], strict=True)
        #     pretrained_ir_depth_pm_model = nn.DataParallel(pretrained_ir_depth_pm_model).to('cuda')
        pretrained_ir_depth_model.eval()
    else:
        pretrained_ir_depth_model=None


    # Setup evaluation dataset
    test_datasets = ["slp-4mod-cover2", "slp-4mod-uncover", "slp-4mod-cover1" ]
    for d in test_datasets:
        dataset_loader = BaseDataset(None, d, is_train=False)
        # Run evaluation
        run_evaluation(model, d, dataset_loader, args.result_file,
                       batch_size=args.batch_size,
                       shuffle=args.shuffle,
                       log_freq=args.log_freq,
                       model_name=args.model,
                       pretrained_ir_depth_model=pretrained_ir_depth_model)
        # run_evaluation(model, d, dataset_loader, 
        #                        args.result_file, args.result_file, 10, 10, 
        #                        model_name=args.model, 
        #                        num_cas_iters=2, 
        #                        mod1_epoch=50)
