import torch
import torch.nn as nn
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2

from datasets import MixedDataset
from models import hmr, SMPL
from smplify import SMPLify
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, rotmat_to_rot6d
# from utils.renderer import Renderer
from utils import BaseTrainer

import config
import constants
from .fits_dict import FitsDict

from matplotlib import rc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


class Trainer(BaseTrainer):
    
    def init_fn(self):
        self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, is_train=True)
        self.model_name = self.options.model
        self.num_cas_iters = self.options.num_cas_iters
        if self.model_name in ["ir_depth_pm_fusion", "ir_depth_pm_rgb_fusion"]:
            self.pretrained_ir_depth_model = hmr(config.SMPL_MEAN_PARAMS, model_name="ir_depth_fusion")
            checkpoint = torch.load("./logs/IR_DEPTH_FUSION_11/checkpoints/epoch_82_0.pt")
            self.pretrained_ir_depth_model.load_state_dict(checkpoint['model'], strict=Ture)
            self.pretrained_ir_depth_model = nn.DataParallel(self.pretrained_ir_depth_model).to(self.device)
            if self.model_name == "ir_depth_pm_rgb_fusion":
                self.pretrained_ir_depth_pm_model = hmr(config.SMPL_MEAN_PARAMS, model_name="ir_depth_pm_fusion")
                checkpoint = torch.load("./logs/irdepthpm_test/checkpoints/epoch_127_0.pt")
                self.pretrained_ir_depth_pm_model.load_state_dict(checkpoint['model'], strict=True)
                self.pretrained_ir_depth_pm_model = nn.DataParallel(self.pretrained_ir_depth_pm_model).to(self.device)

        self.model = hmr(config.SMPL_MEAN_PARAMS, model_name=self.options.model, pretrained=True)
        self.model = nn.DataParallel(self.model).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.options.lr,
                                          weight_decay=0)
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False).to(self.device)
        # self.smpl = nn.DataParallel(self.smpl)
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        # Loss for img recovery
        self.criterion_imgRec = nn.L1Loss().to(self.device)

        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH
        self.IMG_NORM_MEAN = constants.IMG_NORM_MEAN
        self.IMG_NORM_STD = constants.IMG_NORM_STD
        self.DEPTH_NORM_MEAN = constants.DEPTH_NORM_MEAN
        self.DEPTH_NORM_STD = constants.DEPTH_NORM_STD
        self.IR_NORM_MEAN = constants.IR_NORM_MEAN
        self.IR_NORM_STD = constants.IR_NORM_STD
        self.PM_NORM_MEAN = constants.PM_NORM_MEAN
        self.PM_NORM_STD = constants.PM_NORM_STD

        # Initialize SMPLify fitting module
        self.smplify = SMPLify(step_size=1e-2, batch_size=self.options.batch_size, num_iters=self.options.num_smplify_iters, focal_length=self.focal_length)
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        # Load dictionary of fits
        self.fits_dict = FitsDict(self.options, self.train_ds)

        # Create renderer
        if self.options.no_render == False:
            from utils.renderer import Renderer
            self.renderer = Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)

    def finalize(self):
        self.fits_dict.save()

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def depth_loss(self, pred_img, gt_img, mask=None):
        """Compute img recovery loss.
        The loss is weighted by the confidence.
        """
        # threshold = (0.800 - self.DEPTH_NORM_MEAN[0]) / self.DEPTH_NORM_STD[0]
        # pred_ind = covered_img.clone().detach().cpu().numpy()
        # gt_ind = gt_img.clone().detach().cpu().numpy()
        # mask = torch.Tensor((pred_ind < threshold) | (pred_ind != 0) & (gt_ind != 0)).to(self.device)
        
        # for i in range(pred_img.size(0)):
        #     pred_ind = covered_img.clone().detach().cpu().numpy()[i,0,:,:]
        #     mask.append((pred_ind < threshold) & (pred_ind != 0) & (pred_ind != 0))
        # mask = torch.Tensor(mask).unsqueeze(1).to(self.device)

        # pred_img[i, 0, (pred_ind > threshold) | (pred_ind == 0) | (pred_ind == 0)] = 0
        # gt_img[i, 0, (gt_ind > threshold) | (gt_ind == 0) | (gt_ind == 0)] = 0

        # gtimg = masked_gt_img[0].permute(1, 2, 0).detach().cpu().numpy()
        # gtimg = gtimg * self.DEPTH_NORM_STD[0] + self.DEPTH_NORM_MEAN[0]
        # cv2.imshow('masked_gt.png', gtimg)

        # cv2.waitKey(0)

        if mask is not None:
            masked_pred_img = pred_img * mask
            masked_gt_img = gt_img * mask
            return self.criterion_imgRec(masked_pred_img, masked_gt_img)
        else:
            return self.criterion_imgRec(pred_img, gt_img)



    def reconstraction_loss(self, pred_img, gt_img):
        """Compute img recovery loss.
        The loss is weighted by the confidence.
        """

        return self.criterion_imgRec(pred_img, gt_img)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def get_vertices(self, pred_rotmat_1, pred_betas_1, pred_camera_1, batch_size):
        pred_output_1 = self.smpl(betas=pred_betas_1, body_pose=pred_rotmat_1[:,1:], global_orient=pred_rotmat_1[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices_1 = pred_output_1.vertices
        pred_joints_1 = pred_output_1.joints

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t_1 = torch.stack([pred_camera_1[:,1],
                                  pred_camera_1[:,2],
                                  2*self.focal_length/(self.options.img_res * pred_camera_1[:,0] +1e-9)],dim=-1)


        camera_center = torch.zeros(batch_size, 2, device=self.device)
        pred_keypoints_2d_1 = perspective_projection(pred_joints_1,
                                                   rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                   translation=pred_cam_t_1,
                                                   focal_length=self.focal_length,
                                                   camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d_1 = pred_keypoints_2d_1 / (self.options.img_res / 2.)

        return pred_vertices_1, pred_joints_1, pred_keypoints_2d_1 


    def train_step(self, input_batch, epoch):
        self.model.train()

        # Get data from the batch
        images = input_batch['img'].to(self.device) # input image
        gt_keypoints_2d = input_batch['keypoints'].to(self.device) # 2D keypoints
        gt_pose = input_batch['pose'].to(self.device) # SMPL pose parameters
        gt_betas = input_batch['betas'].to(self.device) # SMPL beta parameters
        gt_joints = input_batch['pose_3d'].to(self.device) # 3D pose
        has_smpl = input_batch['has_smpl'].byte().to(self.device) # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].byte().to(self.device) # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from
        indices = input_batch['sample_index'] # index of example inside its dataset
        batch_size = images.shape[0]

        # Get GT vertices and model joints
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # Get current best fits from the dictionary
        opt_pose, opt_betas = self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu())]
        opt_pose = opt_pose.to(self.device)
        opt_betas = opt_betas.to(self.device)
        opt_output = self.smpl(betas=opt_betas, body_pose=opt_pose[:,3:], global_orient=opt_pose[:,:3])
        opt_vertices = opt_output.vertices
        opt_joints = opt_output.joints


        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)

        opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)


        opt_joint_loss = self.smplify.get_fitting_loss(opt_pose, opt_betas, opt_cam_t,
                                                       0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device),
                                                       gt_keypoints_2d_orig).mean(dim=-1)

        # Feed images in the network to predict camera and SMPL parameters
        ir_img = input_batch['ir_img'].to(self.device)
        depth_img = input_batch['depth_img'].to(self.device)
        pm_img = input_batch['pm_img'].to(self.device)
        if self.model_name == "rechmr":
            pred_rotmat, pred_betas, pred_camera, pred_depth = self.model([images, ir_img, depth_img, pm_img])
        elif self.model_name == "rec3hmr":
            pred_rotmat, pred_betas, pred_camera, pred_depth, pred_ir, pred_pm = self.model([images, ir_img, depth_img, pm_img])
        elif self.model_name in ["cashmr", "featcat_cashmr", "featatt_cashmr"]:
            pred_rotmat_1, pred_betas_1, pred_camera_1, pred_depth_1 = self.model([images, ir_img, depth_img, pm_img])
            pred_rotmat, pred_betas, pred_camera, pred_depth = self.model(
                [images, ir_img, pred_depth_1, pm_img])
                # [images, ir_img, pred_depth_1, pm_img], init_pose=rotmat_to_rot6d(pred_rotmat_1), init_shape=pred_betas_1, init_cam=pred_camera_1)

            # for step 1
            pred_output_1 = self.smpl(betas=pred_betas_1, body_pose=pred_rotmat_1[:,1:], global_orient=pred_rotmat_1[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices_1 = pred_output_1.vertices
            pred_joints_1 = pred_output_1.joints

            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t_1 = torch.stack([pred_camera_1[:,1],
                                      pred_camera_1[:,2],
                                      2*self.focal_length/(self.options.img_res * pred_camera_1[:,0] +1e-9)],dim=-1)


            camera_center = torch.zeros(batch_size, 2, device=self.device)
            pred_keypoints_2d_1 = perspective_projection(pred_joints_1,
                                                       rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                       translation=pred_cam_t_1,
                                                       focal_length=self.focal_length,
                                                       camera_center=camera_center)
            # Normalize keypoints to [-1,1]
            pred_keypoints_2d_1 = pred_keypoints_2d_1 / (self.options.img_res / 2.)


            # '''
            # project vertices to plane
            # '''
            # ## project vertices to plane
            # projected_vertices_3d = perspective_projection(pred_vertices_1,
            #                                    rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
            #                                    translation=pred_cam_t_1,
            #                                    focal_length=self.focal_length,
            #                                    camera_center=camera_center,
            #                                    out_3d=True)
            # projected_vertices_2d = projected_vertices_3d[:,:,:-1]
            # projected_vertices_2d = projected_vertices_2d + 0.5 * self.options.img_res
            # projected_vertices_3d[:,:,:-1] = projected_vertices_3d[:,:,:-1] + 0.5 * self.options.img_res

            # # fig = plt.figure()
            # # ax = fig.add_subplot(131)
            # # plt.plot(projected_vertices_2d.cpu().detach().numpy()[0,:,0], projected_vertices_2d.cpu().detach().numpy()[0,:,1], '.')

            # # ax = fig.add_subplot(132)
            # # plt.imshow(ir_img.cpu().detach().numpy()[0][0])

            # # ax = fig.add_subplot(133)
            # # print(input_batch['scale'], input_batch['center']) # 3, (200, 400)
            # # plt.imshow(images.cpu().detach().numpy()[0][0])
            # # plt.plot(projected_vertices_2d.cpu().detach().numpy()[0,:,0], projected_vertices_2d.cpu().detach().numpy()[0,:,1], '.')

            # # mesh_matrix_batch, contact_matrix_batch = vert2map(projected_vertices_3d)
            # padding_x = 10 
            # padding_y = 100 
            # # masks_out = torch.ones([batch_size, 1, self.options.img_res, self.options.img_res])
            # masks = torch.zeros([batch_size, 1, self.options.img_res+padding_y*2, self.options.img_res+padding_x*2]).to(self.device)
            # for i, mask in enumerate(masks): 
            #     x = projected_vertices_3d[i,:,0].type(torch.LongTensor) + padding_x
            #     y = projected_vertices_3d[i,:,1].type(torch.LongTensor) + padding_y
            #     z = projected_vertices_3d[i,:,2]
            #     # print(min(x),max(x))
            #     # print(min(y),max(y))
            #     # print(min(z),max(z))
            #     masks[i, 0, y, x] = 1 # z
            # # x = projected_vertices_3d[:,:,0].type(torch.LongTensor) + padding_x
            # # y = projected_vertices_3d[:,:,1].type(torch.LongTensor) + padding_y
            # # z = 
            # # masks[:, 0, y, x] = projected_vertices_3d[:,:,2]
            # # masks[:,:,1:-1,1:-1] = masks[:,:,:-2,:-2] + masks[:,:,1:-1,:-2] + masks[:,:,2:,:-2] \
            # #                        + masks[:,:,:-2,1:-1] + masks[:,:,1:-1,1:-1] + masks[:,:,2:,1:-1] \
            # #                        + masks[:,:,:-2,2:] + masks[:,:,1:-1,2:] + masks[:,:,2:,2:] 
            # masks[:,:,2:-2,2:-2] = (masks[:,:,:-4,:-4] + masks[:,:,1:-3,:-4] + masks[:,:,2:-2,:-4] + masks[:,:,3:-1,:-4] + masks[:,:,4:,:-4] \
            #                     + masks[:,:,:-4,1:-3] + masks[:,:,1:-3,1:-3] + masks[:,:,2:-2,1:-3] + masks[:,:,3:-1,1:-3] + masks[:,:,4:,1:-3] \
            #                     + masks[:,:,:-4,2:-2] + masks[:,:,1:-3,2:-2] + masks[:,:,2:-2,2:-2] + masks[:,:,3:-1,2:-2] + masks[:,:,4:,2:-2] \
            #                     + masks[:,:,:-4,3:-1] + masks[:,:,1:-3,3:-1] + masks[:,:,2:-2,3:-1] + masks[:,:,3:-1,3:-1] + masks[:,:,4:,3:-1] \
            #                     + masks[:,:,:-4,4:]   + masks[:,:,1:-3,4:]   + masks[:,:,2:-2,4:]   + masks[:,:,3:-1,4:]   + masks[:,:,4:,4:])/25.0
            # # masks_2 = nn.functional.interpolate(masks, scale_factor=0.4, mode='area')
            # # masks_2 = nn.functional.interpolate(masks_2, size=(self.options.img_res,self.options.img_res), mode='area')
            # masks = masks[:,:,padding_y:self.options.img_res+padding_y,padding_x:self.options.img_res+padding_x]
            # masks[masks > 0] = 1

            # plt.figure()
            # masks_show = masks.clone()
            # plt.imshow(masks_show.cpu().detach().numpy()[0][0])

            # # fig = plt.figure()
            # # masks_2_show = masks_2.clone()
            # # plt.imshow(masks_2_show.cpu().detach().numpy()[0][0])

            # # # fig.add_subplot(133, projection='3d')
            # # # plt.plot(projected_vertices_2d.cpu().detach().numpy()[0,:,0], projected_vertices_2d.cpu().detach().numpy()[0,:,1], projected_vertices_3d.cpu().detach().numpy()[0,:,2], marker='.')

            # plt.show()

        elif self.model_name == "ir_depth_fusion":
            pred_rotmat_1, pred_betas_1, pred_camera_1, pred_rotmat, pred_betas, pred_camera, ir_out, depth_out, mask_l = self.model(
                [ir_img, depth_img], self.smpl)
            # pred_rotmat_1, pred_betas_1, pred_camera_1, pred_depth_1 = self.model([ ir_img, depth_img])
            # pred_rotmat, pred_betas, pred_camera, pred_depth = self.model(
            #     [ir_img, pred_depth_1])
            # for step 1
            pred_output_1 = self.smpl(betas=pred_betas_1, body_pose=pred_rotmat_1[:,1:], global_orient=pred_rotmat_1[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices_1 = pred_output_1.vertices
            pred_joints_1 = pred_output_1.joints

            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t_1 = torch.stack([pred_camera_1[:,1],
                                      pred_camera_1[:,2],
                                      2*self.focal_length/(self.options.img_res * pred_camera_1[:,0] +1e-9)],dim=-1)


            camera_center = torch.zeros(batch_size, 2, device=self.device)
            pred_keypoints_2d_1 = perspective_projection(pred_joints_1,
                                                       rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                       translation=pred_cam_t_1,
                                                       focal_length=self.focal_length,
                                                       camera_center=camera_center)
            # Normalize keypoints to [-1,1]
            pred_keypoints_2d_1 = pred_keypoints_2d_1 / (self.options.img_res / 2.)

        elif self.model_name == "ir_pm_fusion":
            pred_rotmat_1, pred_betas_1, pred_camera_1, pred_rotmat, pred_betas, pred_camera, ir_out, pm_out, mask_l = self.model(
                [ir_img, pm_img], self.smpl)
            # pred_rotmat_1, pred_betas_1, pred_camera_1, pred_depth_1 = self.model([ ir_img, depth_img])
            # pred_rotmat, pred_betas, pred_camera, pred_depth = self.model(
            #     [ir_img, pred_depth_1])
            # for step 1
            pred_output_1 = self.smpl(betas=pred_betas_1, body_pose=pred_rotmat_1[:,1:], global_orient=pred_rotmat_1[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices_1 = pred_output_1.vertices
            pred_joints_1 = pred_output_1.joints

            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t_1 = torch.stack([pred_camera_1[:,1],
                                      pred_camera_1[:,2],
                                      2*self.focal_length/(self.options.img_res * pred_camera_1[:,0] +1e-9)],dim=-1)


            camera_center = torch.zeros(batch_size, 2, device=self.device)
            pred_keypoints_2d_1 = perspective_projection(pred_joints_1,
                                                       rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                       translation=pred_cam_t_1,
                                                       focal_length=self.focal_length,
                                                       camera_center=camera_center)
            # Normalize keypoints to [-1,1]
            pred_keypoints_2d_1 = pred_keypoints_2d_1 / (self.options.img_res / 2.)
        elif self.model_name in ["rgb_depth_fusion", "rgb_pm_fusion"]:
            if self.model_name == "rgb_depth_fusion":
                input_2 = depth_img
            elif self.model_name == "rgb_pm_fusion":
                input_2 = pm_img
            pred_rotmat_1, pred_betas_1, pred_camera_1, pred_rotmat, pred_betas, pred_camera, PD_out, mask_l = self.model(
                [images, input_2], self.smpl)
            # pred_rotmat_1, pred_betas_1, pred_camera_1, pred_depth_1 = self.model([ ir_img, depth_img])
            # pred_rotmat, pred_betas, pred_camera, pred_depth = self.model(
            #     [ir_img, pred_depth_1])
            # for step 1
            pred_output_1 = self.smpl(betas=pred_betas_1, body_pose=pred_rotmat_1[:,1:], global_orient=pred_rotmat_1[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices_1 = pred_output_1.vertices
            pred_joints_1 = pred_output_1.joints

            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t_1 = torch.stack([pred_camera_1[:,1],
                                      pred_camera_1[:,2],
                                      2*self.focal_length/(self.options.img_res * pred_camera_1[:,0] +1e-9)],dim=-1)


            camera_center = torch.zeros(batch_size, 2, device=self.device)
            pred_keypoints_2d_1 = perspective_projection(pred_joints_1,
                                                       rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                       translation=pred_cam_t_1,
                                                       focal_length=self.focal_length,
                                                       camera_center=camera_center)
            # Normalize keypoints to [-1,1]
            pred_keypoints_2d_1 = pred_keypoints_2d_1 / (self.options.img_res / 2.)

        elif self.model_name == "ir_depth_pm_fusion":
            with torch.no_grad():
                _,_,_,pred_pose_ori, pred_betas_ori, pred_camera_ori, ir_out_ori, depth_out_ori,_ \
                = self.pretrained_ir_depth_model([ir_img, depth_img], self.smpl, return_pose=True)

            pred_rotmat_1, pred_betas_1, pred_camera_1, pred_rotmat, pred_betas, pred_camera, ir_out, depth_out, pm_out, mask_l = self.model(
                [ir_out_ori, depth_out_ori, pm_img, ir_img, depth_img], self.smpl, 
                init_pose=pred_pose_ori, init_shape=pred_betas_ori, init_cam=pred_camera_ori)

            # for step 1
            pred_output_1 = self.smpl(betas=pred_betas_1, body_pose=pred_rotmat_1[:,1:], global_orient=pred_rotmat_1[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices_1 = pred_output_1.vertices
            pred_joints_1 = pred_output_1.joints

            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t_1 = torch.stack([pred_camera_1[:,1],
                                      pred_camera_1[:,2],
                                      2*self.focal_length/(self.options.img_res * pred_camera_1[:,0] +1e-9)],dim=-1)


            camera_center = torch.zeros(batch_size, 2, device=self.device)
            pred_keypoints_2d_1 = perspective_projection(pred_joints_1,
                                                       rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                       translation=pred_cam_t_1,
                                                       focal_length=self.focal_length,
                                                       camera_center=camera_center)
            # Normalize keypoints to [-1,1]
            pred_keypoints_2d_1 = pred_keypoints_2d_1 / (self.options.img_res / 2.)

        elif self.model_name == "ir_depth_pm_rgb_fusion":
            with torch.no_grad():
                _,_,_,pred_pose_ori, pred_betas_ori, pred_camera_ori, ir_out_ori, depth_out_ori,_ = self.pretrained_ir_depth_model(
                    [ir_img, depth_img], self.smpl, return_pose=True)

                _, _, _, pred_rotmat, pred_betas, pred_camera, ir_out, depth_out, pm_out, mask_l = self.pretrained_ir_depth_pm_model(
                    [ir_out_ori, depth_out_ori, pm_img, ir_img, depth_img], self.smpl, 
                    init_pose=pred_pose_ori, init_shape=pred_betas_ori, init_cam=pred_camera_ori, return_pose=True)

            pred_rotmat_1, pred_betas_1, pred_camera_1, pred_rotmat, pred_betas, pred_camera, ir_out, depth_out, pm_out, mask_l = self.model(
                [ir_out_ori, depth_out_ori, pm_img, ir_img, depth_img], self.smpl, 
                init_pose=pred_pose_ori, init_shape=pred_betas_ori, init_cam=pred_camera_ori)

            # for step 1
            pred_output_1 = self.smpl(betas=pred_betas_1, body_pose=pred_rotmat_1[:,1:], global_orient=pred_rotmat_1[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices_1 = pred_output_1.vertices
            pred_joints_1 = pred_output_1.joints

            # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
            # This camera translation can be used in a full perspective projection
            pred_cam_t_1 = torch.stack([pred_camera_1[:,1],
                                      pred_camera_1[:,2],
                                      2*self.focal_length/(self.options.img_res * pred_camera_1[:,0] +1e-9)],dim=-1)


            camera_center = torch.zeros(batch_size, 2, device=self.device)
            pred_keypoints_2d_1 = perspective_projection(pred_joints_1,
                                                       rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                       translation=pred_cam_t_1,
                                                       focal_length=self.focal_length,
                                                       camera_center=camera_center)
            # Normalize keypoints to [-1,1]
            pred_keypoints_2d_1 = pred_keypoints_2d_1 / (self.options.img_res / 2.)

        elif self.model_name in ["cashmrV2"]:
            pred_rotmat_1, pred_betas_1, pred_camera_1, pred_depth_1 = [], [], [], []
            pred_vertices_1, pred_joints_1, pred_keypoints_2d_1 = [], [], []
            pred_rotmat_temp, pred_betas_temp, pred_camera_temp, pred_depth_temp = self.model([images, ir_img, depth_img, pm_img])
            pred_vertices_temp, pred_joints_temp, pred_keypoints_2d_temp = self.get_vertices(pred_rotmat_temp, pred_betas_temp, pred_camera_temp, batch_size)
            pred_rotmat_1.append(pred_rotmat_temp)
            pred_betas_1.append(pred_betas_temp)
            pred_camera_1.append(pred_camera_temp)
            pred_depth_1.append(pred_depth_temp)
            pred_vertices_1.append(pred_vertices_temp)
            pred_joints_1.append(pred_joints_temp)
            pred_keypoints_2d_1.append(pred_keypoints_2d_temp)
            for cas_iter in range(self.num_cas_iters-2):
                pred_pose_1 = pred_rotmat_1[-1][:,:,:,1:].contiguous().view(batch_size,144)
                pred_rotmat_temp, pred_betas_temp, pred_camera_temp, pred_depth_temp = self.model(
                    [images, ir_img, pred_depth_1[-1], pm_img])
                    # [images, ir_img, pred_depth_1[-1], pm_img], pred_pose_1, pred_betas_1[-1], pred_camera_1[-1])
                pred_vertices_temp, pred_joints_temp, pred_keypoints_2d_temp = self.get_vertices(pred_rotmat_temp, pred_betas_temp, pred_camera_temp, batch_size)
                pred_rotmat_1.append(pred_rotmat_temp)
                pred_betas_1.append(pred_betas_temp)
                pred_camera_1.append(pred_camera_temp)
                pred_depth_1.append(pred_depth_temp)
                pred_vertices_1.append(pred_vertices_temp)
                pred_joints_1.append(pred_joints_temp)
                pred_keypoints_2d_1.append(pred_keypoints_2d_temp)
            pred_pose_1 = pred_rotmat_1[-1][:,:,:,1:].contiguous().view(batch_size,144)
            pred_rotmat, pred_betas, pred_camera, pred_depth = self.model(
                [images, ir_img, pred_depth_1[-1], pm_img])
                # [images, ir_img, pred_depth_1[-1], pm_img], pred_pose_1, pred_betas_1[-1], pred_camera_1[-1])
        elif self.model_name in ["ir_depth_featatt_cashmrV2"]:
            pred_rotmat_1, pred_betas_1, pred_camera_1, pred_depth_1, pred_ir_1 = [], [], [], [], []
            pred_vertices_1, pred_joints_1, pred_keypoints_2d_1 = [], [], []
            pred_rotmat_temp, pred_betas_temp, pred_camera_temp, pred_depth_temp, pred_ir_temp = self.model([ir_img, depth_img])
            pred_vertices_temp, pred_joints_temp, pred_keypoints_2d_temp = self.get_vertices(pred_rotmat_temp, pred_betas_temp, pred_camera_temp, batch_size)
            pred_rotmat_1.append(pred_rotmat_temp)
            pred_betas_1.append(pred_betas_temp)
            pred_camera_1.append(pred_camera_temp)
            pred_depth_1.append(pred_depth_temp)
            pred_ir_1.append(pred_ir_temp)
            pred_vertices_1.append(pred_vertices_temp)
            pred_joints_1.append(pred_joints_temp)
            pred_keypoints_2d_1.append(pred_keypoints_2d_temp)
            for cas_iter in range(self.num_cas_iters-2):
                pred_pose_1 = pred_rotmat_1[-1][:,:,:,1:].contiguous().view(batch_size,144)
                pred_rotmat_temp, pred_betas_temp, pred_camera_temp, pred_depth_temp, pred_ir_temp = self.model(
                    [pred_ir_1[-1], pred_depth_1[-1]])
                    # [images, ir_img, pred_depth_1[-1], pm_img], pred_pose_1, pred_betas_1[-1], pred_camera_1[-1])
                pred_vertices_temp, pred_joints_temp, pred_keypoints_2d_temp = self.get_vertices(pred_rotmat_temp, pred_betas_temp, pred_camera_temp, batch_size)
                pred_rotmat_1.append(pred_rotmat_temp)
                pred_betas_1.append(pred_betas_temp)
                pred_camera_1.append(pred_camera_temp)
                pred_depth_1.append(pred_depth_temp)
                pred_ir_1.append(pred_ir_temp)
                pred_vertices_1.append(pred_vertices_temp)
                pred_joints_1.append(pred_joints_temp)
                pred_keypoints_2d_1.append(pred_keypoints_2d_temp)
            pred_pose_1 = pred_rotmat_1[-1][:,:,:,1:].contiguous().view(batch_size,144)
            pred_rotmat, pred_betas, pred_camera, pred_depth, pred_ir = self.model(
                [pred_ir_1[-1], pred_depth_1[-1]])
                # [images, ir_img, pred_depth_1[-1], pm_img], pred_pose_1, pred_betas_1[-1], pred_camera_1[-1])
        elif self.model_name == "mulhmr":
            # pred_rotmat, pred_betas, pred_camera = self.model([ir_img, depth_img])
            pred_rotmat, pred_betas, pred_camera = self.model([ir_img, depth_img, pm_img])
            # pred_rotmat, pred_betas, pred_camera = self.model([ir_img, depth_img, pm_img, images])
        elif self.model_name == "irhmr":
            pred_rotmat, pred_betas, pred_camera = self.model(ir_img)
        elif self.model_name == "depthhmr":
            pred_rotmat, pred_betas, pred_camera = self.model(depth_img)
        elif self.model_name == "pmhmr":
            pred_rotmat, pred_betas, pred_camera = self.model(pm_img)

        # ''' bodiesAtRest
        # '''
        elif self.model_name == "bodiesAtRest":
            pm_contact = input_batch['pm_contact'].to(self.device)
            if epoch < self.options.mod1_epoch:
                pred_rotmat, pred_betas, pred_camera, pred_pose = self.model(images=torch.cat([pm_img, pm_contact], 1), mode="0")
            else:
                pred_rotmat, pred_betas, pred_camera, pred_pose = self.model(images=torch.cat([pm_img, pm_contact], 1), mode="1")
        elif self.model_name == "bodiesAtRest4mod":
            pm_contact = input_batch['pm_contact'].to(self.device)
            pred_rotmat, pred_betas, pred_camera, pred_pose = self.model(images=torch.cat([images, ir_img, depth_img, pm_img, pm_contact], 1), mode="0")
            # if epoch < self.options.mod1_epoch:
            #     pred_rotmat, pred_betas, pred_camera, pred_pose = self.model(images=torch.cat([images, ir_img, depth_img, pm_img, pm_contact], 1), mode="0")
            # else:
            #     pred_rotmat, pred_betas, pred_camera, pred_pose = self.model(images=torch.cat([images, ir_img, depth_img, pm_img, pm_contact], 1), mode="1")

        # ''' hmr4mod
        # '''
        elif self.model_name == "hmr4mod":
            pred_rotmat, pred_betas, pred_camera = self.model(torch.cat([images, ir_img, depth_img, pm_img], 1))
        else:
            pred_rotmat, pred_betas, pred_camera = self.model(images)

        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)],dim=-1)


        camera_center = torch.zeros(batch_size, 2, device=self.device)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length,
                                                   camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

        if self.model_name in ["bodiesAtRest", "bodiesAtRest4mod"]:
            '''
            project vertices to plane
            '''
            projected_vertices_3d = perspective_projection(pred_vertices,
                                               rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                               translation=pred_cam_t,
                                               focal_length=self.focal_length,
                                               camera_center=camera_center,
                                               out_3d=True)
            projected_vertices_3d[:,:,:-1] = projected_vertices_3d[:,:,:-1] + 0.5 * self.options.img_res

            padding_x = 500 
            padding_y = 500
            masks = torch.zeros([batch_size, 1, self.options.img_res+padding_y*2, self.options.img_res+padding_x*2]).to(self.device)
            for i, mask in enumerate(masks): 
                x = projected_vertices_3d[i,:,0].type(torch.LongTensor) + padding_x
                y = projected_vertices_3d[i,:,1].type(torch.LongTensor) + padding_y
                z = projected_vertices_3d[i,:,2]
                masks[i, 0, y, x] = 1 # z
            masks[:,:,2:-2,2:-2] = (masks[:,:,:-4,:-4] + masks[:,:,1:-3,:-4] + masks[:,:,2:-2,:-4] + masks[:,:,3:-1,:-4] + masks[:,:,4:,:-4] \
                                + masks[:,:,:-4,1:-3] + masks[:,:,1:-3,1:-3] + masks[:,:,2:-2,1:-3] + masks[:,:,3:-1,1:-3] + masks[:,:,4:,1:-3] \
                                + masks[:,:,:-4,2:-2] + masks[:,:,1:-3,2:-2] + masks[:,:,2:-2,2:-2] + masks[:,:,3:-1,2:-2] + masks[:,:,4:,2:-2] \
                                + masks[:,:,:-4,3:-1] + masks[:,:,1:-3,3:-1] + masks[:,:,2:-2,3:-1] + masks[:,:,3:-1,3:-1] + masks[:,:,4:,3:-1] \
                                + masks[:,:,:-4,4:]   + masks[:,:,1:-3,4:]   + masks[:,:,2:-2,4:]   + masks[:,:,3:-1,4:]   + masks[:,:,4:,4:])
            masks = masks[:,:,padding_y:self.options.img_res+padding_y,padding_x:self.options.img_res+padding_x]
            masks[masks > 0] = 1

            # plt.figure()
            # masks_show = masks.clone()
            # plt.imshow(masks_show.cpu().detach().numpy()[0][0])
            # plt.show()
            if self.model_name in ["bodiesAtRest4mod"]:
                all_input = torch.cat([images, ir_img, depth_img, pm_img, pm_contact, masks], 1)
                pred_rotmat_1, pred_betas_1, pred_camera_1 = self.model(
                    images=all_input, mode="2", init_pose=pred_pose, init_shape=pred_betas, init_cam=pred_camera)
                # for step 1
                pred_output_1 = self.smpl(betas=pred_betas_1, body_pose=pred_rotmat_1[:,1:], global_orient=pred_rotmat_1[:,0].unsqueeze(1), pose2rot=False)
                pred_vertices_1 = pred_output_1.vertices
                pred_joints_1 = pred_output_1.joints

                # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
                # This camera translation can be used in a full perspective projection
                pred_cam_t_1 = torch.stack([pred_camera_1[:,1],
                                          pred_camera_1[:,2],
                                          2*self.focal_length/(self.options.img_res * pred_camera_1[:,0] +1e-9)],dim=-1)
                camera_center = torch.zeros(batch_size, 2, device=self.device)
                pred_keypoints_2d_1 = perspective_projection(pred_joints_1,
                                                           rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                           translation=pred_cam_t_1,
                                                           focal_length=self.focal_length,
                                                           camera_center=camera_center)
                # Normalize keypoints to [-1,1]
                pred_keypoints_2d_1 = pred_keypoints_2d_1 / (self.options.img_res / 2.)

            # if epoch >= self.options.mod1_epoch:
            #     if self.model_name == "bodiesAtRest":
            #         all_input = torch.cat([pm_img, pm_contact, masks], 1)
            #     else:
            #         all_input = torch.cat([images, ir_img, depth_img, pm_img, pm_contact, masks], 1)
                # pred_rotmat_1, pred_betas_1, pred_camera_1, _ = self.model(
                #     images=all_input, mode="2", init_pose=pred_pose, init_shape=pred_betas, init_cam=pred_camera)
                # pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
                # pred_vertices = pred_output.vertices
                # pred_joints = pred_output.joints

                # # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
                # # This camera translation can be used in a full perspective projection
                # pred_cam_t = torch.stack([pred_camera[:,1],
                #                           pred_camera[:,2],
                #                           2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)],dim=-1)


                # camera_center = torch.zeros(batch_size, 2, device=self.device)
                # pred_keypoints_2d = perspective_projection(pred_joints,
                #                                            rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                #                                            translation=pred_cam_t,
                #                                            focal_length=self.focal_length,
                #                                            camera_center=camera_center)
                # # Normalize keypoints to [-1,1]
                # pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

        if self.options.run_smplify:

            # Convert predicted rotation matrices to axis-angle
            pred_rotmat_hom = torch.cat([pred_rotmat.detach().view(-1, 3, 3).detach(), torch.tensor([0,0,1], dtype=torch.float32,
                device=self.device).view(1, 3, 1).expand(batch_size * 24, -1, -1)], dim=-1)
            pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch_size, -1)
            # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
            pred_pose[torch.isnan(pred_pose)] = 0.0

            # Run SMPLify optimization starting from the network prediction
            new_opt_vertices, new_opt_joints,\
            new_opt_pose, new_opt_betas,\
            new_opt_cam_t, new_opt_joint_loss = self.smplify(
                                        pred_pose.detach(), pred_betas.detach(),
                                        pred_cam_t.detach(),
                                        0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device),
                                        gt_keypoints_2d_orig)
            new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)

            # Will update the dictionary for the examples where the new loss is less than the current one
            update = (new_opt_joint_loss < opt_joint_loss)
            

            opt_joint_loss[update] = new_opt_joint_loss[update]
            opt_vertices[update, :] = new_opt_vertices[update, :]
            opt_joints[update, :] = new_opt_joints[update, :]
            opt_pose[update, :] = new_opt_pose[update, :]
            opt_betas[update, :] = new_opt_betas[update, :]
            opt_cam_t[update, :] = new_opt_cam_t[update, :]


            self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu(), update.cpu())] = (opt_pose.cpu(), opt_betas.cpu())

        else:
            update = torch.zeros(batch_size, device=self.device).byte()

        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.

        # Replace the optimized parameters with the ground truth parameters, if available
        opt_vertices[has_smpl, :, :] = gt_vertices[has_smpl, :, :]
        opt_cam_t[has_smpl, :] = gt_cam_t[has_smpl, :]
        opt_joints[has_smpl, :, :] = gt_model_joints[has_smpl, :, :]
        opt_pose[has_smpl, :] = gt_pose[has_smpl, :]
        opt_betas[has_smpl, :] = gt_betas[has_smpl, :]

        # Assert whether a fit is valid by comparing the joint loss with the threshold
        valid_fit = (opt_joint_loss < self.options.smplify_threshold).to(self.device)
        # Add the examples with GT parameters to the list of valid fits
        valid_fit = valid_fit | has_smpl

        opt_keypoints_2d = perspective_projection(opt_joints,
                                                  rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                  translation=opt_cam_t,
                                                  focal_length=self.focal_length,
                                                  camera_center=camera_center)


        opt_keypoints_2d = opt_keypoints_2d / (self.options.img_res / 2.)


        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                            self.options.gt_train_weight)

        # Compute 3D keypoint loss
        loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d)

        # Per-vertex loss for the shape
        loss_shape = self.shape_loss(pred_vertices, opt_vertices, valid_fit)

        # Compute depth recoverying loss
        if self.model_name == "rechmr":
            loss_extra = self.depth_loss(pred_depth, input_batch['depth_img_uncover'].to(self.device))
        elif self.model_name == "rec3hmr":
            loss_extra = self.depth_loss(pred_depth, input_batch['depth_img_uncover'].to(self.device)) +\
                        self.reconstraction_loss(pred_ir, input_batch['ir_img_uncover'].to(self.device)) +\
                        self.reconstraction_loss(pred_pm, input_batch['pm_img_uncover'].to(self.device))
        elif self.model_name in ["cashmr", "featcat_cashmr", "featatt_cashmr"]:
            # Compute loss on SMPL parameters
            loss_regr_pose_1, loss_regr_betas_1 = self.smpl_losses(pred_rotmat_1, pred_betas_1, opt_pose, opt_betas, valid_fit)

            # Compute 2D reprojection loss for the keypoints
            loss_keypoints_1 = self.keypoint_loss(pred_keypoints_2d_1, gt_keypoints_2d,
                                                self.options.openpose_train_weight,
                                                self.options.gt_train_weight)

            # Compute 3D keypoint loss
            loss_keypoints_3d_1 = self.keypoint_3d_loss(pred_joints_1, gt_joints, has_pose_3d)

            # Per-vertex loss for the shape
            loss_shape_1 = self.shape_loss(pred_vertices_1, opt_vertices, valid_fit)
            
            # all
            mask_gt = input_batch['mask_uncover'].to(self.device)
            loss_extra = self.depth_loss(pred_depth, input_batch['depth_img_uncover'].to(self.device), mask=mask_gt) +\
                           self.depth_loss(pred_depth_1, input_batch['depth_img_uncover'].to(self.device), mask=mask_gt) +\
                           self.options.shape_loss_weight * loss_shape_1 +\
                           self.options.keypoint_loss_weight * loss_keypoints_1 +\
                           self.options.keypoint_loss_weight * loss_keypoints_3d_1 +\
                           loss_regr_pose_1 + self.options.beta_loss_weight * loss_regr_betas_1 +\
                           ((torch.exp(-pred_camera_1[:,0]*10)) ** 2 ).mean()
            # mask_loss = self.depth_loss(masks, input_batch['pm_img_cover'].to(self.device))
            # loss_extra = loss_extra + mask_loss
            # print("loss", mask_loss)

        elif self.model_name in ["ir_depth_fusion", "ir_depth_pm_fusion"]:
            # Compute loss on SMPL parameters
            loss_regr_pose_1, loss_regr_betas_1 = self.smpl_losses(pred_rotmat_1, pred_betas_1, opt_pose, opt_betas, valid_fit)

            # Compute 2D reprojection loss for the keypoints
            loss_keypoints_1 = self.keypoint_loss(pred_keypoints_2d_1, gt_keypoints_2d,
                                                self.options.openpose_train_weight,
                                                self.options.gt_train_weight)

            # Compute 3D keypoint loss
            loss_keypoints_3d_1 = self.keypoint_3d_loss(pred_joints_1, gt_joints, has_pose_3d)

            # Per-vertex loss for the shape
            loss_shape_1 = self.shape_loss(pred_vertices_1, opt_vertices, valid_fit)
            
            # all
            mask_gt = input_batch['mask_uncover'].to(self.device)
            # loss_extra = 1.0*self.depth_loss(pred_depth, input_batch['depth_img_uncover'].to(self.device), mask=mask_gt) +\
            #             1.0*self.depth_loss(pred_depth_1, input_batch['depth_img_uncover'].to(self.device), mask=mask_gt) +\
            #                self.options.shape_loss_weight * loss_shape_1 +\
            #                self.options.keypoint_loss_weight * loss_keypoints_1 +\
            #                self.options.keypoint_loss_weight * loss_keypoints_3d_1 +\
            #                loss_regr_pose_1 + self.options.beta_loss_weight * loss_regr_betas_1 +\
            #                ((torch.exp(-pred_camera_1[:,0]*10)) ** 2 ).mean()

            loss_extra = 0.01*self.reconstraction_loss(mask_l, mask_gt) +\
                        1.0*self.depth_loss(depth_out, input_batch['depth_img_uncover'].to(self.device), mask=mask_gt) +\
                        1.0*self.depth_loss(ir_out, input_batch['ir_img_uncover'].to(self.device), mask=mask_gt) +\
                           self.options.shape_loss_weight * loss_shape_1 +\
                           self.options.keypoint_loss_weight * loss_keypoints_1 +\
                           self.options.keypoint_loss_weight * loss_keypoints_3d_1 +\
                           loss_regr_pose_1 + self.options.beta_loss_weight * loss_regr_betas_1 +\
                           ((torch.exp(-pred_camera_1[:,0]*10)) ** 2 ).mean()
            if self.model_name in [ "ir_depth_pm_fusion"]:
                loss_extra = loss_extra + 1.0*self.depth_loss(pm_out, input_batch['pm_img_uncover'].to(self.device), mask=mask_gt)

        elif self.model_name in ["ir_pm_fusion"]:
            # Compute loss on SMPL parameters
            loss_regr_pose_1, loss_regr_betas_1 = self.smpl_losses(pred_rotmat_1, pred_betas_1, opt_pose, opt_betas, valid_fit)

            # Compute 2D reprojection loss for the keypoints
            loss_keypoints_1 = self.keypoint_loss(pred_keypoints_2d_1, gt_keypoints_2d,
                                                self.options.openpose_train_weight,
                                                self.options.gt_train_weight)

            # Compute 3D keypoint loss
            loss_keypoints_3d_1 = self.keypoint_3d_loss(pred_joints_1, gt_joints, has_pose_3d)

            # Per-vertex loss for the shape
            loss_shape_1 = self.shape_loss(pred_vertices_1, opt_vertices, valid_fit)
            
            # all
            mask_gt = input_batch['mask_uncover'].to(self.device)
            # loss_extra = 1.0*self.depth_loss(pred_depth, input_batch['depth_img_uncover'].to(self.device), mask=mask_gt) +\
            #             1.0*self.depth_loss(pred_depth_1, input_batch['depth_img_uncover'].to(self.device), mask=mask_gt) +\
            #                self.options.shape_loss_weight * loss_shape_1 +\
            #                self.options.keypoint_loss_weight * loss_keypoints_1 +\
            #                self.options.keypoint_loss_weight * loss_keypoints_3d_1 +\
            #                loss_regr_pose_1 + self.options.beta_loss_weight * loss_regr_betas_1 +\
            #                ((torch.exp(-pred_camera_1[:,0]*10)) ** 2 ).mean()

            loss_extra = 0.01*self.reconstraction_loss(mask_l, mask_gt) +\
                        1.0*self.depth_loss(pm_out, input_batch['pm_img_uncover'].to(self.device), mask=mask_gt) +\
                        1.0*self.depth_loss(ir_out, input_batch['ir_img_uncover'].to(self.device), mask=mask_gt) +\
                           self.options.shape_loss_weight * loss_shape_1 +\
                           self.options.keypoint_loss_weight * loss_keypoints_1 +\
                           self.options.keypoint_loss_weight * loss_keypoints_3d_1 +\
                           loss_regr_pose_1 + self.options.beta_loss_weight * loss_regr_betas_1 +\
                           ((torch.exp(-pred_camera_1[:,0]*10)) ** 2 ).mean()

        elif self.model_name in ["rgb_depth_fusion", "rgb_pm_fusion"]:
            if self.model_name == "rgb_depth_fusion":
                img_uncover = input_batch['depth_img_uncover']
            elif self.model_name == "rgb_pm_fusion":
                img_uncover = input_batch['pm_img_uncover']

            # Compute loss on SMPL parameters
            loss_regr_pose_1, loss_regr_betas_1 = self.smpl_losses(pred_rotmat_1, pred_betas_1, opt_pose, opt_betas, valid_fit)

            # Compute 2D reprojection loss for the keypoints
            loss_keypoints_1 = self.keypoint_loss(pred_keypoints_2d_1, gt_keypoints_2d,
                                                self.options.openpose_train_weight,
                                                self.options.gt_train_weight)

            # Compute 3D keypoint loss
            loss_keypoints_3d_1 = self.keypoint_3d_loss(pred_joints_1, gt_joints, has_pose_3d)

            # Per-vertex loss for the shape
            loss_shape_1 = self.shape_loss(pred_vertices_1, opt_vertices, valid_fit)
            
            # all
            mask_gt = input_batch['mask_uncover'].to(self.device)
            # loss_extra = 1.0*self.depth_loss(pred_depth, input_batch['depth_img_uncover'].to(self.device), mask=mask_gt) +\
            #             1.0*self.depth_loss(pred_depth_1, input_batch['depth_img_uncover'].to(self.device), mask=mask_gt) +\
            #                self.options.shape_loss_weight * loss_shape_1 +\
            #                self.options.keypoint_loss_weight * loss_keypoints_1 +\
            #                self.options.keypoint_loss_weight * loss_keypoints_3d_1 +\
            #                loss_regr_pose_1 + self.options.beta_loss_weight * loss_regr_betas_1 +\
            #                ((torch.exp(-pred_camera_1[:,0]*10)) ** 2 ).mean()

            loss_extra = 0.01*self.reconstraction_loss(mask_l, mask_gt) +\
                        1.0*self.depth_loss(PD_out, img_uncover.to(self.device), mask=mask_gt) +\
                           self.options.shape_loss_weight * loss_shape_1 +\
                           self.options.keypoint_loss_weight * loss_keypoints_1 +\
                           self.options.keypoint_loss_weight * loss_keypoints_3d_1 +\
                           loss_regr_pose_1 + self.options.beta_loss_weight * loss_regr_betas_1 +\
                           ((torch.exp(-pred_camera_1[:,0]*10)) ** 2 ).mean()


        elif self.model_name in ["cashmrV2"]:
            loss_extra = self.depth_loss(pred_depth, input_batch['depth_img_uncover'].to(self.device))
            for i in range(len(pred_rotmat_1)):
                # Compute loss on SMPL parameters
                loss_regr_pose_1, loss_regr_betas_1 = self.smpl_losses(pred_rotmat_1[i], pred_betas_1[i], opt_pose, opt_betas, valid_fit)

                # Compute 2D reprojection loss for the keypoints
                loss_keypoints_1 = self.keypoint_loss(pred_keypoints_2d_1[i], gt_keypoints_2d,
                                                    self.options.openpose_train_weight,
                                                    self.options.gt_train_weight)

                # Compute 3D keypoint loss
                loss_keypoints_3d_1 = self.keypoint_3d_loss(pred_joints_1[i], gt_joints, has_pose_3d)

                # Per-vertex loss for the shape
                loss_shape_1 = self.shape_loss(pred_vertices_1[i], opt_vertices, valid_fit)
                
                # all
                loss_extra = loss_extra + self.depth_loss(pred_depth_1[i], input_batch['depth_img_uncover'].to(self.device)) +\
                               self.options.shape_loss_weight * loss_shape_1 +\
                               self.options.keypoint_loss_weight * loss_keypoints_1 +\
                               self.options.keypoint_loss_weight * loss_keypoints_3d_1 +\
                               loss_regr_pose_1 + self.options.beta_loss_weight * loss_regr_betas_1 +\
                               ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()
        elif self.model_name in ["ir_depth_featatt_cashmrV2"]:
            loss_extra = self.depth_loss(pred_depth, input_batch['depth_img_uncover'].to(self.device)) +\
                         self.depth_loss(pred_ir, input_batch['ir_img_uncover'].to(self.device))
            for i in range(len(pred_rotmat_1)):
                # Compute loss on SMPL parameters
                loss_regr_pose_1, loss_regr_betas_1 = self.smpl_losses(pred_rotmat_1[i], pred_betas_1[i], opt_pose, opt_betas, valid_fit)

                # Compute 2D reprojection loss for the keypoints
                loss_keypoints_1 = self.keypoint_loss(pred_keypoints_2d_1[i], gt_keypoints_2d,
                                                    self.options.openpose_train_weight,
                                                    self.options.gt_train_weight)

                # Compute 3D keypoint loss
                loss_keypoints_3d_1 = self.keypoint_3d_loss(pred_joints_1[i], gt_joints, has_pose_3d)

                # Per-vertex loss for the shape
                loss_shape_1 = self.shape_loss(pred_vertices_1[i], opt_vertices, valid_fit)
                
                # all
                loss_extra = loss_extra + self.depth_loss(pred_depth_1[i], input_batch['depth_img_uncover'].to(self.device)) +\
                               self.depth_loss(pred_ir_1[i], input_batch['ir_img_uncover'].to(self.device)) +\
                               self.options.shape_loss_weight * loss_shape_1 +\
                               self.options.keypoint_loss_weight * loss_keypoints_1 +\
                               self.options.keypoint_loss_weight * loss_keypoints_3d_1 +\
                               loss_regr_pose_1 + self.options.beta_loss_weight * loss_regr_betas_1 +\
                               ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()
        elif self.model_name in ["bodiesAtRest"]:
            if epoch < self.options.mod1_epoch:
                loss_extra = 0.1*self.reconstraction_loss(masks, input_batch['mask_uncover'].to(self.device))
                # loss_extra = self.reconstraction_loss(masks, input_batch['pm_contact'][:,0,:,:].unsqueeze(1))
                
                # plt.figure()
                # masks_show = masks.clone()
                # plt.imshow(masks_show.cpu().detach().numpy()[0][0])
                # plt.show()
            else:
                loss_extra = 0
        elif self.model_name in ["bodiesAtRest4mod"]:
            # Compute loss on SMPL parameters
            loss_regr_pose_1, loss_regr_betas_1 = self.smpl_losses(pred_rotmat_1, pred_betas_1, opt_pose, opt_betas, valid_fit)
            # Compute 2D reprojection loss for the keypoints
            loss_keypoints_1 = self.keypoint_loss(pred_keypoints_2d_1, gt_keypoints_2d,
                                                self.options.openpose_train_weight,
                                                self.options.gt_train_weight)
            # Compute 3D keypoint loss
            loss_keypoints_3d_1 = self.keypoint_3d_loss(pred_joints_1, gt_joints, has_pose_3d)
            # Per-vertex loss for the shape
            loss_shape_1 = self.shape_loss(pred_vertices_1, opt_vertices, valid_fit)
            loss_extra = 0.1*self.reconstraction_loss(masks, input_batch['mask_uncover'].to(self.device)) +\
                           self.options.shape_loss_weight * loss_shape_1 +\
                           self.options.keypoint_loss_weight * loss_keypoints_1 +\
                           self.options.keypoint_loss_weight * loss_keypoints_3d_1 +\
                           loss_regr_pose_1 + self.options.beta_loss_weight * loss_regr_betas_1 +\
                           ((torch.exp(-pred_camera_1[:,0]*10)) ** 2 ).mean()
        else:
            loss_extra = 0


        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = self.options.shape_loss_weight * loss_shape +\
               self.options.keypoint_loss_weight * loss_keypoints +\
               self.options.keypoint_loss_weight * loss_keypoints_3d +\
               loss_regr_pose + self.options.beta_loss_weight * loss_regr_betas +\
               ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean() +\
               1.0 * loss_extra
        loss *= 60


        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': pred_vertices.detach(),
                  'opt_vertices': opt_vertices,
                  'pred_cam_t': pred_cam_t.detach(),
                  'opt_cam_t': opt_cam_t}
        losses = {'loss': loss.detach().item(),
                  'loss_keypoints': loss_keypoints.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                  'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regr_betas': loss_regr_betas.detach().item(),
                  'loss_shape': loss_shape.detach().item()}
                  
        if self.model_name == "rechmr":
            losses['loss_extra'] = loss_extra.detach().item()

        return output, losses

    def train_summaries(self, input_batch, output, losses):
        images = input_batch['img']
        images = images * torch.tensor(constants.IMG_NORM_STD, device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor(constants.IMG_NORM_MEAN, device=images.device).reshape(1,3,1,1)

        ir_imgs = input_batch['ir_img']
        ir_imgs = ir_imgs * torch.tensor(constants.IR_NORM_STD, device=ir_imgs.device).reshape(1,1,1,1)
        ir_imgs = ir_imgs + torch.tensor(constants.IR_NORM_MEAN, device=ir_imgs.device).reshape(1,1,1,1)
        # depth_imgs = input_batch['depth_img']
        # pm_imgs = input_batch['pm_img']

        pred_vertices = output['pred_vertices']
        opt_vertices = output['opt_vertices']
        pred_cam_t = output['pred_cam_t']
        opt_cam_t = output['opt_cam_t']
        # if self.options.no_render == False:
        #     images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images, ir_imgs)
        #     images_opt = self.renderer.visualize_tb(opt_vertices, opt_cam_t, images, ir_imgs)
        #     self.summary_writer.add_image('pred_shape', images_pred, self.step_count)
        #     self.summary_writer.add_image('opt_shape', images_opt, self.step_count)
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
