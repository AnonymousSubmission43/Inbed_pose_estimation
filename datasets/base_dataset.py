from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
import matplotlib.pyplot as plt
from os.path import join

import config
import constants
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa

from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=True, is_train=True):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = config.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.normalize_depth = Normalize(mean=constants.DEPTH_NORM_MEAN, std=constants.DEPTH_NORM_STD)
        self.normalize_ir = Normalize(mean=constants.IR_NORM_MEAN, std=constants.IR_NORM_STD)
        self.normalize_pm = Normalize(mean=constants.PM_NORM_MEAN, std=constants.PM_NORM_STD)
        self.data = np.load(config.DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']

        # IR
        try:
            self.irimgname = self.data['irimgname']
            self.hasIR = True
        except KeyError:
            print("No ir image found in {}, replace with RGB images".format(dataset))
            self.hasIR = False
            self.irimgname = self.data['imgname']
            # self.irimgname = None

        # depth
        try:
            self.depthname = self.data['depthname']
            self.hasDEPTH = True
        except KeyError:
            print("No depth image found in {}, replace with RGB images".format(dataset))
            self.hasDEPTH = False
            self.depthname = self.data['imgname']
            # self.depthname = None

        # PM
        try:
            self.pmname = self.data['pmname']
            self.hasPM = True
        except KeyError:
            print("No pm image found in {}, replace with RGB images".format(dataset))
            self.hasPM = False
            self.pmname = self.data['imgname']
            # self.pmname = None
        
        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        
        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))
        
        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0
        
        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array(gender).astype(np.int32)
            # self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)
        
        self.length = self.scale.shape[0]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                    max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def gray_processing(self, gray_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        gray_img = crop(gray_img, center, scale, 
                      [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # flip the image 
        if flip:
            gray_img = flip_img(gray_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        gray_img[:,:] = np.minimum(255.0, np.maximum(0.0, gray_img[:,:]*pn[0]))
        # (1,224,224),float,[0,1]
        gray_img = np.expand_dims(gray_img.astype('float32'), axis=0)/255.0
        return gray_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        # S[:,2] = S[:,2] * (-1.0)
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def gen_contact(self, pm_img, mask, sigma=1, edges=True):
        pm_contact = np.copy(pm_img) #get the pmat contact map
        # print("1",pm_contact.shape, np.min(pm_contact), np.max(pm_contact))
        pm_contact[pm_contact > 0] = 1
        pm_contact[mask == 0] = 0
        pm_contact = gaussian_filter(pm_contact, sigma=sigma)

        if edges== False:
            return pm_contact
        else:
            ## generate pm edges 
            # this makes a sobel edge on the image
            sx = ndimage.sobel(pm_contact, axis=0, mode='constant')
            sy = ndimage.sobel(pm_contact, axis=1, mode='constant')
            p_map_inter = np.hypot(sx, sy)
            # p_map_inter = np.clip(p_map_inter, a_min=0, a_max = 1)
            p_map_inter = p_map_inter / np.max(p_map_inter)

            return np.concatenate((pm_contact, p_map_inter), axis=0)


    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()
        
        # Load image
        imgname = join(self.img_dir, self.imgname[index])
        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(imgname)
        orig_shape = np.array(img.shape)[:2]

        # Load ir image
        irimgname = join(self.img_dir, self.irimgname[index])
        if self.hasIR:
            try:
                ir_img = cv2.imread(irimgname, 0).copy().astype(np.float32)
            except TypeError:
                print(irimgname)
        else:
            try:
                ir_img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
            except TypeError:
                print(imgname) 

        # Load depth image
        depthname = join(self.img_dir, self.depthname[index])
        if self.hasDEPTH:
            try:
                depth_img = cv2.imread(depthname, 0).copy().astype(np.float32)           
            except TypeError:
                print(depthname)
        else:
            try:
                depth_img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
            except TypeError:
                print(imgname)

        # Load PM image
        pmname = join(self.img_dir, self.pmname[index])
        if self.hasPM:
            try:
                pm_img = cv2.imread(pmname, 0).copy().astype(np.float32)
            except TypeError:
                print(pmname)
        else:
            try:
                pm_img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
            except TypeError:
                print(imgname)

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        ir_img = self.gray_processing(ir_img, center, sc*scale, rot, flip, pn)
        depth_img = self.gray_processing(depth_img, center, sc*scale, rot, flip, pn)
        pm_img = self.gray_processing(pm_img, center, sc*scale, rot, flip, pn)

        # img2 = img.copy()
        # ir_img2 = ir_img.copy()
        # depth_img2 = depth_img.copy()
        # pm_img2 = pm_img.copy()
        # cv2.imshow('rgb1', np.transpose(img2, (1,2,0)))
        # cv2.imshow("ir1",  np.transpose(ir_img2, (1,2,0)))   
        # cv2.imshow("depth1",  np.transpose(depth_img2, (1,2,0)))   
        # cv2.imshow("pm1",  np.transpose(pm_img2, (1,2,0)))
        # cv2.waitKey(0)

        ## uncovered
        imguncovername = imgname.replace("cover1", "uncover").replace("cover2", "uncover")
        img_uncover = cv2.imread(imguncovername)[:,:,::-1].copy().astype(np.float32)
        img_uncover = self.rgb_processing(img_uncover, center, sc*scale, rot, flip, pn)

        irimguncovername = irimgname.replace("cover1", "uncover").replace("cover2", "uncover")
        ir_img_uncover = cv2.imread(irimguncovername, 0).copy().astype(np.float32)
        ir_img_uncover = self.gray_processing(ir_img_uncover, center, sc*scale, rot, flip, pn)

        depthuncovername = depthname.replace("cover1", "uncover").replace("cover2", "uncover")
        depth_img_uncover = cv2.imread(depthuncovername, 0).copy().astype(np.float32)
        depth_img_uncover = self.gray_processing(depth_img_uncover, center, sc*scale, rot, flip, pn)
        # depth_img = 1.0 - depth_img
        # depth_img_uncover = 1.0 - depth_img_uncover
        # threshold = 0.805
        # depth_img[(depth_img > threshold) | (depth_img == 0) | (depth_img_uncover == 0)] = 0
        # depth_img_uncover[(depth_img_uncover > threshold) | (depth_img == 0) | (depth_img_uncover == 0)] = 0
        # item['depth_mask'] = torch.Tensor((depth_img < threshold) & (depth_img != 0) & (depth_img_uncover != 0))

        pmuncovername = pmname.replace("cover1", "uncover").replace("cover2", "uncover")
        pm_img_uncover = cv2.imread(pmuncovername, 0).copy().astype(np.float32)
        pm_img_uncover = self.gray_processing(pm_img_uncover, center, sc*scale, rot, flip, pn)

        ## masks and contact images
        maskuncovername = pmuncovername.replace("PM_aligned", "masks")
        mask_uncover = cv2.imread(maskuncovername, 0).copy().astype(np.float32)
        mask_uncover = self.gray_processing(mask_uncover, center, sc*scale, rot, flip, pn)

        pm_contact = self.gen_contact(pm_img, mask_uncover, sigma=1, edges=True)
        # pm_contact = self.gen_contact(depth_img_uncover, sigma=1, edges=True)

        # # img2 = img.copy()
        # # ir_img2 = ir_img.copy()
        # depth_img2 = depth_img.copy()
        # # pm_img2 = pm_img.copy()
        # # ir_img2 = ir_img.copy()
        # # print(img.shape, pm_contact.shape, mask_uncover.shape)
        # plt.figure()
        # plt.imshow( depth_img2[0])
        # # plt.imshow( pm_contact[0])
        # # # plt.figure()
        # # # # plt.imshow(pm_contact[1])
        # # plt.figure()
        # # plt.imshow( mask_uncover[0])
        # plt.show()

        img = torch.from_numpy(img).float()
        ir_img = torch.from_numpy(ir_img).float()
        depth_img = torch.from_numpy(depth_img).float()
        pm_img = torch.from_numpy(pm_img).float()


        # if self.irimgname
        # Store image before normalization to use it in visualization
        item['img'] = self.normalize_img(img)
        item['ir_img'] = self.normalize_ir(ir_img)
        item['depth_img'] = self.normalize_depth(depth_img)
        item['pm_img'] = self.normalize_pm(pm_img)

        item['img_uncover'] = torch.from_numpy(img_uncover).float() ## used for saving ref
        item['ir_img_uncover'] = self.normalize_ir(torch.from_numpy(ir_img_uncover).float())
        item['depth_img_uncover'] = self.normalize_depth(torch.from_numpy(depth_img_uncover).float())
        item['pm_img_uncover'] = self.normalize_pm(torch.from_numpy(pm_img_uncover).float())

        item['mask_uncover'] = torch.from_numpy(mask_uncover).float()
        item['pm_contact'] = torch.from_numpy(pm_contact).float()


        # img = np.array(item['img'])
        # ir_img = np.array(item['ir_img'])
        # depth_img = np.array(item['depth_img'])+2
        # pm_img = np.array(item['pm_img'])
        # print("-"*50)
        # print(img.shape, np.min(img), np.max(img))
        # print(ir_img.shape, np.min(ir_img), np.max(ir_img))
        # print(depth_img.shape, np.min(depth_img), np.max(depth_img))
        # print(pm_img.shape, np.min(pm_img), np.max(pm_img))
        # cv2.imshow('rgb', np.transpose(img, (1,2,0)))
        # cv2.imshow("ir",  np.transpose(ir_img, (1,2,0)))   
        # cv2.imshow("depth",  np.transpose(depth_img, (1,2,0)))   
        # cv2.imshow("pm",  np.transpose(pm_img, (1,2,0)))   
        # cv2.waitKey(0)  


        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
            # if int(imgname.split("_")[-1].split(".")[0]) > 15:
            #     item['pose_3d'][:,2] =  item['pose_3d'][:,2] * 5
        else:
            item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot, flip)).float()

        item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = self.has_pose_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        return item

    def __len__(self):
        return len(self.imgname)
