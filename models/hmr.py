import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import numpy as np
import math
from utils.geometry import rot6d_to_rotmat, perspective_projection
# from models import SMPL

class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size,
        bias=False, bn=True, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale


    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=True, act=False, bias=False):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, kernel_size=3, padding=1, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class Reconstruct(nn.Module):
    def __init__(self, num_feat=1, out_dim=1):
        super(Reconstruct, self).__init__()
        self.decDepth1 = nn.Sequential(*[
                                nn.Conv2d(2048*num_feat, 1024, kernel_size=1, bias=False),
                                ResBlock(1024, kernel_size=3),
                                Upsampler(2, 1024)
                                ])
        self.decDepth2 = nn.Sequential(*[
                                nn.Conv2d(1024*(num_feat+1), 512, kernel_size=1, bias=False),
                                ResBlock(512, kernel_size=3),
                                Upsampler(2, 512)
                                ])
        self.decDepth3 = nn.Sequential(*[
                                nn.Conv2d(512*(num_feat+1), 256, kernel_size=1, bias=False),
                                ResBlock(256, kernel_size=3),
                                Upsampler(2, 256)
                                ])
        self.decDepth4 = nn.Sequential(*[
                                nn.Conv2d(256*(num_feat+1), 128, kernel_size=1, bias=False),
                                ResBlock(128, kernel_size=3),
                                Upsampler(2, 128)
                                ])
        self.decDepth = nn.Sequential(*[
                                nn.Conv2d(128+64*num_feat, 128, kernel_size=1, bias=False),
                                ResBlock(128, kernel_size=3),
                                ResBlock(128, kernel_size=3),
                                Upsampler(2, 128),
                                nn.Conv2d(128, out_dim, kernel_size=3, padding=1, bias=False)
                                ])

    def forward(self, x0, x1, x2, x3, x4):
        xf = self.decDepth1(x4) # torch.Size([batch_size, 1024, 14, 14])
        xf = torch.cat((x3, xf), 1)
        xf = self.decDepth2(xf) # torch.Size([batch_size, 512, 28, 28])
        xf = torch.cat((x2, xf), 1) 
        xf = self.decDepth3(xf) # torch.Size([batch_size, 256, 56, 56])
        xf = torch.cat((x1, xf), 1) 
        xf = self.decDepth4(xf) # torch.Size([batch_size, 128, 112, 112])
        xf = torch.cat((x0, xf), 1) 
        depth = self.decDepth(xf) # torch.Size([batch_size, 3, 112, 112])

        return depth


class HMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params, input_dim=3):
        print("Model: HMR.")
        self.inplanes = 64
        super(HMR, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, return_pose=False):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x0 = self.conv1(x)
        x = self.bn1(x0)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        if not return_pose:
            return pred_rotmat, pred_shape, pred_cam
        else:
            return pred_rotmat, pred_shape, pred_cam, pred_pose, [x0, x1, x2, x3, x4]


class SingleHMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params):
        print("Model: SingleHMR.")
        self.inplanes = 64
        super(SingleHMR, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam


class MULHMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params):
        print("Model: MULHMR.")
        self.inplanes = 64
        super(MULHMR, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, fused_x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        # x = fused_x[0] # RGB
        # x_ir = fused_x[1] # IR
        # x = torch.cat([fused_x[0], fused_x[1]], 1)
        x = torch.cat([fused_x[0], fused_x[1], fused_x[2]], 1)
        # x = torch.cat([fused_x[0], fused_x[1], fused_x[2], fused_x[3]], 1) # torch.Size([64, 6, 224, 224]) 
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x) # torch.Size([64, 64, 112, 112])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # torch.Size([64, 64, 56, 56])

        x1 = self.layer1(x) # torch.Size([64, 256, 56, 56])
        x2 = self.layer2(x1) # torch.Size([64, 512, 28, 28])
        x3 = self.layer3(x2) # torch.Size([64, 1024, 14, 14])
        x4 = self.layer4(x3) # torch.Size([64, 2048, 7, 7])

        xf = self.avgpool(x4) # torch.Size([64, 2048, 1, 1])
        xf = xf.view(xf.size(0), -1) # torch.Size([64, 2048])

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam


class RECHMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params):
        print("Model: RECHMR.")
        self.inplanes = 64
        super(RECHMR, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        self.decDepth1 = nn.Sequential(*[
                                nn.Conv2d(2048, 1024, kernel_size=1, bias=False),
                                ResBlock(1024, kernel_size=3),
                                Upsampler(2, 1024)
                                ])
        self.decDepth2 = nn.Sequential(*[
                                nn.Conv2d(2048, 512, kernel_size=1, bias=False),
                                ResBlock(512, kernel_size=3),
                                Upsampler(2, 512)
                                ])
        self.decDepth3 = nn.Sequential(*[
                                nn.Conv2d(1024, 256, kernel_size=1, bias=False),
                                ResBlock(256, kernel_size=3),
                                Upsampler(2, 256)
                                ])
        self.decDepth4 = nn.Sequential(*[
                                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                                ResBlock(128, kernel_size=3),
                                Upsampler(2, 128)
                                ])
        self.decDepth = nn.Sequential(*[
                                nn.Conv2d(128+64, 128, kernel_size=1, bias=False),
                                ResBlock(128, kernel_size=3),
                                ResBlock(128, kernel_size=3),
                                Upsampler(2, 128),
                                nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=False)
                                ])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, fused_x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        # x = fused_x[0] # RGB
        # x_ir = fused_x[1] # IR
        # x = torch.cat([fused_x[0], fused_x[1]], 1)
        x = torch.cat([fused_x[0], fused_x[1], fused_x[2], fused_x[3]], 1) # torch.Size([64, 6, 224, 224]) 
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x0 = self.conv1(x) # torch.Size([batch_size, 64, 112, 112])
        x = self.bn1(x0)
        x = self.relu(x)
        x = self.maxpool(x) # torch.Size([batch_size, 64, 56, 56])

        x1 = self.layer1(x) # torch.Size([batch_size, 256, 56, 56])
        x2 = self.layer2(x1) # torch.Size([batch_size, 512, 28, 28])
        x3 = self.layer3(x2) # torch.Size([batch_size, 1024, 14, 14])
        x4 = self.layer4(x3) # torch.Size([batch_size, 2048, 7, 7])

        xf = self.decDepth1(x4) # torch.Size([batch_size, 1024, 14, 14])
        xf = torch.cat((x3, xf), 1)
        xf = self.decDepth2(xf) # torch.Size([batch_size, 512, 28, 28])
        xf = torch.cat((x2, xf), 1) 
        xf = self.decDepth3(xf) # torch.Size([batch_size, 256, 56, 56])
        xf = torch.cat((x1, xf), 1) 
        xf = self.decDepth4(xf) # torch.Size([batch_size, 128, 112, 112])
        xf = torch.cat((x0, xf), 1) 
        depth = self.decDepth(xf) # torch.Size([batch_size, 3, 112, 112])

        xf = self.avgpool(x4) # torch.Size([batch_size, 2048, 1, 1])
        xf = xf.view(xf.size(0), -1) # torch.Size([batch_size, 2048])

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam, depth


class REC3HMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params):
        print("Model: Reconstruct 3 modalities (IR, DEPTH, PM) & HMR.")
        self.inplanes = 64
        super(REC3HMR, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        self.Reconstruct_depth = Reconstruct()
        self.Reconstruct_ir = Reconstruct()
        self.Reconstruct_pm = Reconstruct()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, fused_x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        # x = fused_x[0] # RGB
        # x_ir = fused_x[1] # IR
        # x = torch.cat([fused_x[0], fused_x[1]], 1)
        x = torch.cat([fused_x[0], fused_x[1], fused_x[2], fused_x[3]], 1) # torch.Size([64, 6, 224, 224]) 
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x0 = self.conv1(x) # torch.Size([batch_size, 64, 112, 112])
        x = self.bn1(x0)
        x = self.relu(x)
        x = self.maxpool(x) # torch.Size([batch_size, 64, 56, 56])

        x1 = self.layer1(x) # torch.Size([batch_size, 256, 56, 56])
        x2 = self.layer2(x1) # torch.Size([batch_size, 512, 28, 28])
        x3 = self.layer3(x2) # torch.Size([batch_size, 1024, 14, 14])
        x4 = self.layer4(x3) # torch.Size([batch_size, 2048, 7, 7])

        depth = self.Reconstruct_depth(x0, x1, x2, x3, x4)
        ir = self.Reconstruct_ir(x0, x1, x2, x3, x4)
        pm = self.Reconstruct_pm(x0, x1, x2, x3, x4)

        xf = self.avgpool(x4) # torch.Size([batch_size, 2048, 1, 1])
        xf = xf.view(xf.size(0), -1) # torch.Size([batch_size, 2048])

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam, depth, ir, pm


class CASHMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params):
        print("Model: Cascade & Reconstruction depth.")
        self.inplanes = 64
        super(CASHMR, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        self.Reconstruct_depth = Reconstruct()
        # self.Reconstruct_ir = Reconstruct()
        # self.Reconstruct_pm = Reconstruct()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, fused_x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        # x = fused_x[0] # RGB
        # x_ir = fused_x[1] # IR
        # x = torch.cat([fused_x[0], fused_x[1]], 1)
        x = torch.cat([fused_x[0], fused_x[1], fused_x[2], fused_x[3]], 1) # torch.Size([64, 6, 224, 224]) 
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x0 = self.conv1(x) # torch.Size([batch_size, 64, 112, 112])
        x = self.bn1(x0)
        x = self.relu(x)
        x = self.maxpool(x) # torch.Size([batch_size, 64, 56, 56])

        x1 = self.layer1(x) # torch.Size([batch_size, 256, 56, 56])
        x2 = self.layer2(x1) # torch.Size([batch_size, 512, 28, 28])
        x3 = self.layer3(x2) # torch.Size([batch_size, 1024, 14, 14])
        x4 = self.layer4(x3) # torch.Size([batch_size, 2048, 7, 7])

        depth = self.Reconstruct_depth(x0, x1, x2, x3, x4)
        # ir = self.Reconstruct_ir(x0, x1, x2, x3, x4)
        # pm = self.Reconstruct_pm(x0, x1, x2, x3, x4)

        xf = self.avgpool(x4) # torch.Size([batch_size, 2048, 1, 1])
        xf = xf.view(xf.size(0), -1) # torch.Size([batch_size, 2048])

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam, depth
        # return pred_rotmat, pred_shape, pred_cam, depth, ir, pm

class Feat_extraction(nn.Module):
    def __init__(self, block, layers, input_dim=3):
        super(Feat_extraction, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x0 = self.conv1(x) # torch.Size([batch_size, 64, 112, 112])
        x = self.bn1(x0)
        x = self.relu(x)
        x = self.maxpool(x) # torch.Size([batch_size, 64, 56, 56])

        x1 = self.layer1(x) # torch.Size([batch_size, 256, 56, 56])
        x2 = self.layer2(x1) # torch.Size([batch_size, 512, 28, 28])
        x3 = self.layer3(x2) # torch.Size([batch_size, 1024, 14, 14])
        x4 = self.layer4(x3) # torch.Size([batch_size, 2048, 7, 7])


        return x0, x1, x2, x3, x4

class MULHMRFeatCat(nn.Module):
    """ MULHMRFeatCat Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params):
        print("Model: MULHMRFeatCat.")
        super(MULHMRFeatCat, self).__init__()
        npose = 24 * 6
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(2 * 512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.feat_extraction_rgb = Feat_extraction(block, layers)
        self.feat_extraction_ir = Feat_extraction(block, layers)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def forward(self, fused_x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        # fused_x[0] # RGB
        # fused_x[1] # IR
        batch_size = fused_x[0].shape[0]

        ## image concat
        # x = torch.cat([fused_x[0], fused_x[1]], 1)
        ## feature concat
        xf_rgb = self.feat_extraction_rgb(fused_x[0])
        xf_ir = self.feat_extraction_ir(fused_x[1])
        xf = torch.cat([xf_rgb, xf_ir], 1)

        # xf = self.avgpool(x4)
        xf = self.avgpool(xf)
        xf = xf.view(xf.size(0), -1)


        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam


class FeatCatCASHMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params):
        print("Model: Encoder: feat concat; Decoder: Cascade & Reconstruction depth.")
        self.inplanes = 64
        npose = 24 * 6
        super(FeatCatCASHMR, self).__init__()

        ## encoder
        self.feat_extraction_rgb = Feat_extraction(block, layers, input_dim=3)
        self.feat_extraction_ir = Feat_extraction(block, layers, input_dim=1)
        self.feat_extraction_depth = Feat_extraction(block, layers, input_dim=1)
        self.feat_extraction_pm = Feat_extraction(block, layers, input_dim=1)

        ## 3d pose regression
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion * 4 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        ## img reconstruction
        self.Reconstruct_depth = Reconstruct(num_feat=4)
        # self.Reconstruct_ir = Reconstruct(num_feat=4)
        # self.Reconstruct_pm = Reconstruct(num_feat=4)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, fused_x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        ## extract features
        batch_size = fused_x[0].shape[0]
        x0_rgb, x1_rgb, x2_rgb, x3_rgb, x4_rgb = self.feat_extraction_rgb(fused_x[0])
        _, _, _, _, x4_ir = self.feat_extraction_ir(fused_x[1])
        x0_depth, x1_depth, x2_depth, x3_depth, x4_depth = self.feat_extraction_depth(fused_x[2])
        _, _, _, _, x4_pm = self.feat_extraction_pm(fused_x[3])
        
        x4 = torch.cat([x4_rgb, x4_ir, x4_depth, x4_pm], 1) #torch.Size([Batch_size, 8192, 7, 7])

        ## img reconstraction
        depth = self.Reconstruct_depth(x0_depth, x1_depth, x2_depth, x3_depth, x4)
        # ir = self.Reconstruct_ir(x0, x1, x2, x3, x4)
        # pm = self.Reconstruct_pm(x0, x1, x2, x3, x4)

        ## estimate pose
        xf = self.avgpool(x4) # torch.Size([batch_size, 2048, 1, 1])
        xf = xf.view(xf.size(0), -1) # torch.Size([batch_size, 2048])

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam, depth
        # return pred_rotmat, pred_shape, pred_cam, depth, ir, pm



class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature 
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out, attention


class Cross_Attn(nn.Module):
    """ cross attention Layer"""
    def __init__(self, in_dim):
        super(Cross_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(4))

        self.softmax  = nn.Softmax(dim=-1) 

    def att_map(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 

        return attention

    def adding(self, x, attention):
        m_batchsize, C, width, height = x.size()
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        out = x
        for i, att in enumerate(attention):
            out = torch.bmm(proj_value, att.permute(0,2,1))
            out = out.view(m_batchsize, C, width, height)
        
            out = self.gamma[i]*out + out

        return out

    def forward(self, x, x_ir, x_depth, x_pm):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature 
                attention: B * N * N (N is Width*Height)
        """
        attention = self.att_map(x)
        attention_ir = self.att_map(x_ir)
        attention_depth = self.att_map(x_depth)
        attention_pm = self.att_map(x_pm)

        attention_list = [attention, attention_ir, attention_depth, attention_pm]
        x = self.adding(x, attention_list)
        x_ir = self.adding(x_ir, attention_list)
        x_depth = self.adding(x_depth, attention_list)
        x_pm = self.adding(x_pm, attention_list)

        out = torch.cat([x, x_ir, x_depth, x_pm], 1)

        return out, attention_list


class Feat_extraction_with_attention(nn.Module):
    def __init__(self, block, layers, input_dim=3):
        super(Feat_extraction, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        ## attention
        self.cross_att = Cross_Attn_ir_depth(in_dim=2048)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x[0] = x4_ir
        # x[1] = x4_depth

        x0_ir = self.conv1(x[0]) # torch.Size([batch_size, 64, 112, 112])
        x0_ir = self.bn1(x0_ir)
        x0_ir = self.relu(x0_ir)

        x0_depth = self.conv1(x[1]) # torch.Size([batch_size, 64, 112, 112])
        x0_depth = self.bn1(x0_depth)
        x0_depth = self.relu(x0_depth)

        ## attention
        x0, _ = self.cross_att(x0_ir, x0_depth)

        x = self.maxpool(x0) # torch.Size([batch_size, 64, 56, 56])
        x1 = self.layer1(x) # torch.Size([batch_size, 256, 56, 56])
        x2 = self.layer2(x1) # torch.Size([batch_size, 512, 28, 28])
        x3 = self.layer3(x2) # torch.Size([batch_size, 1024, 14, 14])
        x4 = self.layer4(x3) # torch.Size([batch_size, 2048, 7, 7])


        return x0, x1, x2, x3, x4


class Cross_Attn_ir_depth(nn.Module):
    """ cross attention Layer"""
    def __init__(self, in_dim):
        super(Cross_Attn_ir_depth,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(2))

        self.softmax  = nn.Softmax(dim=-1) 

    def att_map(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 

        return attention

    def adding(self, x, attention):
        m_batchsize, C, width, height = x.size()
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        out = x
        for i, att in enumerate(attention):
            out = torch.bmm(proj_value, att.permute(0,2,1))
            out = out.view(m_batchsize, C, width, height)
        
            out = self.gamma[i]*out + out

        return out

    def forward(self, x_ir, x_depth):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature 
                attention: B * N * N (N is Width*Height)
        """
        attention_ir = self.att_map(x_ir)
        attention_depth = self.att_map(x_depth)

        attention_list = [attention_ir, attention_depth]
        x_ir = self.adding(x_ir, attention_list)
        x_depth = self.adding(x_depth, attention_list)

        out = torch.cat([x_ir, x_depth], 1)

        return out, attention_list

class FeatAttCASHMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """  
    def __init__(self, block, layers, smpl_mean_params):
        print("Model: Encoder: feat attention fusion; Decoder: Cascade & Reconstruction depth.")
        self.inplanes = 64
        npose = 24 * 6
        super(FeatAttCASHMR, self).__init__()

        ## encoder
        self.feat_extraction_rgb = Feat_extraction(block, layers, input_dim=3)
        self.feat_extraction_ir = Feat_extraction(block, layers, input_dim=1)
        self.feat_extraction_depth = Feat_extraction(block, layers, input_dim=1)
        self.feat_extraction_pm = Feat_extraction(block, layers, input_dim=1)

        ## attention
        self.cross_att = Cross_Attn(in_dim=2048)

        ## 3d pose regression
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion * 4 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        ## img reconstruction
        self.Reconstruct_depth = Reconstruct(num_feat=4)
        # self.Reconstruct_ir = Reconstruct(num_feat=4)
        # self.Reconstruct_pm = Reconstruct(num_feat=4)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, fused_x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        ## extract features
        batch_size = fused_x[0].shape[0]
        x0_rgb, x1_rgb, x2_rgb, x3_rgb, x4_rgb = self.feat_extraction_rgb(fused_x[0])
        _, _, _, _, x4_ir = self.feat_extraction_ir(fused_x[1])
        x0_depth, x1_depth, x2_depth, x3_depth, x4_depth = self.feat_extraction_depth(fused_x[2])
        _, _, _, _, x4_pm = self.feat_extraction_pm(fused_x[3])

        ## attention
        x4, _ = self.cross_att(x4_rgb, x4_ir, x4_depth, x4_pm)
        # x4 = torch.cat([x4_rgb, x4_ir, x4_depth, x4_pm], 1) #torch.Size([Batch_size, 8192, 7, 7])

        ## img reconstraction
        depth = self.Reconstruct_depth(x0_depth, x1_depth, x2_depth, x3_depth, x4)
        # ir = self.Reconstruct_ir(x0, x1, x2, x3, x4)
        # pm = self.Reconstruct_pm(x0, x1, x2, x3, x4)

        ## estimate pose
        xf = self.avgpool(x4) # torch.Size([batch_size, 2048, 1, 1])
        xf = xf.view(xf.size(0), -1) # torch.Size([batch_size, 2048])

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam, depth
        # return pred_rotmat, pred_shape, pred_cam, depth, ir, pm

class IR_DEPTH_FeatAttCASHMR_V1(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """  
    def __init__(self, block, layers, smpl_mean_params):
        print("Model: for IR & DEPTH; Encoder: feat attention fusion; Decoder: Cascade & Reconstruction depth.")
        self.inplanes = 64
        npose = 24 * 6
        super(IR_DEPTH_FeatAttCASHMR, self).__init__()

        ## encoder
        self.feat_extraction_ir = Feat_extraction(block, layers, input_dim=1)
        self.feat_extraction_depth = Feat_extraction(block, layers, input_dim=1)

        ## attention
        self.cross_att = Cross_Attn_ir_depth(in_dim=2048)

        ## 3d pose regression
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion * 2 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        ## img reconstruction
        self.Reconstruct_depth_ir = Reconstruct(num_feat=2, out_dim=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, fused_x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        ## extract features
        batch_size = fused_x[0].shape[0]
        x0_ir, x1_ir, x2_ir, x3_ir, x4_ir = self.feat_extraction_ir(fused_x[0])
        x0_depth, x1_depth, x2_depth, x3_depth, x4_depth = self.feat_extraction_depth(fused_x[1])

        ## attention
        x4, _ = self.cross_att(x4_ir, x4_depth)
        # x4 = torch.cat([x4_rgb, x4_ir, x4_depth, x4_pm], 1) #torch.Size([Batch_size, 8192, 7, 7])

        ## img reconstraction
        rec_img = self.Reconstruct_depth_ir(torch.cat([x0_depth, x0_ir], 1),
                                        torch.cat([x1_depth, x1_ir], 1),
                                        torch.cat([x2_depth, x2_ir], 1),
                                        torch.cat([x3_depth, x3_ir], 1),
                                        x4)
        depth = rec_img[:,0,:,:].unsqueeze(1)
        ir = rec_img[:,1,:,:].unsqueeze(1)

        ## estimate pose
        xf = self.avgpool(x4) # torch.Size([batch_size, 2048, 1, 1])
        xf = xf.view(xf.size(0), -1) # torch.Size([batch_size, 2048])

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam, depth, ir
        # return pred_rotmat, pred_shape, pred_cam, depth, ir, pm


class IR_DEPTH_FeatAttCASHMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """  
    def __init__(self, block, layers, smpl_mean_params):
        print("Model: for IR & DEPTH; Encoder: feat attention fusion; Decoder: Cascade & Reconstruction depth.")
        self.inplanes = 64
        npose = 24 * 6
        super(IR_DEPTH_FeatAttCASHMR, self).__init__()

        ## encoder
        self.feat_extraction_ir = Feat_extraction_with_attention(block, layers, input_dim=1)
        # self.feat_extraction_depth = Feat_extraction_with_attention(block, layers, input_dim=1)

        ## 3d pose regression
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion * 2 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        ## img reconstruction
        self.Reconstruct_depth_ir = Reconstruct(num_feat=2, out_dim=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, fused_x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        ## extract features
        batch_size = fused_x[0].shape[0]
        # x0_ir, x1_ir, x2_ir, x3_ir, x4_ir = self.feat_extraction_ir(fused_x[0])
        # x0_depth, x1_depth, x2_depth, x3_depth, x4_depth = self.feat_extraction_depth(fused_x[1])

        # ## img reconstraction
        # rec_img = self.Reconstruct_depth_ir(torch.cat([x0_depth, x0_ir], 1),
        #                                 torch.cat([x1_depth, x1_ir], 1),
        #                                 torch.cat([x2_depth, x2_ir], 1),
        #                                 torch.cat([x3_depth, x3_ir], 1),
        #                                 torch.cat([x4_depth, x4_ir], 1))
        x0, x1, x2, x3, x4 = self.feat_extraction_ir(fused_x)

        ## img reconstraction
        rec_img = self.Reconstruct_depth_ir(x0, x1, x2, x3, x4)

        depth = rec_img[:,0,:,:].unsqueeze(1)
        ir = rec_img[:,1,:,:].unsqueeze(1)

        ## estimate pose
        xf = self.avgpool(x4) # torch.Size([batch_size, 2048, 1, 1])
        xf = xf.view(xf.size(0), -1) # torch.Size([batch_size, 2048])

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam, depth, ir
        # return pred_rotmat, pred_shape, pred_cam, depth, ir, pm



class IR_DEPTH_Fusion(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params, input_dim=2):
        print("Model: IR_DEPTH_Fusion.")
        self.focal_length = 5000. # constants.FOCAL_LENGTH 
        self.img_res = 224 # constants.IMG_RES 
        self.inplanes = 64
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        super(IR_DEPTH_Fusion, self).__init__()
        self.encoder_1 = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, input_dim)

        # self.encoder_2 = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, 5)

        self.dec1 = nn.Sequential(*[
                                nn.Conv2d(2048, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                ])

        # self.decIR1 = nn.Sequential(*[
                                # nn.Conv2d(2048, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # ])
        self.decIR2 = nn.Sequential(*[
                                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
                                ResBlock(64, kernel_size=3),
                                ])
        self.decIR3 = nn.Sequential(*[
                                nn.Conv2d(128+64+64, 64*4, kernel_size=3, padding=1, stride=1),
                                ResBlock(256, kernel_size=3),
                                nn.PixelShuffle(2),
                                nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
                                ])

        # self.decDepth1 = nn.Sequential(*[
                                # nn.Conv2d(2048, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # ])
        self.decDepth2 = nn.Sequential(*[
                                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
                                ResBlock(64, kernel_size=3),
                                ])
        self.decDepth3 = nn.Sequential(*[
                                nn.Conv2d(128+64+64, 64*4, kernel_size=3, padding=1, stride=1),
                                ResBlock(256, kernel_size=3),
                                nn.PixelShuffle(2),
                                nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
                                ])


    def get_mask(self, smpl, pred_rotmat, pred_betas, pred_camera, batch_size):
        '''
        project vertices to plane
        '''
        scale = 2
        mask_res = int(self.img_res / 2)
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.img_res * pred_camera[:,0] +1e-9)],dim=-1)


        camera_center = torch.zeros(batch_size, 2, device=self.device)

        projected_vertices_3d = perspective_projection(pred_vertices,
                                           rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                           translation=pred_cam_t,
                                           focal_length=self.focal_length,
                                           camera_center=camera_center,
                                           out_3d=True)
        projected_vertices_3d[:,:,:-1] = (projected_vertices_3d[:,:,:-1] + 0.5 * self.img_res) / scale

        padding_x = 500 
        padding_y = 500
        masks = torch.zeros([batch_size, 1, mask_res+padding_y*2, mask_res+padding_x*2], device=self.device)
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
        masks = masks[:, :, padding_y : mask_res + padding_y, padding_x : mask_res + padding_x]
        masks[masks > 0] = 1

        return masks

    def forward(self, fused_x, smpl, init_pose=None, init_shape=None, init_cam=None, n_iter=3, return_pose=False):
        # fused_x: ir_img, depth_img
        x = torch.cat([fused_x[0], fused_x[1]], 1) # torch.Size([64, 2, 224, 224]) 
        batch_size = x.shape[0]

        ## encoder 1
        pred_rotmat, pred_shape, pred_cam, pred_pose, x_feats = self.encoder_1(x, return_pose=True) # x, init_pose=None, init_shape=None, init_cam=None, n_iter=3
        
        ## generate masks
        masks = self.get_mask(smpl, pred_rotmat, pred_shape, pred_cam, batch_size)
        mask_l = torch.nn.functional.interpolate(masks.data, (self.img_res,self.img_res), mode='bilinear')
        ir_uncover = fused_x[0] * mask_l
        depth_uncover = fused_x[1] * mask_l
        # print('=> encoder_1 params: %.2fM' % (sum(p.numel() for p in self.encoder_1.parameters()) / (1024. * 1024)))
        # 25.72M  

        ## recovery ir
        xf = self.dec1(x_feats[-1]) # 39.38
        xf_ir_uncover = self.decIR2(ir_uncover) # 0
        ir_out = self.decIR3(torch.cat((xf, xf_ir_uncover, x_feats[0]), 1)) # 0.56
        ## recovery depth        
        xf_depth_uncover = self.decDepth2(depth_uncover)
        depth_out = self.decDepth3(torch.cat((xf, xf_depth_uncover, x_feats[0]), 1)) 

        # ## recovery ir
        # xf_ir = self.decIR1(x_feats[-1]) # 39.38
        # xf_ir_uncover = self.decIR2(ir_uncover) # 0
        # ir_out = self.decIR3(torch.cat((xf_ir, xf_ir_uncover, x_feats[0]), 1)) # 0.56
        # ## recovery depth        
        # xf_depth = self.decDepth1(x_feats[-1])
        # xf_depth_uncover = self.decDepth2(depth_uncover)
        # depth_out = self.decDepth3(torch.cat((xf_depth, xf_depth_uncover, x_feats[0]), 1)) 
        
        ## encoder 2
        if return_pose:
            pred_rotmat_1, pred_shape_1, pred_cam_1, pred_pose_1, _ = self.encoder_1(torch.cat([ir_out, depth_out], 1), return_pose=True) # x, init_pose=None, init_shape=None, init_cam=None, n_iter=3
            return pred_rotmat, pred_shape, pred_cam, pred_pose_1, pred_shape_1, pred_cam_1, ir_out, depth_out, mask_l

        else:
            pred_rotmat_1, pred_shape_1, pred_cam_1 = self.encoder_1(torch.cat([ir_out, depth_out], 1)) 
            # pred_rotmat_1, pred_shape_1, pred_cam_1 = self.encoder_1(
                # torch.cat([ir_out, depth_out], 1), init_pose=pred_pose, init_shape=pred_shape, init_cam=pred_cam) 
            # pred_rotmat_1, pred_shape_1, pred_cam_1 = self.encoder_2(
            #     torch.cat([x, mask_l, ir_uncover, depth_uncover], 1), init_pose=pred_pose, init_shape=pred_shape, init_cam=pred_cam) 

            return pred_rotmat, pred_shape, pred_cam, pred_rotmat_1, pred_shape_1, pred_cam_1, ir_out, depth_out, mask_l


class RGB_DEPTH_Fusion(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params, input_dim=4):
        print("Model: RGB_DEPTH_Fusion.")
        self.focal_length = 5000. # constants.FOCAL_LENGTH 
        self.img_res = 224 # constants.IMG_RES 
        self.inplanes = 64
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        super(RGB_DEPTH_Fusion, self).__init__()
        self.encoder_1 = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, input_dim)

        # self.encoder_2 = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, 5)

        self.dec1 = nn.Sequential(*[
                                nn.Conv2d(2048, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                ])

        # self.decIR1 = nn.Sequential(*[
                                # nn.Conv2d(2048, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # ])
        # self.decIR2 = nn.Sequential(*[
        #                         nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
        #                         ResBlock(64, kernel_size=3),
        #                         ])
        # self.decIR3 = nn.Sequential(*[
        #                         nn.Conv2d(128+64+64, 64*4, kernel_size=3, padding=1, stride=1),
        #                         ResBlock(256, kernel_size=3),
        #                         nn.PixelShuffle(2),
        #                         nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
        #                         ])

        # self.decDepth1 = nn.Sequential(*[
                                # nn.Conv2d(2048, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                # nn.PixelShuffle(2),
                                # ])
        self.decDepth2 = nn.Sequential(*[
                                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
                                ResBlock(64, kernel_size=3),
                                ])
        self.decDepth3 = nn.Sequential(*[
                                nn.Conv2d(128+64+64, 64*4, kernel_size=3, padding=1, stride=1),
                                ResBlock(256, kernel_size=3),
                                nn.PixelShuffle(2),
                                nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
                                ])


    def get_mask(self, smpl, pred_rotmat, pred_betas, pred_camera, batch_size):
        '''
        project vertices to plane
        '''
        scale = 2
        mask_res = int(self.img_res / 2)
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.img_res * pred_camera[:,0] +1e-9)],dim=-1)


        camera_center = torch.zeros(batch_size, 2, device=self.device)

        projected_vertices_3d = perspective_projection(pred_vertices,
                                           rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                           translation=pred_cam_t,
                                           focal_length=self.focal_length,
                                           camera_center=camera_center,
                                           out_3d=True)
        projected_vertices_3d[:,:,:-1] = (projected_vertices_3d[:,:,:-1] + 0.5 * self.img_res) / scale

        padding_x = 500 
        padding_y = 500
        masks = torch.zeros([batch_size, 1, mask_res+padding_y*2, mask_res+padding_x*2], device=self.device)
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
        masks = masks[:, :, padding_y : mask_res + padding_y, padding_x : mask_res + padding_x]
        masks[masks > 0] = 1

        return masks

    def forward(self, fused_x, smpl, init_pose=None, init_shape=None, init_cam=None, n_iter=3, return_pose=False):
        # fused_x: ir_img, depth_img
        x = torch.cat([fused_x[0], fused_x[1]], 1) # torch.Size([64, 2, 224, 224]) 
        batch_size = x.shape[0]

        ## encoder 1
        pred_rotmat, pred_shape, pred_cam, pred_pose, x_feats = self.encoder_1(x, return_pose=True) # x, init_pose=None, init_shape=None, init_cam=None, n_iter=3
        
        ## generate masks
        masks = self.get_mask(smpl, pred_rotmat, pred_shape, pred_cam, batch_size)
        mask_l = torch.nn.functional.interpolate(masks.data, (self.img_res,self.img_res), mode='bilinear')
        ir_uncover = fused_x[0] * mask_l
        depth_uncover = fused_x[1] * mask_l
        # print('=> encoder_1 params: %.2fM' % (sum(p.numel() for p in self.encoder_1.parameters()) / (1024. * 1024)))
        # 25.72M  

        # ## recovery ir
        xf = self.dec1(x_feats[-1]) # 39.38
        # xf_ir_uncover = self.decIR2(ir_uncover) # 0
        # ir_out = self.decIR3(torch.cat((xf, xf_ir_uncover, x_feats[0]), 1)) # 0.56
        ## recovery depth        
        xf_depth_uncover = self.decDepth2(depth_uncover)
        depth_out = self.decDepth3(torch.cat((xf, xf_depth_uncover, x_feats[0]), 1)) 

        # ## recovery ir
        # xf_ir = self.decIR1(x_feats[-1]) # 39.38
        # xf_ir_uncover = self.decIR2(ir_uncover) # 0
        # ir_out = self.decIR3(torch.cat((xf_ir, xf_ir_uncover, x_feats[0]), 1)) # 0.56
        # ## recovery depth        
        # xf_depth = self.decDepth1(x_feats[-1])
        # xf_depth_uncover = self.decDepth2(depth_uncover)
        # depth_out = self.decDepth3(torch.cat((xf_depth, xf_depth_uncover, x_feats[0]), 1)) 
        
        ## encoder 2
        if return_pose:
            pred_rotmat_1, pred_shape_1, pred_cam_1, pred_pose_1, _ = self.encoder_1(torch.cat([fused_x[0], depth_out], 1), return_pose=True) # x, init_pose=None, init_shape=None, init_cam=None, n_iter=3
            return pred_rotmat, pred_shape, pred_cam, pred_pose_1, pred_shape_1, pred_cam_1, depth_out, mask_l

        else:
            pred_rotmat_1, pred_shape_1, pred_cam_1 = self.encoder_1(torch.cat([fused_x[0], depth_out], 1)) 
            # pred_rotmat_1, pred_shape_1, pred_cam_1 = self.encoder_1(
                # torch.cat([ir_out, depth_out], 1), init_pose=pred_pose, init_shape=pred_shape, init_cam=pred_cam) 
            # pred_rotmat_1, pred_shape_1, pred_cam_1 = self.encoder_2(
            #     torch.cat([x, mask_l, ir_uncover, depth_uncover], 1), init_pose=pred_pose, init_shape=pred_shape, init_cam=pred_cam) 

            return pred_rotmat, pred_shape, pred_cam, pred_rotmat_1, pred_shape_1, pred_cam_1, depth_out, mask_l

class IR_DEPTH_PM_Fusion(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params, input_dim=5):
        print("Model: IR_DEPTH_PM_Fusion.")
        self.focal_length = 5000. # constants.FOCAL_LENGTH 
        self.img_res = 224 # constants.IMG_RES 
        self.inplanes = 64
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        super(IR_DEPTH_PM_Fusion, self).__init__()
        self.encoder_1 = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, input_dim)
        self.dec1 = nn.Sequential(*[
                                nn.Conv2d(2048, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                ])
        self.decIR2 = nn.Sequential(*[
                                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
                                ResBlock(64, kernel_size=3),
                                ])
        self.decIR3 = nn.Sequential(*[
                                nn.Conv2d(128+64+64, 64*4, kernel_size=3, padding=1, stride=1),
                                ResBlock(256, kernel_size=3),
                                nn.PixelShuffle(2),
                                nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
                                ])
        self.decDepth2 = nn.Sequential(*[
                                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
                                ResBlock(64, kernel_size=3),
                                ])
        self.decDepth3 = nn.Sequential(*[
                                nn.Conv2d(128+64+64, 64*4, kernel_size=3, padding=1, stride=1),
                                ResBlock(256, kernel_size=3),
                                nn.PixelShuffle(2),
                                nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
                                ])


    def get_mask(self, smpl, pred_rotmat, pred_betas, pred_camera, batch_size):
        '''
        project vertices to plane
        '''
        scale = 2
        mask_res = int(self.img_res / 2)
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.img_res * pred_camera[:,0] +1e-9)],dim=-1)


        camera_center = torch.zeros(batch_size, 2, device=self.device)

        projected_vertices_3d = perspective_projection(pred_vertices,
                                           rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                           translation=pred_cam_t,
                                           focal_length=self.focal_length,
                                           camera_center=camera_center,
                                           out_3d=True)
        projected_vertices_3d[:,:,:-1] = (projected_vertices_3d[:,:,:-1] + 0.5 * self.img_res) / scale

        padding_x = 500 
        padding_y = 500
        masks = torch.zeros([batch_size, 1, mask_res+padding_y*2, mask_res+padding_x*2], device=self.device)
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
        masks = masks[:, :, padding_y : mask_res + padding_y, padding_x : mask_res + padding_x]
        masks[masks > 0] = 1

        return masks

    def forward(self, fused_x, smpl, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        # fused_x: ir_img, depth_img
        x = torch.cat([fused_x[0], fused_x[1], fused_x[2], fused_x[3], fused_x[4]], 1) # torch.Size([64, 2, 224, 224]) 
        batch_size = x.shape[0]

        ## encoder 1
        # pred_rotmat, pred_shape, pred_cam, pred_pose, x_feats = self.encoder_1(x, return_pose=True) 
        pred_rotmat, pred_shape, pred_cam, pred_pose, x_feats = self.encoder_1(
            x, init_pose=init_pose, init_shape=init_shape, init_cam=init_cam, return_pose=True) 
        
        ## generate masks
        masks = self.get_mask(smpl, pred_rotmat, pred_shape, pred_cam, batch_size)
        mask_l = torch.nn.functional.interpolate(masks.data, (self.img_res,self.img_res), mode='bilinear')
        ir_uncover = fused_x[0] * mask_l
        depth_uncover = fused_x[1] * mask_l
        pm_uncover = fused_x[2] * mask_l

        ## recovery ir
        xf = self.dec1(x_feats[-1]) # 39.38
        xf_ir_uncover = self.decIR2(ir_uncover) # 0
        ir_out = self.decIR3(torch.cat((xf, xf_ir_uncover, x_feats[0]), 1)) # 0.56
        ## recovery depth        
        xf_depth_uncover = self.decDepth2(depth_uncover)
        depth_out = self.decDepth3(torch.cat((xf, xf_depth_uncover, x_feats[0]), 1)) 
        # ## recovery pm        
        # xf_pm_uncover = self.decPM2(pm_uncover)
        # pm_out = self.decPM3(torch.cat((xf, xf_pm_uncover, x_feats[0]), 1)) 

        ## encoder 2
        pred_rotmat_1, pred_shape_1, pred_cam_1 = self.encoder_1(
            torch.cat([ir_out, depth_out, fused_x[2], fused_x[0], fused_x[1]], 1)) 
        # pred_rotmat_1, pred_shape_1, pred_cam_1 = self.encoder_1(
        #     torch.cat([ir_out, depth_out, pm_out, fused_x[0], fused_x[1]], 1), 
        #     init_pose=pred_pose, init_shape=pred_shape, init_cam=pred_cam) 

        return pred_rotmat, pred_shape, pred_cam, pred_rotmat_1, pred_shape_1, pred_cam_1, ir_out, depth_out, mask_l

## add pm reconstruction 
class IR_DEPTH_PM_Fusion_modified(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params, input_dim=5):
        print("Model: IR_DEPTH_PM_Fusion.")
        self.focal_length = 5000. # constants.FOCAL_LENGTH 
        self.img_res = 224 # constants.IMG_RES 
        self.inplanes = 64
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        super(IR_DEPTH_PM_Fusion, self).__init__()
        self.encoder_1 = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, input_dim)
        self.dec1 = nn.Sequential(*[
                                nn.Conv2d(2048, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                ])
        self.decIR2 = nn.Sequential(*[
                                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
                                ResBlock(64, kernel_size=3),
                                ])
        self.decIR3 = nn.Sequential(*[
                                nn.Conv2d(128+64+64, 64*4, kernel_size=3, padding=1, stride=1),
                                ResBlock(256, kernel_size=3),
                                nn.PixelShuffle(2),
                                nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
                                ])
        self.decDepth2 = nn.Sequential(*[
                                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
                                ResBlock(64, kernel_size=3),
                                ])
        self.decDepth3 = nn.Sequential(*[
                                nn.Conv2d(128+64+64, 64*4, kernel_size=3, padding=1, stride=1),
                                ResBlock(256, kernel_size=3),
                                nn.PixelShuffle(2),
                                nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
                                ])
        self.decPM2 = nn.Sequential(*[
                                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
                                ResBlock(64, kernel_size=3),
                                ])
        self.decPM3 = nn.Sequential(*[
                                nn.Conv2d(128+64+64, 64*4, kernel_size=3, padding=1, stride=1),
                                ResBlock(256, kernel_size=3),
                                nn.PixelShuffle(2),
                                nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
                                ])


    def get_mask(self, smpl, pred_rotmat, pred_betas, pred_camera, batch_size):
        '''
        project vertices to plane
        '''
        scale = 2
        mask_res = int(self.img_res / 2)
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.img_res * pred_camera[:,0] +1e-9)],dim=-1)


        camera_center = torch.zeros(batch_size, 2, device=self.device)

        projected_vertices_3d = perspective_projection(pred_vertices,
                                           rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                           translation=pred_cam_t,
                                           focal_length=self.focal_length,
                                           camera_center=camera_center,
                                           out_3d=True)
        projected_vertices_3d[:,:,:-1] = (projected_vertices_3d[:,:,:-1] + 0.5 * self.img_res) / scale

        padding_x = 500 
        padding_y = 500
        masks = torch.zeros([batch_size, 1, mask_res+padding_y*2, mask_res+padding_x*2], device=self.device)
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
        masks = masks[:, :, padding_y : mask_res + padding_y, padding_x : mask_res + padding_x]
        masks[masks > 0] = 1

        return masks

    def forward(self, fused_x, smpl, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        # fused_x: ir_img, depth_img
        x = torch.cat([fused_x[0], fused_x[1], fused_x[2], fused_x[3], fused_x[4]], 1) # torch.Size([64, 2, 224, 224]) 
        batch_size = x.shape[0]

        ## encoder 1
        # pred_rotmat, pred_shape, pred_cam, pred_pose, x_feats = self.encoder_1(x, return_pose=True) 
        pred_rotmat, pred_shape, pred_cam, pred_pose, x_feats = self.encoder_1(
            x, init_pose=init_pose, init_shape=init_shape, init_cam=init_cam, return_pose=True) 
        
        ## generate masks
        masks = self.get_mask(smpl, pred_rotmat, pred_shape, pred_cam, batch_size)
        mask_l = torch.nn.functional.interpolate(masks.data, (self.img_res,self.img_res), mode='bilinear')
        ir_uncover = fused_x[0] * mask_l
        depth_uncover = fused_x[1] * mask_l
        pm_uncover = fused_x[2] * mask_l

        ## recovery ir
        xf = self.dec1(x_feats[-1]) # 39.38
        xf_ir_uncover = self.decIR2(ir_uncover) # 0
        ir_out = self.decIR3(torch.cat((xf, xf_ir_uncover, x_feats[0]), 1)) # 0.56
        ## recovery depth        
        xf_depth_uncover = self.decDepth2(depth_uncover)
        depth_out = self.decDepth3(torch.cat((xf, xf_depth_uncover, x_feats[0]), 1)) 
        ## recovery pm        
        xf_pm_uncover = self.decPM2(pm_uncover)
        pm_out = self.decPM3(torch.cat((xf, xf_pm_uncover, x_feats[0]), 1)) 

        ## encoder 2
        # pred_rotmat_1, pred_shape_1, pred_cam_1 = self.encoder_1(
        #     torch.cat([ir_out, depth_out, fused_x[2], fused_x[0], fused_x[1]], 1)) 
        pred_rotmat_1, pred_shape_1, pred_cam_1 = self.encoder_1(
            torch.cat([ir_out, depth_out, pm_out, fused_x[0], fused_x[1]], 1), 
            init_pose=pred_pose, init_shape=pred_shape, init_cam=pred_cam) 

        return pred_rotmat, pred_shape, pred_cam, pred_rotmat_1, pred_shape_1, pred_cam_1, ir_out, depth_out, pm_out, mask_l



class IR_DEPTH_PM_RGB_Fusion(nn.Module):
    """ IR_DEPTH_PM_RGB_Fusion
    """
    def __init__(self, block, layers, smpl_mean_params, input_dim=6):
        print("Model: IR_DEPTH_PM_RGB_Fusion.")
        self.focal_length = 5000. # constants.FOCAL_LENGTH 
        self.img_res = 224 # constants.IMG_RES 
        self.inplanes = 64
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        super(IR_DEPTH_PM_Fusion, self).__init__()
        self.encoder_1 = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, input_dim)
        self.dec1 = nn.Sequential(*[
                                nn.Conv2d(2048, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1, stride=1),
                                nn.PixelShuffle(2),
                                ])
        self.decIR2 = nn.Sequential(*[
                                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
                                ResBlock(64, kernel_size=3),
                                ])
        self.decIR3 = nn.Sequential(*[
                                nn.Conv2d(128+64+64, 64*4, kernel_size=3, padding=1, stride=1),
                                ResBlock(256, kernel_size=3),
                                nn.PixelShuffle(2),
                                nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
                                ])
        self.decDepth2 = nn.Sequential(*[
                                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
                                ResBlock(64, kernel_size=3),
                                ])
        self.decDepth3 = nn.Sequential(*[
                                nn.Conv2d(128+64+64, 64*4, kernel_size=3, padding=1, stride=1),
                                ResBlock(256, kernel_size=3),
                                nn.PixelShuffle(2),
                                nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
                                ])
        self.decPM2 = nn.Sequential(*[
                                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
                                ResBlock(64, kernel_size=3),
                                ])
        self.decPM3 = nn.Sequential(*[
                                nn.Conv2d(128+64+64, 64*4, kernel_size=3, padding=1, stride=1),
                                ResBlock(256, kernel_size=3),
                                nn.PixelShuffle(2),
                                nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
                                ])


    def get_mask(self, smpl, pred_rotmat, pred_betas, pred_camera, batch_size):
        '''
        project vertices to plane
        '''
        scale = 2
        mask_res = int(self.img_res / 2)
        pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.img_res * pred_camera[:,0] +1e-9)],dim=-1)


        camera_center = torch.zeros(batch_size, 2, device=self.device)

        projected_vertices_3d = perspective_projection(pred_vertices,
                                           rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                           translation=pred_cam_t,
                                           focal_length=self.focal_length,
                                           camera_center=camera_center,
                                           out_3d=True)
        projected_vertices_3d[:,:,:-1] = (projected_vertices_3d[:,:,:-1] + 0.5 * self.img_res) / scale

        padding_x = 500 
        padding_y = 500
        masks = torch.zeros([batch_size, 1, mask_res+padding_y*2, mask_res+padding_x*2], device=self.device)
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
        masks = masks[:, :, padding_y : mask_res + padding_y, padding_x : mask_res + padding_x]
        masks[masks > 0] = 1

        return masks

    def forward(self, fused_x, smpl, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        # fused_x: ir_img, depth_img
        x = torch.cat([fused_x[0], fused_x[1], fused_x[2], fused_x[3], fused_x[4]], 1) # torch.Size([64, 2, 224, 224]) 
        batch_size = x.shape[0]

        ## encoder 1
        # pred_rotmat, pred_shape, pred_cam, pred_pose, x_feats = self.encoder_1(x, return_pose=True) 
        pred_rotmat, pred_shape, pred_cam, pred_pose, x_feats = self.encoder_1(
            x, init_pose=init_pose, init_shape=init_shape, init_cam=init_cam, return_pose=True) 
        
        ## generate masks
        masks = self.get_mask(smpl, pred_rotmat, pred_shape, pred_cam, batch_size)
        mask_l = torch.nn.functional.interpolate(masks.data, (self.img_res,self.img_res), mode='bilinear')
        ir_uncover = fused_x[0] * mask_l
        depth_uncover = fused_x[1] * mask_l
        pm_uncover = fused_x[2] * mask_l

        ## recovery ir
        xf = self.dec1(x_feats[-1]) # 39.38
        xf_ir_uncover = self.decIR2(ir_uncover) # 0
        ir_out = self.decIR3(torch.cat((xf, xf_ir_uncover, x_feats[0]), 1)) # 0.56
        ## recovery depth        
        xf_depth_uncover = self.decDepth2(depth_uncover)
        depth_out = self.decDepth3(torch.cat((xf, xf_depth_uncover, x_feats[0]), 1)) 
        ## recovery pm        
        xf_pm_uncover = self.decPM2(pm_uncover)
        pm_out = self.decPM3(torch.cat((xf, xf_pm_uncover, x_feats[0]), 1)) 

        ## encoder 2
        # pred_rotmat_1, pred_shape_1, pred_cam_1 = self.encoder_1(
        #     torch.cat([ir_out, depth_out, fused_x[2], fused_x[0], fused_x[1]], 1)) 
        pred_rotmat_1, pred_shape_1, pred_cam_1 = self.encoder_1(
            torch.cat([ir_out, depth_out, pm_out, fused_x[0], fused_x[1]], 1), 
            init_pose=pred_pose, init_shape=pred_shape, init_cam=pred_cam) 

        return pred_rotmat, pred_shape, pred_cam, pred_rotmat_1, pred_shape_1, pred_cam_1, ir_out, depth_out, pm_out, mask_l



class IR_DEPTH_Fusion_large(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params, input_dim=2):
        print("Model: IR_DEPTH_Fusion.")
        self.focal_length = 5000. # constants.FOCAL_LENGTH 
        self.img_res = 224 # constants.IMG_RES 
        self.inplanes = 64
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        super(IR_DEPTH_Fusion, self).__init__()
        self.encoder = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, input_dim)

        self.Reconstruct = Reconstruct(out_dim=2)

    def forward(self, fused_x, smpl, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        # fused_x: ir_img, depth_img
        x = torch.cat([fused_x[0], fused_x[1]], 1) # torch.Size([64, 2, 224, 224]) 
        batch_size = x.shape[0]

        ## encoder 1
        pred_rotmat, pred_shape, pred_cam, pred_pose, x_feats = self.encoder(x, return_pose=True) # x, init_pose=None, init_shape=None, init_cam=None, n_iter=3
        
        # ## generate masks
        # masks = self.get_mask(smpl, pred_rotmat, pred_shape, pred_cam, batch_size)
        # mask_l = torch.nn.functional.interpolate(masks.data, (self.img_res,self.img_res), mode='bilinear')
        # ir_uncover = fused_x[0] * mask_l
        # depth_uncover = fused_x[1] * mask_l
        # # print('=> encoder_1 params: %.2fM' % (sum(p.numel() for p in self.encoder_1.parameters()) / (1024. * 1024)))
        # # 25.72M  

        ## recovery 
        recovery_img = self.Reconstruct(x_feats[0], x_feats[1], x_feats[2], x_feats[3], x_feats[4]) # 39.38
        ir_out = recovery_img[:,0,:,:]
        depth_out = recovery_img[:,1,:,:]

        ## encoder 2
        pred_rotmat_1, pred_shape_1, pred_cam_1 = self.encoder(recovery_img)
            # recovery_img, init_pose=pred_pose, init_shape=pred_shape, init_cam=pred_cam) 
        mask_l = None
        # pred_rotmat_1, pred_shape_1, pred_cam_1 = self.encoder_2(
        #     torch.cat([x, mask_l, ir_uncover, depth_uncover], 1), init_pose=pred_pose, init_shape=pred_shape, init_cam=pred_cam) 

        return pred_rotmat, pred_shape, pred_cam, pred_rotmat_1, pred_shape_1, pred_cam_1, ir_out, depth_out, mask_l


class IR_DEPTH_Fusion_ori(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """
    def __init__(self, block, layers, smpl_mean_params):
        print("Model: IR_DEPTH_Fusion.")
        self.inplanes = 64
        super(IR_DEPTH_Fusion, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

        self.decDepth1 = nn.Sequential(*[
                                nn.Conv2d(2048, 1024, kernel_size=1, bias=False),
                                ResBlock(1024, kernel_size=3),
                                Upsampler(2, 1024)
                                ])
        self.decDepth2 = nn.Sequential(*[
                                nn.Conv2d(2048, 512, kernel_size=1, bias=False),
                                ResBlock(512, kernel_size=3),
                                Upsampler(2, 512)
                                ])
        self.decDepth3 = nn.Sequential(*[
                                nn.Conv2d(1024, 256, kernel_size=1, bias=False),
                                ResBlock(256, kernel_size=3),
                                Upsampler(2, 256)
                                ])
        self.decDepth4 = nn.Sequential(*[
                                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                                ResBlock(128, kernel_size=3),
                                Upsampler(2, 128)
                                ])
        self.decDepth = nn.Sequential(*[
                                nn.Conv2d(128+64, 128, kernel_size=1, bias=False),
                                ResBlock(128, kernel_size=3),
                                ResBlock(128, kernel_size=3),
                                Upsampler(2, 128),
                                nn.Conv2d(128, 1, kernel_size=3, padding=1, bias=False)
                                ])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, fused_x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        # x = fused_x[0] # RGB
        # x_ir = fused_x[1] # IR
        x = torch.cat([fused_x[0], fused_x[1]], 1)
        # x = torch.cat([fused_x[0], fused_x[1], fused_x[2], fused_x[3]], 1) # torch.Size([64, 6, 224, 224]) 
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x0 = self.conv1(x) # torch.Size([batch_size, 64, 112, 112])
        x = self.bn1(x0)
        x = self.relu(x)
        x = self.maxpool(x) # torch.Size([batch_size, 64, 56, 56])

        x1 = self.layer1(x) # torch.Size([batch_size, 256, 56, 56])
        x2 = self.layer2(x1) # torch.Size([batch_size, 512, 28, 28])
        x3 = self.layer3(x2) # torch.Size([batch_size, 1024, 14, 14])
        x4 = self.layer4(x3) # torch.Size([batch_size, 2048, 7, 7])

        xf = self.decDepth1(x4) # torch.Size([batch_size, 1024, 14, 14])
        xf = torch.cat((x3, xf), 1)
        xf = self.decDepth2(xf) # torch.Size([batch_size, 512, 28, 28])
        xf = torch.cat((x2, xf), 1) 
        xf = self.decDepth3(xf) # torch.Size([batch_size, 256, 56, 56])
        xf = torch.cat((x1, xf), 1) 
        xf = self.decDepth4(xf) # torch.Size([batch_size, 128, 112, 112])
        xf = torch.cat((x0, xf), 1) 
        depth = self.decDepth(xf) # torch.Size([batch_size, 3, 112, 112])

        xf = self.avgpool(x4) # torch.Size([batch_size, 2048, 1, 1])
        xf = xf.view(xf.size(0), -1) # torch.Size([batch_size, 2048])

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam, depth



class Bodies_At_Rest(nn.Module):
    """ Bodies_At_Rest
    """  
    def __init__(self, mod1_input_dim=3, mod2_input_dim=4, smpl_mean_params=None):
        print("Model: Bodies_At_Rest.")
        npose = 24 * 6
        super(Bodies_At_Rest, self).__init__()
        self.CNN_packtanh = nn.Sequential(
            nn.Conv2d(mod1_input_dim, 192, kernel_size=7, stride=2, padding=3),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
        )      
        self.CNN_fc1 = nn.Sequential(nn.Linear(55296, 1024))

        # self.CNN_fc1 = nn.Linear(55296+npose+13, 1024)
        # self.drop1 = nn.Dropout()
        # self.CNN_fc2 = nn.Linear(1024, 1024)
        # self.drop2 = nn.Dropout()

        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        # mean_params = np.load(smpl_mean_params)
        # init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        # init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        # init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        # self.register_buffer('init_pose', init_pose)
        # self.register_buffer('init_shape', init_shape)
        # self.register_buffer('init_cam', init_cam)



        # ## img reconstruction
        self.CNN_packtanh_mode2 = nn.Sequential(
            nn.Conv2d(mod2_input_dim, 192, kernel_size=7, stride=2, padding=3),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
        )
        self.CNN_fc1_mode2 = nn.Sequential(nn.Linear(55296, 1024))

        # self.CNN_fc1_mode2 = nn.Linear(55296+npose+13, 1024)
        # self.drop1_mode2 = nn.Dropout()
        # self.CNN_fc2_mode2 = nn.Linear(1024, 1024)
        # self.drop2_mode2 = nn.Dropout()

        self.decpose_mode2 = nn.Linear(1024, npose)
        self.decshape_mode2 = nn.Linear(1024, 10)
        self.deccam_mode2 = nn.Linear(1024, 3)

        nn.init.xavier_uniform_(self.decpose_mode2.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape_mode2.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam_mode2.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # mean_params = np.load(smpl_mean_params)
        # init_pose_mode2 = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        # init_shape_mode2 = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        # init_cam_mode2 = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        # self.register_buffer('init_pose', init_pose_mode2)
        # self.register_buffer('init_shape', init_shape_mode2)
        # self.register_buffer('init_cam', init_cam_mode2)



    def forward(self, images, mode="0", init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = images.size(0)

        # if init_pose is None:
        #     init_pose = self.init_pose.expand(batch_size, -1)
        # if init_shape is None:
        #     init_shape = self.init_shape.expand(batch_size, -1)
        # if init_cam is None:
        #     init_cam = self.init_cam.expand(batch_size, -1)
        '''
        mode 0: in->out: pm_img->smpl parameters 
        mode 1: use the model in mode 0, but without grad 
        mode 2: in->out: [pm_img, est_pm]->smpl parameters 
        '''
        if mode == "0":
            scores_cnn = self.CNN_packtanh(images)
            scores_size = scores_cnn.size()
            scores_cnn = scores_cnn.view(batch_size, scores_size[1] *scores_size[2]*scores_size[3])

            scores = self.CNN_fc1(scores_cnn)
            pred_pose = self.decpose(scores)
            pred_shape = self.decshape(scores)
            pred_cam = self.deccam(scores)

            # pred_pose = init_pose
            # pred_shape = init_shape
            # pred_cam = init_cam
            # for i in range(n_iter):
            #     xc = torch.cat([scores_cnn, pred_pose, pred_shape, pred_cam],1)
            #     xc = self.CNN_fc1(xc)
            #     xc = self.drop1(xc)
            #     xc = self.CNN_fc2(xc)
            #     xc = self.drop2(xc)
            #     pred_pose = self.decpose(xc) + pred_pose
            #     pred_shape = self.decshape(xc) + pred_shape
            #     pred_cam = self.deccam(xc) + pred_cam
            
            pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

            return pred_rotmat, pred_shape, pred_cam, pred_pose
        elif mode == "1":
            with torch.no_grad():
                scores_cnn = self.CNN_packtanh(images)
                scores_size = scores_cnn.size()
                scores_cnn = scores_cnn.view(batch_size, scores_size[1] *scores_size[2]*scores_size[3])

                scores = self.CNN_fc1(scores_cnn)
                pred_pose = self.decpose(scores)
                pred_shape = self.decshape(scores)
                pred_cam = self.deccam(scores)
                
                # pred_pose = init_pose
                # pred_shape = init_shape
                # pred_cam = init_cam
                # for i in range(n_iter):
                #     xc = torch.cat([scores_cnn, pred_pose, pred_shape, pred_cam],1)
                #     xc = self.CNN_fc1(xc)
                #     xc = self.drop1(xc)
                #     xc = self.CNN_fc2(xc)
                #     xc = self.drop2(xc)
                #     pred_pose = self.decpose(xc) + pred_pose
                #     pred_shape = self.decshape(xc) + pred_shape
                #     pred_cam = self.deccam(xc) + pred_cam

                pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

            return pred_rotmat, pred_shape, pred_cam, pred_pose
        elif mode == "2":
            scores_cnn2 = self.CNN_packtanh_mode2(images)
            scores_size = scores_cnn2.size()
            scores_cnn2 = scores_cnn2.view(batch_size, scores_size[1] *scores_size[2]*scores_size[3])

            scores2 = self.CNN_fc1_mode2(scores_cnn2)
            pred_pose = self.decpose_mode2(scores2)
            pred_shape = self.decshape_mode2(scores2)
            pred_cam = self.deccam_mode2(scores2)

            # pred_pose = init_pose
            # pred_shape = init_shape
            # pred_cam = init_cam
            # for i in range(n_iter):
            #     xc = torch.cat([scores_cnn2, pred_pose, pred_shape, pred_cam],1)
            #     xc = self.CNN_fc1_mode2(xc)
            #     xc = self.drop1_mode2(xc)
            #     xc = self.CNN_fc2_mode2(xc)
            #     xc = self.drop2_mode2(xc)
            #     pred_pose = self.decpose_mode2(xc) + pred_pose
            #     pred_shape = self.decshape_mode2(xc) + pred_shape
            #     pred_cam = self.deccam_mode2(xc) + pred_cam
            
            pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

            return pred_rotmat, pred_shape, pred_cam, pred_pose


class Bodies_At_Rest_para(nn.Module):
    """ Bodies_At_Rest
    """  
    def __init__(self, mod1_input_dim=3, mod2_input_dim=4, smpl_mean_params=None):
        print("Model: Bodies_At_Rest.")
        npose = 24 * 6
        super(Bodies_At_Rest, self).__init__()
        self.CNN_packtanh = nn.Sequential(
            nn.Conv2d(mod1_input_dim, 192, kernel_size=7, stride=2, padding=3),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
        )      
        self.CNN_fc1 = nn.Sequential(nn.Linear(921984, 157))

        # ## img reconstruction
        self.CNN_packtanh_mode2 = nn.Sequential(
            nn.Conv2d(mod2_input_dim, 192, kernel_size=7, stride=2, padding=3),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
        )
        self.CNN_fc1_mode2 = nn.Sequential(nn.Linear(921984, 157))

    def forward(self, images, mode="0", init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = images.size(0)

        # if init_pose is None:
        #     init_pose = self.init_pose.expand(batch_size, -1)
        # if init_shape is None:
        #     init_shape = self.init_shape.expand(batch_size, -1)
        # if init_cam is None:
        #     init_cam = self.init_cam.expand(batch_size, -1)
        '''
        mode 0: in->out: pm_img->smpl parameters 
        mode 1: use the model in mode 0, but without grad 
        mode 2: in->out: [pm_img, est_pm]->smpl parameters 
        '''
        images= torch.cat([images, images[:,0,:,:].unsqueeze(1)],1)
        if mode == "0":
            scores_cnn = self.CNN_packtanh(images)
            scores_size = scores_cnn.size()
            scores_cnn = scores_cnn.view(batch_size, scores_size[1] *scores_size[2]*scores_size[3])

            scores = self.CNN_fc1(scores_cnn)
            pred_pose = scores[0:24*6]
            pred_shape = scores[24*6:24*6+13]
            pred_cam = scores[24*6+13:]

            pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

            return pred_rotmat, pred_shape, pred_cam, pred_pose
        elif mode == "1":
            with torch.no_grad():
                scores_cnn = self.CNN_packtanh(images)
                scores_size = scores_cnn.size()
                scores_cnn = scores_cnn.view(batch_size, scores_size[1] *scores_size[2]*scores_size[3])

                scores = self.CNN_fc1(scores_cnn)
                pred_pose = self.decpose(scores)
                pred_shape = self.decshape(scores)
                pred_cam = self.deccam(scores)
                pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

            return pred_rotmat, pred_shape, pred_cam, pred_pose
        elif mode == "2":
            scores_cnn2 = self.CNN_packtanh_mode2(images)
            scores_size = scores_cnn2.size()
            scores_cnn2 = scores_cnn2.view(batch_size, scores_size[1] *scores_size[2]*scores_size[3])

            scores2 = self.CNN_fc1_mode2(scores_cnn2)
            pred_pose = self.decpose_mode2(scores2)
            pred_shape = self.decshape_mode2(scores2)
            pred_cam = self.deccam_mode2(scores2)

            
            pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

            return pred_rotmat, pred_shape, pred_cam, pred_pose



def hmr(smpl_mean_params, model_name="hmr", pretrained=True, **kwargs):
    """ Constructs an HMR model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if model_name == "hmr":
        model = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, 3, **kwargs)
    if model_name == "hmr4mod":
        model = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, 6, **kwargs)
    elif model_name in ["irhmr","depthhmr","pmhmr"]:
        model = SingleHMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif model_name == "mulhmr":
        model = MULHMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif model_name == "featcat":
        model = MULHMRFeatCat(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif model_name in ["rechmr", "cashmr"]:
        model = RECHMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif model_name in ["cashmrV2"]:
        model = CASHMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif model_name in ["rec3hmr", "cas3hmr"]:
        model = REC3HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif model_name in ["featcat_cashmr"]:
        model = FeatCatCASHMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif model_name in ["featatt_cashmr"]:
        model = FeatAttCASHMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif model_name in ["ir_depth_featatt_cashmrV2"]:
        model = IR_DEPTH_FeatAttCASHMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif model_name in ["ir_depth_fusion", "ir_pm_fusion"]:
        model = IR_DEPTH_Fusion(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif model_name in ["rgb_depth_fusion", "rgb_pm_fusion"]:
        model = RGB_DEPTH_Fusion(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif model_name in ["ir_depth_pm_fusion"]:
        model = IR_DEPTH_PM_Fusion(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    elif model_name in ["ir_depth_pm_rgb_fusion"]:
        model = IR_DEPTH_PM_RGB_Fusion(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)


    elif model_name in ["bodiesAtRest"]:
        model = Bodies_At_Rest(mod1_input_dim=3, mod2_input_dim=4, smpl_mean_params=smpl_mean_params)
    elif model_name in ["bodiesAtRest4mod"]:
        model = Bodies_At_Rest(mod1_input_dim=8, mod2_input_dim=9, smpl_mean_params=smpl_mean_params)

    print('=> Model params: %.2fM' % (sum(p.numel() for p in model.parameters()) / (1024. * 1024)))

    # if pretrained and model_name not in ["mulhmr", "rechmr", "cashmr", "rec3hmr", "cas3hmr"]:
    #     resnet_imagenet = resnet.resnet50(pretrained=True)
    #     model.load_state_dict(resnet_imagenet.state_dict(),strict=False)
    return model

