import os
import torch
import torchvision
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import scipy.misc

from torchvision import transforms

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.387, 0.391, 0.376], std=[0.214, 0.226, 0.212]),
    # transforms.Normalize(mean=[0.1748], std=[0.1151]),
])

preprocess2 = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.1748], std=[0.1151]),
])
# IMG_NORM_MEAN = [0.387, 0.391, 0.376]
# IMG_NORM_STD = [0.214, 0.226, 0.212]
# DEPTH_NORM_MEAN = [0.7444]
# DEPTH_NORM_STD = [0.1147]
# IR_NORM_MEAN = [0.1748]
# IR_NORM_STD = [0.1151]
# PM_NORM_MEAN = [0.00457]
# PM_NORM_STD = [0.0253]

if __name__ == '__main__':
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
	model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
	model = model.to(device)
	model.eval()

	data_root = '/media/yuyin/10THD1/Dataset/pose/SLP/SLP/danaLab/'
	sub_list = range(1, 102)
	img_types = ['RGB/uncover']
	for sub_ind in sub_list:
		print(f"Processing subject {sub_ind}...")
		sub_folder = "%05d" % (sub_ind)
		for img_type in img_types:
			imgs = range(45)
			for img_i in imgs:
				## load img
				imgname = 'image_%06d.png' % (img_i+1)
				imgname_full = os.path.join(sub_folder, img_type, imgname)
				image = PIL.Image.open(os.path.join(data_root, imgname_full))

				'''maskrcnn_resnet50_fpn
				'''
				# data
				# image_tensor = torchvision.transforms.functional.to_tensor(image)

				# ## model
				# # pass a list of (potentially different sized) tensors
				# # to the model, in 0-1 range. The model will take care of
				# # batching them together and normalizing
				# # output = model([image_tensor.to(device)]) # 'boxes', 'labels', 'scores', 'masks'

				# ## save maks
				# output = output[0]
				# # print(output['masks'].size())
				# masks = output['masks']
				# # print(masks.min(), masks.max())
				# # masks[masks>0.3] = 1
				# masks_show = masks.clone().cpu().detach().numpy()[0][0]



				'''deeplabv3_resnet101
				'''
				input_tensor = preprocess(image)
				# ir = PIL.Image.open(os.path.join(data_root, imgname_full).replace("RGB", "IR_aligned").replace("image_", ""))
				# ir_tensor = preprocess2(ir)

				# # ir_tensor = torch.cat([ir_tensor,ir_tensor,ir_tensor],0)
				# input_tensor = input_tensor * ir_tensor *10
				input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

				with torch.no_grad():
					output = model(input_batch.to(device))['out'][0]
				output_predictions = output.argmax(0)

				masks_show = output_predictions.clone().cpu().detach().numpy()
				masks_show = gaussian_filter(masks_show, sigma=1)
				masks_show[masks_show>0] = 1

				# ## save
				# folder_1 = os.path.join(data_root, sub_folder, 'masks')
				# if not os.path.exists(folder_1):
				# 	os.mkdir(folder_1)
				# folder_2 = os.path.join(folder_1, 'uncover')
				# if not os.path.exists(folder_2):
				# 	os.mkdir(folder_2)
				# save_imgname_full = os.path.join(folder_2, '%06d.png' % (img_i+1))
				# scipy.misc.imsave(save_imgname_full, masks_show)

				plt.figure()
				plt.imshow(masks_show)
				plt.show()



