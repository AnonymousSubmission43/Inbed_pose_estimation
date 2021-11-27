"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join
DATA_ROOT = '../../Dataset/pose/' 
H36M_ROOT = DATA_ROOT + 'human36m'
LSP_ROOT = DATA_ROOT + 'lsp'
LSP_ORIGINAL_ROOT = DATA_ROOT + 'lsp_original'
LSPET_ROOT = DATA_ROOT + 'lspextend_hr'
MPII_ROOT = DATA_ROOT + 'mpii'
COCO_ROOT = DATA_ROOT + 'coco'
MPI_INF_3DHP_ROOT = DATA_ROOT + 'mpi_inf_3dhp'
PW3D_ROOT = DATA_ROOT + '3DPW'
UPI_S1H_ROOT = DATA_ROOT + 'upi_s1h'
SLP_ROOT = DATA_ROOT + 'SLP/SLP/danaLab'

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = 'datasets/openpose'

# Path to test/train npz files
DATASET_FILES = [ {'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
                   'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2.npz'),
                   'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
                   '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                   'slp-rgb-uncover': join(DATASET_NPZ_PATH, 'slp_rgb_uncover_test.npz'),
                   'slp-rgb-cover1': join(DATASET_NPZ_PATH, 'slp_rgb_cover1_test.npz'),
                   'slp-rgb-cover2': join(DATASET_NPZ_PATH, 'slp_rgb_cover2_test.npz'),
                   'slp-ir-uncover': join(DATASET_NPZ_PATH, 'slp_ir_uncover_test.npz'),
                   'slp-ir-cover1': join(DATASET_NPZ_PATH, 'slp_ir_cover1_test.npz'),
                   'slp-ir-cover2': join(DATASET_NPZ_PATH, 'slp_ir_cover2_test.npz'),
                   'slp-uncover': join(DATASET_NPZ_PATH, 'slp_multi_mod_uncover_test.npz'),
                   'slp-cover1': join(DATASET_NPZ_PATH, 'slp_multi_mod_cover1_test.npz'),
                   'slp-cover2': join(DATASET_NPZ_PATH, 'slp_multi_mod_cover2_test.npz'),
                   'slp-rgb': join(DATASET_NPZ_PATH, 'slp_rgb_train.npz'),
                   'slp-ir': join(DATASET_NPZ_PATH, 'slp_ir_train.npz'),
                   'slp-multi': join(DATASET_NPZ_PATH, 'slp_multi_mod_train.npz'),
                   'slp-4mod-train': join(DATASET_NPZ_PATH, 'slp_4mod_train.npz'),
                   'slp-4mod-uncover': join(DATASET_NPZ_PATH, 'slp_4mod_uncover.npz'),
                   'slp-4mod-cover1': join(DATASET_NPZ_PATH, 'slp_4mod_cover1.npz'),
                   'slp-4mod-cover2': join(DATASET_NPZ_PATH, 'slp_4mod_cover2.npz')
                  },

                  {'h36m': join(DATASET_NPZ_PATH, 'h36m_train.npz'),
                   'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
                   'mpii': join(DATASET_NPZ_PATH, 'mpii_train.npz'),
                   'coco': join(DATASET_NPZ_PATH, 'coco_2014_train.npz'),
                   'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz'),
                   'slp': join(DATASET_NPZ_PATH, 'slp_rgb_uncover_train.npz'), # generate GT
                   'slp-rgb': join(DATASET_NPZ_PATH, 'slp_rgb_train.npz'),
                   'slp-ir': join(DATASET_NPZ_PATH, 'slp_ir_train.npz'),
                   'slp-multi': join(DATASET_NPZ_PATH, 'slp_multi_mod_train.npz'),
                   'slp-4mod-train': join(DATASET_NPZ_PATH, 'slp_4mod_train.npz')
                  } # train
                ]

DATASET_FOLDERS = {'slp': SLP_ROOT,
                   'slp-rgb': SLP_ROOT,
                   'slp-ir': SLP_ROOT,
                   'slp-multi': SLP_ROOT,
                   'slp-rgb-uncover': SLP_ROOT,
                   'slp-rgb-cover1': SLP_ROOT,
                   'slp-rgb-cover2': SLP_ROOT,
                   'slp-ir-uncover': SLP_ROOT,
                   'slp-ir-cover1': SLP_ROOT,
                   'slp-ir-cover2': SLP_ROOT,
                   'slp-uncover': SLP_ROOT,
                   'slp-cover1': SLP_ROOT,
                   'slp-cover2': SLP_ROOT,
                   'slp-4mod-train':  SLP_ROOT,
                   'slp-4mod-uncover':  SLP_ROOT,
                   'slp-4mod-cover1':  SLP_ROOT,
                   'slp-4mod-cover2':  SLP_ROOT,
                   'h36m': H36M_ROOT,
                   'h36m-p1': H36M_ROOT,
                   'h36m-p2': H36M_ROOT,
                   'lsp-orig': LSP_ORIGINAL_ROOT,
                   'lsp': LSP_ROOT,
                   'lspet': LSPET_ROOT,
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                   'mpii': MPII_ROOT,
                   'coco': COCO_ROOT,
                   '3dpw': PW3D_ROOT,
                   'upi-s1h': UPI_S1H_ROOT,
                }

CUBE_PARTS_FILE = 'data/cube_parts.npy'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy'
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy'
STATIC_FITS_DIR = 'data/static_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
