import torch
from torch.nn import functional as F
import numpy as np

"""
Useful geometric operations, e.g. Perspective projection and a differentiable Rodrigues formula
Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""
def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat    

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_rot6d(matrix):
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)
    Returns:
        6D rotation representation, of size (*, 6)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

def perspective_projection(points, rotation, translation, focal_length, camera_center, out_3d=False):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    if not out_3d:
        return projected_points[:, :, :-1]
    else:
        # # Apply perspective distortion
        # projected_points_2 = points# / points[:,:,-1].unsqueeze(-1)

        # # Apply camera intrinsics
        projected_points[:, :, -1] = torch.einsum('bij,bkj->bki', K, points)[:, :, -1]
        return projected_points



def estimate_translation_np(S, joints_2d, joints_conf, focal_length=5000, img_size=224):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length,focal_length])
    # optical center
    center = np.array([img_size/2., img_size/2.])

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(S, joints_2d, focal_length=5000., img_size=224.):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """

    device = S.device
    # Use only joints 25:49 (GT joints)
    S = S[:, 25:, :].cpu().numpy()
    joints_2d = joints_2d[:, 25:, :].cpu().numpy()
    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        trans[i] = estimate_translation_np(S_i, joints_i, conf_i, focal_length=focal_length, img_size=img_size)
    return torch.from_numpy(trans).to(device)



def vert2map(verts_taxel):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    dtype = torch.FloatTensor
    dtypeInt = torch.LongTensor
    batch_size = verts_taxel.size()[0]
    cbs= batch_size

    filler_taxels = []
    w = 112
    h = 112
    for i in range(w+1):
        for j in range(h+1):
            filler_taxels.append([i - 1, j - 1, 100])
    filler_taxels = torch.Tensor(filler_taxels).type(dtypeInt).unsqueeze(0).repeat(batch_size, 1, 1)
    mesh_patching_array = torch.zeros((batch_size, h+2, w+2, 4)).type(dtype)

    extra = verts_taxel*0
    extra[:, :, 0:2] += w*1.0
    extra[:, :, 2] += 100
    verts_taxel = torch.cat((verts_taxel, extra), dim = 1)
    verts_taxel[:, :, 0] -= 30
    verts_taxel[:, :, 1] -= 30
    verts_taxel[:, :, 2] *= 100
    print("verts_taxel3________>, ", verts_taxel.size()) #torch.Size([1, 13780, 3])
    print("________>, ", verts_taxel[:, :, 0].min(), verts_taxel[:, :, 0].max())
    print("________>, ", verts_taxel[:, :, 1].min(), verts_taxel[:, :, 1].max())
    print("________>, ", verts_taxel[:, :, 2].min(), verts_taxel[:, :, 2].max())

    verts_taxel_int = (verts_taxel).type(dtypeInt)

    # if get_mesh_bottom_dist == False:
    #     print("GETTING THE TOP MESH DIST")
    #     verts_taxel_int[:, :, 2] *= -1

    print("filler_taxels", filler_taxels[0:cbs, :, :].size())

    verts_taxel_int = torch.cat((filler_taxels[0:cbs, :, :], verts_taxel_int), dim=1)
    print("verts_taxel_int 1:", verts_taxel_int.size())
    print("________>, ", verts_taxel_int[:, :, 0].min(), verts_taxel_int[:, :, 0].max())
    print("________>, ", verts_taxel_int[:, :, 1].min(), verts_taxel_int[:, :, 1].max())
    print("________>, ", verts_taxel_int[:, :, 2].min(), verts_taxel_int[:, :, 2].max())

    vertice_sorting_method = (verts_taxel_int[:, :, 0:1] + 1) * 10000000 + \
                             (verts_taxel_int[:, :, 1:2] + 1) * 100000 + \
                             verts_taxel_int[:, :, 2:3]
    verts_taxel_int = torch.cat((vertice_sorting_method, verts_taxel_int), dim=2)
    print("verts_taxel_int bef for loop:", verts_taxel_int.size())
    print("________>, ", verts_taxel_int[:, :, 0].min(), verts_taxel_int[:, :, 0].max())
    print("________>, ", verts_taxel_int[:, :, 1].min(), verts_taxel_int[:, :, 1].max())
    print("________>, ", verts_taxel_int[:, :, 2].min(), verts_taxel_int[:, :, 2].max())


    for i in range(cbs):
        x = torch.unique(verts_taxel_int[i, :, :], sorted=True, return_inverse=False,
                         dim=0)  # this takes the most time
        print("x", x.size()) # torch.Size([54659, 4])

        x[1:, 0] = torch.abs((x[:-1, 1] - x[1:, 1]) + (x[:-1, 2] - x[1:, 2]))
        print("x2", x.size())
        x2 = x.clone()
        x2[1:, 1:4] = x2[1:, 1:4] * x2[1:, 0:1]
        x = x[x2[:, 0] != 0, :]
        x = x[x[:, 0] != 0, :]
        x = x[:, 1:]
        print("x5", x.size())
        print(min(x[:, 0]), max(x[:, 0]))
        print(min(x[:, 1]), max(x[:, 1]))
        x = x[x[:, 1] < h, :]
        print("x6", x.size())
        x = x[x[:, 1] > 0, :]
        print("x7", x.size())
        x = x[x[:, 0] < w, :]
        print("x8", x.size())
        x = x[x[:, 0] > 0, :]

        print("x2", x.size()) # torch.Size([50176, 3])

        mesh_matrix = x[0:w*h, 2].view(w, h)

        if i == 0:
            # print mesh_matrix[0:15, 32:]
            mesh_matrix = mesh_matrix.transpose(0, 1).flip(0).unsqueeze(0)
            # print mesh_matrix[0, 1:32, 0:15]
            mesh_matrix_batch = mesh_matrix.clone()
        else:
            mesh_matrix = mesh_matrix.transpose(0, 1).flip(0).unsqueeze(0)
            mesh_matrix_batch = torch.cat((mesh_matrix_batch, mesh_matrix), dim=0)

        # t3 = time.time()
    # print i, t3 - t2, t2 - t1

    mesh_matrix_batch[mesh_matrix_batch == 100] = 0

    mesh_matrix_batch = mesh_matrix_batch.type(dtype)
    mesh_matrix_batch *= 0.0286  # shouldn't need this. leave as int.

    mesh_patching_array *= 0
    mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 0] = mesh_matrix_batch.clone()
    mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 0][mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 0] > 0] = 0
    mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 0] = mesh_patching_array[0:cbs, 0:h, 0:w, 0] + \
                                                     mesh_patching_array[0:cbs, 1:h+1, 0:w, 0] + \
                                                     mesh_patching_array[0:cbs, 2:h+2, 0:w, 0] + \
                                                     mesh_patching_array[0:cbs, 0:h, 1:w+1, 0] + \
                                                     mesh_patching_array[0:cbs, 2:h+2, 1:w+1, 0] + \
                                                     mesh_patching_array[0:cbs, 0:h, 2:w+2, 0] + \
                                                     mesh_patching_array[0:cbs, 1:h+1, 2:w+2, 0] + \
                                                     mesh_patching_array[0:cbs, 2:h+2, 2:w+2, 0]
    mesh_patching_array[0:cbs, :, :, 0] /= 8
    mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 1] = mesh_matrix_batch.clone()
    mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 1][mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 1] < 0] = 0
    mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 1][mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 1] >= 0] = 1
    mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 2] = mesh_patching_array[0:cbs, 1:h+1, 1:w+1,
                                                     0] * mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 1]
    mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 3] = mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 2].clone()
    mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 3][mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 3] != 0] = 1.
    mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 3] = 1 - mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 3]
    mesh_matrix_batch = mesh_matrix_batch * mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 3]
    mesh_matrix_batch += mesh_patching_array[0:cbs, 1:h+1, 1:w+1, 2]
    mesh_matrix_batch = mesh_matrix_batch.type(dtypeInt)


    contact_matrix_batch = mesh_matrix_batch.clone()
    contact_matrix_batch[contact_matrix_batch >= 0] = 1
    contact_matrix_batch[contact_matrix_batch < 0] = 0

    #print mesh_matrix_batch

    #print torch.min(mesh_matrix_batch[0, :, :]), torch.max(mesh_matrix_batch[0, :, :]), "A"

    # if get_mesh_bottom_dist == False:
    #     mesh_matrix_batch *= -1

    #print torch.min(mesh_matrix_batch[0, :, :]), torch.max(mesh_matrix_batch[0, :, : ])
    print("-------->",mesh_matrix_batch.size()) #torch.Size([batch_size, 64, 27])
    print("-------->",contact_matrix_batch.size()) #torch.Size([batch_size, 64, 27])
    from matplotlib import rc
    import matplotlib.pyplot as plt
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(mesh_matrix_batch.cpu().numpy()[0])
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(contact_matrix_batch.cpu().numpy()[0])
    plt.show()

    return mesh_matrix_batch, contact_matrix_batch
