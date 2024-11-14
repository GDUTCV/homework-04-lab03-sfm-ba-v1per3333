import numpy as np
import cv2


def compute_ba_residuals(parameters: np.ndarray, intrinsics: np.ndarray, num_cameras: int, points2d: np.ndarray,
                         camera_idxs: np.ndarray, points3d_idxs: np.ndarray) -> np.ndarray:
    """
    For each point2d in <points2d>, find its 3d point, reproject it back into the image and return the residual
    i.e. euclidean distance between the point2d and reprojected point.

    Args:
        parameters: list of camera parameters [r1, r2, r3, t1, t2, t3, ...] where r1, r2, r3 corresponds to the
                    Rodriguez vector. There are 6C + 3M parameters where C is the number of cameras
        intrinsics: camera intrinsics 3 x 3 array
        num_cameras: number of cameras, C
        points2d: N x 2 array of 2d points
        camera_idxs: camera_idxs[i] returns the index of the camera for points2d[i]
        points3d_idxs: points3d[points3d_idxs[i]] returns the 3d point corresponding to points2d[i]

    Returns:
        N residuals

    """
    num_camera_parameters = 6 * num_cameras
    camera_parameters = parameters[:num_camera_parameters]
    points3d = parameters[num_camera_parameters:]
    num_points3d = points3d.shape[0] // 3
    points3d = points3d.reshape(num_points3d, 3)

    camera_parameters = camera_parameters.reshape(num_cameras, 6)
    camera_rvecs = camera_parameters[:, :3]
    camera_tvecs = camera_parameters[:, 3:]

    extrinsics = []
    for rvec in camera_rvecs:
        rot_mtx, _ = cv2.Rodrigues(rvec)
        extrinsics.append(rot_mtx)
    extrinsics = np.array(extrinsics)  # C x 3 x 3
    extrinsics = np.concatenate([extrinsics, camera_tvecs.reshape(-1, 3, 1)], axis=2)  # C x 3 x 4

    residuals = np.zeros(shape=points2d.shape[0], dtype=float)
    """ 
    YOUR CODE HERE: 
    NOTE: DO NOT USE LOOPS 
    HINT: I used np.matmul; np.sum; np.sqrt; np.square, np.concatenate etc.
    """
    selected_points3d = points3d[points3d_idxs]  # 根据索引选择3D点

    # 将3D点转换为齐次坐标（加一维为1）
    homo_3d_points = np.concatenate((selected_points3d, np.ones((selected_points3d.shape[0], 1))), axis=1)
    homo_3d_points_T = np.transpose(homo_3d_points)  # 转置齐次3D点

    selected_extrinsics = extrinsics[camera_idxs]  # 根据相机索引选择外参矩阵
    P = np.matmul(intrinsics, selected_extrinsics)  # 计算投影矩阵

    # 计算2D投影点
    calculated_points2d = np.einsum('ijk,ki->ij', P, homo_3d_points_T)
    calculated_points2d /= calculated_points2d[:, -1].reshape((calculated_points2d.shape[0], 1))  # 归一化
    calculated_points2d = calculated_points2d[:, :-1]  # 去掉齐次坐标的最后一维

    # 计算残差：2D点与计算的2D投影点的欧几里得距离
    residuals = np.linalg.norm(points2d - calculated_points2d, axis=1)

    """ END YOUR CODE HERE """
    return residuals
