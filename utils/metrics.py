import torch
import numpy as np
import open3d
import mcubes


def calculate_iou(prediction, gt):
    """  The prediction and gt are 3 dim voxels. Each voxel has values 1 or 0"""
    prediction = prediction.to(torch.int8)
    gt = gt.to(torch.int8)
    intersection = torch.sum(torch.logical_and(prediction, gt), dim=[1, 2, 3])
    union = torch.sum(torch.logical_or(prediction, gt), dim=[1, 2, 3])
    return intersection / union


def calculate_fscore(list_pr: np.array, list_gt: np.array, th: float = 0.01) -> float:
    """
        based on: https://github.com/lmb-freiburg/what3d

        Calculates the F-score between two point clouds with the corresponding threshold value.
    """
    num_sampled_pts = 8192
    assert list_pr.shape == list_gt.shape
    b_size = list_gt.shape[0]

    list_gt, list_pr = list_gt.detach().cpu().numpy(), list_pr.detach().cpu().numpy()

    result = []

    for i in range(b_size):
        gt, pr = list_gt[i], list_pr[i]

        if (gt.sum() == 0 and pr.sum() != 0) or (gt.sum() != 0 and pr.sum() == 0):
            result.append(0)
            continue

        gt = voxel_grid_to_mesh(gt).sample_points_uniformly(num_sampled_pts)
        pr = voxel_grid_to_mesh(pr).sample_points_uniformly(num_sampled_pts)

        d1 = gt.compute_point_cloud_distance(pr)
        d2 = pr.compute_point_cloud_distance(gt)

        if len(d1) and len(d2):
            recall = float(sum(d < th for d in d2)) / float(len(d2))
            precision = float(sum(d < th for d in d1)) / float(len(d1))

            if recall + precision > 0:
                fscore = 2 * recall * precision / (recall + precision)
            else:
                fscore = 0
        else:
            fscore = 0
        result.append(fscore)

    return np.array(result)


def voxel_grid_to_mesh(vox_grid: np.array) -> open3d.geometry.TriangleMesh:
    """
        taken from: https://github.com/lmb-freiburg/what3d

        Converts a voxel grid represented as a numpy array into a mesh.
    """
    sp = vox_grid.shape
    if len(sp) != 3 or sp[0] != sp[1] or \
            sp[1] != sp[2] or sp[0] == 0:
        raise ValueError("Only non-empty cubic 3D grids are supported.")
    padded_grid = np.pad(vox_grid, ((1, 1), (1, 1), (1, 1)), 'constant')
    m_vert, m_tri = mcubes.marching_cubes(padded_grid, 0)
    m_vert = m_vert / (padded_grid.shape[0] - 1)
    out_mesh = open3d.geometry.TriangleMesh()
    out_mesh.vertices = open3d.utility.Vector3dVector(m_vert)
    out_mesh.triangles = open3d.utility.Vector3iVector(m_tri)
    return out_mesh
