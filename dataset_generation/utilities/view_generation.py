import os

os.environ["OMP_NUM_THREADS"] = "1"

import open3d as o3d
import numpy as np
from scipy import spatial
from tqdm.notebook import tqdm
from utilities.common import center, auto_radius, unit_vector, rotate_axis_random

# import random
from sklearn.cluster import KMeans, DBSCAN
from skspatial.objects import Plane, Points
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors


def view_generation(pointcloud: np.ndarray, camera_wd=0.5, boundary_cluster_num=20):
    """
    Search for the most likely optimal next views.
    :param pointcloud: The point cloud to be processed.
    # :param current_view: The current view position, expected to be (target_pos, camera_pos)
    :param camera_wd: The camera's working distance.
    """
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    
    pcd = o3d.t.geometry.PointCloud()
    # 点云位置和法线
    pcd.point.positions = o3d.core.Tensor(pointcloud[:, 0:3], dtype, device)
    pcd.point.normals = o3d.core.Tensor(pointcloud[:, 3:6], dtype, device)
    # print(f'pcd.point.positions.shape:{pcd.point.positions.shape}')
    # print(f'pcd.point.normals.shape:{pcd.point.normals.shape}')
    # print(f"{pcd.point.positions[0:1].numpy()}")
    # print(f"{pcd.point.normals[0:1].numpy()}")
    # exit()

    my_max_nn = 30
    my_max_nn = my_max_nn if my_max_nn < pcd.point.positions.shape[0] else pcd.point.positions.shape[0]
    my_radius = auto_radius(pcd.point.positions.numpy(), max_nn=my_max_nn)

    # print(pcd.point.positions.shape[0])
    # print(f"my_radius:{my_radius} mm")
    # print("-------------------")
    # exit()


    boundary, mask = pcd.compute_boundary_points(radius=my_radius, max_nn=my_max_nn, angle_threshold=120)  # mm

    # select randomly the camera position
    # cluster $boundary_cluster_num group
    # random select one of every group
    # cause the boundary is on the GPU so now move to CPU
    boundary_points = boundary.point.positions.numpy().reshape(-1, 3)
    boundary_normals = pcd.select_by_mask(mask).point.normals.numpy().reshape(-1, 3)

    # print (f'boundary_points.shape:{boundary_points.shape}')
    boundary_cluster_num = boundary_points.shape[0] if boundary_points.shape[0] < boundary_cluster_num else boundary_cluster_num

    if boundary_cluster_num == 0:
        return None

    boundary_points_kmeans = KMeans(n_clusters=boundary_cluster_num, n_init="auto").fit(boundary_points)
    boundary_points_kmeans_cluster = dict()
    for idx, label in enumerate(boundary_points_kmeans.labels_):
        curr_pos, curr_nol = boundary_points[idx], boundary_normals[idx]
        curr_p = np.concatenate([curr_pos, curr_nol], axis=0)
        if label in boundary_points_kmeans_cluster.keys():
            boundary_points_kmeans_cluster[label].append(curr_p)
        else:
            boundary_points_kmeans_cluster[label] = [curr_p]
    # boundary_points_kmeans_cluster
    print(f'boundary_points_kmeans_cluster:{len(boundary_points_kmeans_cluster)} clusters of camera positions')
    # print (f'boundary_points_kmeans_cluster:{boundary_points_kmeans_cluster}')


    # # DBSCAN
    # dbscan_min_samples = 5
    # neighbors = NearestNeighbors(n_neighbors=dbscan_min_samples)
    # neighbors_fit = neighbors.fit(boundary_points)
    # distances, indices = neighbors_fit.kneighbors(boundary_points)
    # distances = np.sort(distances[:, dbscan_min_samples-1], axis=0)
    # dbscan_eps = np.mean(distances)
    # print(f'Auto-calculated DBSCAN eps: {dbscan_eps:.4f}')
    
    # boundary_points_clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(boundary_points)
    # cluster_labels = boundary_points_clusterer.labels_
    
    # # 统计DBSCAN结果
    # n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    # n_noise = list(cluster_labels).count(-1)
    # print(f'DBSCAN clustering: {n_clusters} clusters found, {n_noise} noise points')

    # # 构建DBSCAN聚类结果字典
    # boundary_points_dbscan_cluster = dict()
    # for idx, label in enumerate(cluster_labels):
    #     # 跳过噪声点（标签为-1）
    #     if label == -1:
    #         continue
            
    #     curr_pos, curr_nol = boundary_points[idx], boundary_normals[idx]
    #     curr_p = np.concatenate([curr_pos, curr_nol], axis=0)
    #     if label in boundary_points_dbscan_cluster.keys():
    #         boundary_points_dbscan_cluster[label].append(curr_p)
    #     else:
    #         boundary_points_dbscan_cluster[label] = [curr_p]

    # print(f'boundary_points_dbscan_cluster: {len(boundary_points_dbscan_cluster)} clusters of camera positions')

    # # 如果没有有效聚类，返回None
    # if len(boundary_points_dbscan_cluster) == 0:
    #     print("No valid clusters found after DBSCAN")
    #     return None
    # boundary_points_kmeans_cluster = boundary_points_dbscan_cluster



    boundary_selected_pos = []
    for d_k, d_v in boundary_points_kmeans_cluster.items():
        s_i = np.random.choice(len(d_v))
        boundary_selected_pos.append(d_v[s_i])
    boundary_selected_pos = np.asarray(boundary_selected_pos)
    # print (f'boundary_selected_pos:{boundary_selected_pos}')

    filtered_points = pcd.point.positions.numpy().reshape(-1, 3)
    kd_tree = spatial.cKDTree(filtered_points, balanced_tree=False)
    view_info = []
    for p_i, p_v in enumerate(boundary_selected_pos):
        tar_p, p_n = p_v[0:3], p_v[3:6]
        _, idx = kd_tree.query(tar_p, k=my_max_nn)

        nei_points = filtered_points[idx[1:]]

        cp = center(nei_points)

        distance, closest_idx = kd_tree.query(cp, k=1)
        closest_point = filtered_points[closest_idx]
        # print(f"距离cp最近的点: {closest_point}")
        # print(f"距离: {distance}")
        # print tar_p  type
        # print(f"tar_p: {tar_p}, type: {type(tar_p)}")
        # print(f"closest_point: {closest_point}, type: {type(closest_point)}")

        if not np.array_equal(tar_p, closest_point):
            cp = closest_point

        vec = tar_p - cp
        p_outer_u = unit_vector(vec)

        view_direction = rotate_axis_random(p_outer_u, p_n)

        cam_p = tar_p + camera_wd * view_direction

        view_info.append(
            [
                tar_p,  # target position on original point cloud
                cam_p,  # camera position
                p_outer_u,  # direction to explore
                view_direction,
            ]
        )  # the normal vector of the plane at boundary target_pos to camera_pos
    # end for
    view_info = np.array(view_info).reshape(-1, 12)
    return view_info, boundary_points
