import open3d as o3d
import numpy as np
import time
import os
from datetime import datetime
import copy

def calculate_intrinsic(width, height, fov=60):
    aspect = width / height   
    fov_rad = np.radians(fov)
    f = 1.0 / np.tan(fov_rad / 2)
    
    # proj_matrix
    P_0 = f / aspect  # proj_matrix[0]
    P_1 = f           # proj_matrix[5]
    P_2 = 0           # proj_matrix[2] 
    P_3 = 0           # proj_matrix[6] 
    
    # 计算内参
    fx = P_0 * width / 2
    fy = P_1 * height / 2
    cx = (-P_2 * width + width) / 2  
    cy = (P_3 * height + height) / 2  
    
    # 内参矩阵
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # print(f"fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    
    return intrinsic

def capture_pointcloud_from_mesh(mesh_path, width=640, height=480, fov=60, near=0.1, far=10, depth_trunc_mode="auto"):
    
    
    intrinsic = calculate_intrinsic(width, height, fov)
    
    # 加载3D网格模型
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    
    # 获取模型中心和尺寸
    mesh_center = mesh.get_center()
    mesh_size = np.max(mesh.get_max_bound() - mesh.get_min_bound())
    camera_distance = mesh_size * 2
    
    # 设置渲染环境
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=True)
    vis.add_geometry(mesh)
    
    # 设置相机视角参数
    ctrl = vis.get_view_control()
    front = [0, 0, -1]  # 相机朝向 - 看向z轴负方向
    # camera_pos = mesh_center + np.array([0, 0, camera_distance])
    
    # 相机参数
    ctrl.set_front(front)
    ctrl.set_lookat(mesh_center)
    ctrl.set_up([0, -1, 0])
    
    # 渲染 捕获 rgbd
    vis.poll_events()
    vis.update_renderer()
    depth = vis.capture_depth_float_buffer(True)
    color = vis.capture_screen_float_buffer(True)
    
    time.sleep(1)
    
    depth_o3d = o3d.geometry.Image(np.asarray(depth))
    color_o3d = o3d.geometry.Image((np.asarray(color) * 255).astype(np.uint8))

    filename_lower = os.path.basename(mesh_path).lower()
    if any(keyword in filename_lower for keyword in ["bunny","armadillo","happy"]):
        depth_scale = 1.0
        print("使用depth_scale=1.0")
    elif "sds"  in filename_lower:
        depth_scale = 1000.0
        print("使用depth_scale=1000.0")
    else:
        depth_scale = 1.0
        print("使用depth_scale=1000.0")
    
    if depth_trunc_mode == "auto":
        depth_trunc = camera_distance * 3
    if depth_trunc_mode == "pybullet":
        depth_trunc = far * near / (far - (far - near) * depth)
    
    # 创建RGBD图像
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )
    # 从RGBD图像生成点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    # pcd_before = copy.deepcopy(pcd)  # 深拷贝
    # print(intrinsic)
    
    # 设置外参矩阵
    view_param = ctrl.convert_to_pinhole_camera_parameters()
    extrinsic = view_param.extrinsic
    # print(extrinsic)
    # exit()
    pcd.transform(np.linalg.inv(extrinsic))

    # pcd_before = pcd

    # pcd.transform(np.linalg.inv(extrinsic))
    # pcd.transform(extrinsic)
    # extrinsic = np.eye(4)
    # extrinsic[0:3, 3] = mesh_center
    # pcd.transform(np.linalg.inv(extrinsic))
    
    # 计算点云尺寸进行比较
    # pcd_size = np.max(pcd.get_max_bound() - pcd.get_min_bound())
    # print(f"点云尺寸: {pcd_size}")
    # print(f"尺寸比例 (点云/网格): {pcd_size/mesh_size:.4f}")
    
    vis.destroy_window()
    
    # return pcd_before , pcd
    return pcd

# 主函数
if __name__ == "__main__":
    input_file = "armadillo.obj"
    
    pcd = capture_pointcloud_from_mesh(input_file)
    # pcd_before , pcd = capture_pointcloud_from_mesh(input_file)
    
    base_name = os.path.splitext(input_file)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{base_name}_{timestamp}_pcd.ply"
    # output_file_before = f"{base_name}_{timestamp}_pcd_before.ply"
    
    o3d.io.write_point_cloud(output_file, pcd)
    # o3d.io.write_point_cloud(output_file_before, pcd_before)
    print(f"点云已保存为 '{output_file}'")
    
    # 可视化
    # o3d.visualization.draw_geometries([pcd])