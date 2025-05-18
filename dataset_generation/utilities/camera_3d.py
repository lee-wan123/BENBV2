import open3d as o3d
import numpy as np
import time
import os
from datetime import datetime
import copy
import math
import pybullet as p
from env_3d import *

class Camera:
    def __init__(self, cam_pos, cam_tar, cam_up_vector, near=0.1, far=10, size=(640, 480), fov=60, env=None , mesh_path=None,use_env=False,mode="depth"):
        self.cam_pos = cam_pos
        self.cam_tar = cam_tar
        self.cam_up_vector = cam_up_vector
        self.near = near
        self.far = far
        self.width, self.height = size
        self.fov = fov
        self.use_env = use_env
        self.mode = mode
        self.mesh_path = mesh_path
        self.env = env
        
    def projection_matrix(self):
        # btScalar aspect;
        # float width = float(renderer->getScreenWidth());
        # float height = float(renderer->getScreenHeight());
        # aspect = width / height;

        # float projectionMatrix[16]
        # float yScale = 1.0 / tan((B3_PI / 180.0) * fov / 2);
        # float xScale = yScale / aspect;

        # projectionMatrix[0 * 4 + 0] = xScale;
        # projectionMatrix[0 * 4 + 1] = float(0);
        # projectionMatrix[0 * 4 + 2] = float(0);
        # projectionMatrix[0 * 4 + 3] = float(0);

        # projectionMatrix[1 * 4 + 0] = float(0);
        # projectionMatrix[1 * 4 + 1] = yScale;
        # projectionMatrix[1 * 4 + 2] = float(0);
        # projectionMatrix[1 * 4 + 3] = float(0);

        # projectionMatrix[2 * 4 + 0] = 0;
        # projectionMatrix[2 * 4 + 1] = 0;
        # projectionMatrix[2 * 4 + 2] = (nearVal + farVal) / (nearVal - farVal);
        # projectionMatrix[2 * 4 + 3] = float(-1);

        # projectionMatrix[3 * 4 + 0] = float(0);
        # projectionMatrix[3 * 4 + 1] = float(0);
        # projectionMatrix[3 * 4 + 2] = (float(2) * farVal * nearVal) / (nearVal - farVal);
        # projectionMatrix[3 * 4 + 3] = float(0);

        aspect = self.width / self.height
        projection_matrix = [0.0] * 16
        y_scale = 1.0 / math.tan((math.pi / 180.0) * self.fov / 2)
        x_scale = y_scale / aspect
        
        projection_matrix[0] = x_scale  
        projection_matrix[5] = y_scale  
        projection_matrix[10] = (self.near + self.far) / (self.near - self.far)  
        projection_matrix[11] = -1.0  
        projection_matrix[14] = (2.0 * self.far * self.near) / (self.near - self.far)  

        # projection_matrix = np.array(projection_matrix, dtype=np.float32).reshape((4, 4))
        
        return projection_matrix

    def calculate_intrinsic(self, proj_matrix):
        # NBV
        # P_0 = projection_matrix[0][0]  # proj_matrix[0]
        # P_1 = projection_matrix[1][1]  # proj_matrix[5]
        # P_2 = projection_matrix[0][2]  # proj_matrix[2] 
        # P_3 = projection_matrix[1][2]  # proj_matrix[6] 
        
        # fx = P_0 * width / 2
        # fy = P_1 * height / 2
        # cx = (-P_2 * width + width) / 2  
        # cy = (P_3 * height + height) / 2  
        
        # intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        # print(f"fx={fx}, fy={fy}, cx={cx}, cy={cy}")

        # print(proj_matrix)

        P_0, P_1 = proj_matrix[0], proj_matrix[5]
        P_2, P_3 = proj_matrix[2], proj_matrix[6]

        # intrinsic = o3d.camera.PinholeCameraIntrinsic(
        #         # o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
        #         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault )
        # print(intrinsic)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.width,
            height=self.height,
            fx=P_0 * self.width / 2,
            fy=P_1 * self.height / 2,
            cx=(-P_2 * self.width + self.width) / 2,
            cy=(P_3 * self.height + self.height) / 2,
        )
        print(f"相机内参:\n {intrinsic.intrinsic_matrix}\n,type: {type(intrinsic.intrinsic_matrix)}")
        # intrinsic_matrix = intrinsic.intrinsic_matrix
        # print(f"相机内参矩阵:\n {intrinsic_matrix}\n,type: {type(intrinsic_matrix)}")

        return intrinsic
        # return intrinsic_matrix

    def calculate_extrinsic(self, vis):
        # b3Vector3 eye = b3MakeVector3(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
        # b3Vector3 center = b3MakeVector3(cameraTargetPosition[0], cameraTargetPosition[1], cameraTargetPosition[2]);
        # b3Vector3 up = b3MakeVector3(cameraUp[0], cameraUp[1], cameraUp[2]);
        # b3Vector3 f = (center - eye).normalized();
        # b3Vector3 u = up.normalized();
        # b3Vector3 s = (f.cross(u)).normalized();
        # u = s.cross(f);

        # viewMatrix[0 * 4 + 0] = s.x;
        # viewMatrix[1 * 4 + 0] = s.y;
        # viewMatrix[2 * 4 + 0] = s.z;

        # viewMatrix[0 * 4 + 1] = u.x;
        # viewMatrix[1 * 4 + 1] = u.y;
        # viewMatrix[2 * 4 + 1] = u.z;

        # viewMatrix[0 * 4 + 2] = -f.x;
        # viewMatrix[1 * 4 + 2] = -f.y;
        # viewMatrix[2 * 4 + 2] = -f.z;

        # viewMatrix[0 * 4 + 3] = 0.f;
        # viewMatrix[1 * 4 + 3] = 0.f;
        # viewMatrix[2 * 4 + 3] = 0.f;

        # viewMatrix[3 * 4 + 0] = -s.dot(eye);
        # viewMatrix[3 * 4 + 1] = -u.dot(eye);
        # viewMatrix[3 * 4 + 2] = f.dot(eye);
        # viewMatrix[3 * 4 + 3] = 1.f;

        cam_tar_np = np.array(self.cam_tar)
        cam_pos_np = np.array(self.cam_pos)
        front = cam_tar_np - cam_pos_np
        front = front / np.linalg.norm(front)
        print(f"front: {front}")
        # z = front @ cam_pos_np
        # print(f"z: {z}")

        
        ctrl = vis.get_view_control()
        # front = [0, 0, -1]  # 相机朝向 - 看向z轴负方向
        ctrl.set_front(front)
        ctrl.set_lookat(self.cam_tar)
        ctrl.set_up(self.cam_up_vector)      
        view_param = ctrl.convert_to_pinhole_camera_parameters()
        extrinsic = view_param.extrinsic
        print(f"extrinsic:\n {extrinsic}\n,type: {type(extrinsic)}")

        # pybullet
        # view_matrix_bullet = p.computeViewMatrix(cam_pos, cam_tar, cam_up_vector)
        # view_matrix_bullet = np.array(view_matrix_bullet).reshape(4, 4).T
        # view_matrix_bullet  = np.linalg.inv(view_matrix_bullet)
        # convert = np.array([
        #     [-1, 0, 0, 0],
        #     [0, -1, 0, 0], 
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]
        # ])
        # view_matrix_bullet = convert @ view_matrix_bullet
        # print(f"view_matrix_bullet:\n {view_matrix_bullet}\n,type: {type(view_matrix_bullet)}")


        # view_param.extrinsic =view_matrix_bullet
        # print(f"view_param.extrinsic:\n {view_param.extrinsic}\n,type: {type(view_param.extrinsic)}")
        # extrinsic = view_param.extrinsic

        return extrinsic, ctrl

    def update_pose(self, cam_pos=None, cam_tar=None, cam_up_vector=None):
        if cam_pos is not None:
            self.cam_pos = cam_pos
        if cam_tar is not None:
            self.cam_tar = cam_tar
        if cam_up_vector is not None:
            self.cam_up_vector = cam_up_vector
        
        # TODO: 添加返回
        return np.eye(4) 
    
    def save(self, pcd, timestamp=True):

        if self.mesh_path :
            base_name = os.path.splitext(input_file)[0]
            
            if timestamp:
                time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"{base_name}_{time_str}_pcd.ply"
            else:
                output_file = f"{base_name}_pcd.ply"
                
            o3d.io.write_point_cloud(output_file, pcd)
            print(f"success for saving pcd '{output_file}'")
        

    def shot(self):
        if not self.use_env:
            mesh = o3d.io.read_triangle_mesh(self.mesh_path)
            mesh.compute_vertex_normals()
            print("use_mesh_path")
        else:
            mesh = self.env
            print("use_env")


        
        # get pcd 
        # camera_distanc
        mesh_size = np.max(mesh.get_max_bound() - mesh.get_min_bound())
        camera_distance = mesh_size * 2
        # 设置渲染环境
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.width, height=self.height, visible=True)
        vis.add_geometry(mesh)
        
        # 获取投影矩阵
        proj_matrix = self.projection_matrix()
        
        # 计算相机内参
        intrinsic_matrix = self.calculate_intrinsic(proj_matrix)
        
        # 计算相机外参 (ctrl 传回才能正常拍照，否则貌似使用默认相机)
        extrinsic, ctrl = self.calculate_extrinsic(vis)

         # 渲染 捕获 rgbd
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(True)
        color = vis.capture_screen_float_buffer(True)
        
        time.sleep(0.5)

        depth_o3d = o3d.geometry.Image(np.asarray(depth))
        color_o3d = o3d.geometry.Image((np.asarray(color) * 255).astype(np.uint8))

        # filename_lower = os.path.basename(mesh_path).lower()
        # if any(keyword in filename_lower for keyword in ["bunny","armadillo","happy"]):
        #     depth_scale = 1.0 # m
        #     print("使用depth_scale=1.0")
        # elif "sds"  in filename_lower:
        #     depth_scale = 1000.0   # mm
        #     print("使用depth_scale=1000.0")
        # else:
        #     depth_scale = 1.0
        #     print("使用depth_scale=1000.0")

        depth_trunc = camera_distance * 3
        
        if self.mode == "color":
            print("color")
           
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0, # m
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_matrix)

        if self.mode == "depth":
            print("depth")
            
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_o3d,
                intrinsic_matrix,
                depth_scale=1.0,
                depth_trunc = camera_distance * 3
            )       

        # pcd_before = copy.deepcopy(pcd)  
        # print(intrinsic)

        # print(extrinsic)
        # exit()
        pcd.transform(np.linalg.inv(extrinsic))
        
        # pcd.transform(np.linalg.inv(extrinsic))
        # pcd.transform(extrinsic)
        # extrinsic = np.eye(4)
        # extrinsic[0:3, 3] = mesh_center
        # pcd.transform(np.linalg.inv(extrinsic))
        
        # 计算点云尺寸进行比较
        pcd_size = np.max(pcd.get_max_bound() - pcd.get_min_bound())
        print(f"点云尺寸: {pcd_size}")
        print(f"尺寸比例 (点云/网格): {pcd_size/mesh_size:.4f}")

        self.save(pcd)
        # self.save(copy.deepcopy(pcd))
        
        vis.destroy_window()

          
        # return pcd_before , pcd
        # return pcd   
    

        

        
if __name__ == "__main__":
    """ test class Camera """
    # input_file = r"E:\BENBV\dataset_generation\open3d\stanford-bunny.obj"

    # # center - eye(cam_pos) = front >> eye(cam_pos) = center - front
    # cam_pos= np.array([-0.02666264,  0.0949021 ,  0.00899104]) -  np.array([0, 0, -1])
    # # print(cam_pos)
    
    # camera = Camera(
    #     mesh_path=input_file,
    #     cam_pos=cam_pos,
    #     cam_tar=[-0.02666264, 0.0949021, 0.00899104],
    #     cam_up_vector=[0, -1, 0],
    #     near=0.1,
    #     far=10,
    #     size=(640, 480),
    #     fov=60,
    #     mode="depth",
    #     use_env=False,
    # )
    # camera.shot()

    """ test class Camera with class NBVScanEnv """
    input_file = r"E:\BENBV\dataset_generation\open3d\stanford-bunny.obj"

    env = NBVScanEnv()
    env.load_target_model(input_file)
    # env.visualize_model()
    input_env = env.mesh_data

    # center - eye(cam_pos) = front >> eye(cam_pos) = center - front
    cam_pos= np.array([-0.02666264,  0.0949021 ,  0.00899104]) -  np.array([0, 0, -1])
    # print(cam_pos)
    
    camera = Camera(
        mesh_path=input_file,
        env=input_env,
        cam_pos=cam_pos,
        cam_tar=[-0.02666264, 0.0949021, 0.00899104],
        cam_up_vector=[0, -1, 0],
        near=0.1,
        far=10,
        size=(640, 480),
        fov=60,
        mode="depth",
        use_env=True,
    )

    camera.shot()

