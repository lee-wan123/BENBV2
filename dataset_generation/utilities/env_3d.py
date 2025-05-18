import numpy as np
import open3d as o3d
from pathlib import Path
import sys
import os
from common import read_mesh_off, scale_mesh
from camera_3d import Camera
import cv2
from main_pcd import *

# sample_number >> 扫描点云的数量
# points_max_length >> 3D模型点的数量 


class NBVScanEnv:
    def __init__(self):
        self.model_mesh = None
        self.model_pts = None
        self.model_ID = None
        self.mesh_center = np.zeros(3)
        self.mesh_scale = 1.0

    def load_target_model(self, filename: str):
        try:
            self.model_ID, self.model_pts = self.add_3D_model(filename)
            if self.model_ID == -1 and self.model_pts is None:
                print(f"model_id: {self.model_ID}, model_pts: {self.model_pts.shape}")
                return False
            else:
                print(f"model_id: {self.model_ID}, model_pts: {self.model_pts.shape}")
                return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def load_off_model(self, filename: str, points_max_length=100000):
        vertices, faces = read_mesh_off(path=filename, scale=1.0)
        self.mesh_data = o3d.geometry.TriangleMesh()
        self.mesh_data.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh_data.triangles = o3d.utility.Vector3iVector(faces)
        self.mesh_data.compute_vertex_normals()
        
        mesh_point_size = np.array(self.mesh_data.vertices).shape[0]
        if mesh_point_size > points_max_length:
            print(f"Mesh has too many points: {mesh_point_size} > {points_max_length}")
            return -1, None
        
        # Get center and scale to normalize the model
        self.mesh_center = self.mesh_data.get_center()
        self.mesh_data, self.mesh_scale = scale_mesh(mesh_data=self.mesh_data, center=self.mesh_center)
        
        # Just store the mesh data, no PyBullet creation
        model_id = 1  # Dummy ID since we're not using PyBullet
        
        return model_id, self.mesh_data


    def load_obj_model(self, filename: str):

        # Load mesh using Open3D
        self.mesh_data = o3d.io.read_triangle_mesh(filename)
        self.mesh_data.compute_vertex_normals()
        
        # Get center and scale to normalize the model
        # self.mesh_center = self.mesh_data.get_center()
        # self.mesh_data, self.mesh_scale = scale_mesh(mesh_data=self.mesh_data, center=self.mesh_center)
        
        # Dummy ID 
        model_id = 1  

        mesh_data = self.mesh_data
            
        return model_id, mesh_data
    
    def add_3D_model(self, filename: str, sample_number=60000):
        model_id, self.mesh_data = -1, None
        filename = Path(filename)
        
        if filename.suffix == ".obj":
            model_id, self.mesh_data = self.load_obj_model(str(filename))
            if model_id == -1:
                return -1, None
        elif filename.suffix == ".off":
            model_id, self.mesh_data = self.load_off_model(str(filename))
            if model_id == -1:
                return -1, None
        else:
            print("Not Support!")
            
        if model_id == -1 or self.mesh_data is None:
            return -1, None
            
        # Store the loaded mesh for future use
        self.model_mesh = self.mesh_data
        
        # Check if sampled point cloud already exists
        simple_suffix = f"simple{sample_number//1000}k.npy"
        sampled_filename = filename.parent / Path(f"{filename.stem}_{simple_suffix}")
        
        if sampled_filename.exists():
            model_pts = np.load(sampled_filename)
            print(f"{model_pts.shape} points loaded from existing file")
        else:
            print(f"The sampled file not found, running sample_points_poisson_disk...")
            # Sample points using Poisson disk sampling
            self.mesh_data.compute_vertex_normals()  # Ensure normals are computed
            pcd_sampled = self.mesh_data.sample_points_poisson_disk(number_of_points=sample_number)
            model_pts = np.asarray(pcd_sampled.points, dtype=np.float64)
            print(f"{model_pts.shape} points sampled and loaded")
            np.save(sampled_filename, model_pts)
            
        return model_id, model_pts
    
    def visualize_model(self):
        if self.model_mesh is None or self.model_pts is None:
            print("No model loaded or no points sampled.")
            return
        
        pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(self.model_pts)
        pcd.points = o3d.utility.Vector3dVector(self.model_pts)
        # print(f"pcd.points: {np.asarray(pcd.points).shape}")
        # x- read y-green z-blue    
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])       
        print("Visualizing sampled points")
        o3d.visualization.draw_geometries([pcd, coordinate_frame])





if __name__ == "__main__":  
    """ test env_3d.py  by loading model """
    file =r"E:\BENBV\dataset_generation\open3d\stanford-bunny.obj"

    # # test camera
    # output_dir = "./output"
    # os.makedirs(output_dir, exist_ok=True)
    env = NBVScanEnv()
    env.load_target_model(file)
    env.visualize_model()
    print(env.model_mesh)


    # Define camera positions for scanning (similar to env.py test)
    # camera_positions = [
    #     [0.3, 0, 0.3],
    #     [0, 0.3, 0.3],
    #     [-0.3, 0, 0.3],
    #     [0, -0.3, 0.3]
    # ]
    # camera_positions = [
    #     [0.3, 0, 0.3]
    # ]



    # file = r"E:\BENBV\dataset_generation\open3d\train\chair_0008.off"
    # use load_target_model to load the model


    # # Camera setup parameters
    # camera_pos = [0.3, 0, 0.3]
    # target_pos = [0, 0, 0]
    # cam_up_vector = [0, 0, 1]
    # near = 0.01
    # far = 1.0
    # size = (640, 480)
    # fov = 60
     
    # # Initialize camera
    # camera = Camera(camera_pos, target_pos, cam_up_vector, near, far, size, fov)
    
    # # Create the environment
    # env = NBVScanEnv()
    
    # model_id, mesh_data =  env.load_target_model(file)

    # if not mesh_data :
    #     print("Failed to load model")

    

    # exit()

    
    # # Scan the model from each position
    # for i, pos in enumerate(camera_positions):
    #     print(f"Capturing from position {i+1}/{len(camera_positions)}: {pos}")
        
    #     # Update camera pose
    #     view_matrix = camera.update_pose(pos, target_pos, cam_up_vector)
        
    #     # Take a shot with the camera
    #     # Pass the model mesh to the camera for rendering
    #     rgb, depth, point_cloud, _ = camera.shot(mesh=env.model_mesh)
        
    #     # Save the depth image
    #     if depth is not None:
    #         depth_filename = os.path.join(output_dir, f"depth_{i+1}.png")
    #         cv2.imwrite(depth_filename, depth)
    #         print(f"Saved depth image to {depth_filename}")
        
    #     # Save the point cloud
    #     if point_cloud is not None:
    #         pc_filename = os.path.join(output_dir, f"pointcloud_{i+1}.ply")
    #         o3d.io.write_point_cloud(pc_filename, point_cloud)
    #         print(f"Saved point cloud to {pc_filename}")
            
    #         # Optionally, visualize the point cloud 
    #         if i == 0:  # Just for the first viewpoint to avoid too many visualization windows
    #             print(f"Visualizing point cloud from position {i+1}")
    #             coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #                 size=0.1, origin=[0, 0, 0]
    #             )
    #             o3d.visualization.draw_geometries([point_cloud, coordinate_frame])
    
    
    # print("Scanning complete!")