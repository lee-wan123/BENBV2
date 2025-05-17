import open3d as o3d
import numpy as np
import time
import os
from datetime import datetime
import copy


class MeshCamera:
    def __init__(self, cam_pos, cam_tar, cam_up_vector, near, far, size, fov):
        """
        Initialize the MeshCamera for capturing point clouds from meshes
        
        Args:
            cam_pos (list): Camera position [x, y, z]
            cam_tar (list): Camera target/look-at position [x, y, z]
            cam_up_vector (list): Camera up vector [x, y, z]
            near (float): Near clipping plane
            far (float): Far clipping plane
            size (tuple): Image size as (width, height)
            fov (float): Field of view in degrees
        """
        self.width, self.height = size
        self.aspect = self.width / self.height
        self.fov = fov
        self.near, self.far = near, far
        
        self.cam_pos = cam_pos
        self.cam_tar = cam_tar
        self.cam_up_vector = cam_up_vector
        
        # Calculate intrinsic parameters based on width, height, fov
        self.intrinsic = self.calculate_intrinsic()
    
    def calculate_intrinsic(self):
        """
        Calculate camera intrinsic parameters based on width, height and FOV
        
        Returns:
            o3d.camera.PinholeCameraIntrinsic: Camera intrinsic parameters
        """
        aspect = self.width / self.height   
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2)
        
        # proj_matrix components
        P_0 = f / aspect  # proj_matrix[0]
        P_1 = f           # proj_matrix[5]
        P_2 = 0           # proj_matrix[2] 
        P_3 = 0           # proj_matrix[6] 
        
        # Calculate intrinsic parameters
        fx = P_0 * self.width / 2
        fy = P_1 * self.height / 2
        cx = (-P_2 * self.width + self.width) / 2  
        cy = (P_3 * self.height + self.height) / 2  
        
        # Create intrinsic matrix
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, fx, fy, cx, cy)
        return intrinsic
    
    def update_pose(self, camera_pos, target_pos, cam_up_vector):
        """
        Update camera pose parameters
        
        Args:
            camera_pos (list): New camera position
            target_pos (list): New camera target position
            cam_up_vector (list): New camera up vector
            
        Returns:
            np.ndarray: The view matrix
        """
        self.cam_pos = camera_pos
        self.cam_tar = target_pos
        self.cam_up_vector = cam_up_vector
        
        # Compute a simplified view matrix for reference (not actually used)
        # In a real implementation, this would be a proper 4x4 view matrix
        view_matrix = np.eye(4)
        return view_matrix
    
    def shot(self):
        """
        Capture a point cloud from the current view (placeholder method)
        
        Returns:
            tuple: (rgbImage, depthImage, pointcloud, _) to match Camera.shot() interface
        """
        # This is a placeholder. In actual use, this would need a mesh to render
        return None, None, None, None
    
    def capture_pointcloud_from_mesh(self, mesh_path):
        """
        Capture a point cloud from a mesh file using the camera parameters
        
        Args:
            mesh_path (str): Path to the mesh file
            
        Returns:
            o3d.geometry.PointCloud: Generated point cloud
        """
        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        
        # Get mesh center and size for auto-positioning if needed
        mesh_center = mesh.get_center()
        mesh_size = np.max(mesh.get_max_bound() - mesh.get_min_bound())
        camera_distance = mesh_size * 2
        
        # Set up the visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.width, height=self.height, visible=True)
        vis.add_geometry(mesh)
        
        # Set up the view based on camera parameters
        ctrl = vis.get_view_control()
        
        # Calculate front vector from camera position to target
        front = np.array(self.cam_tar) - np.array(self.cam_pos)
        if np.linalg.norm(front) > 0:
            front = front / np.linalg.norm(front)
        else:
            front = [0, 0, -1]  # Default front if cam_pos == cam_tar
            
        # Set the view parameters
        ctrl.set_front(front.tolist())
        ctrl.set_lookat(self.cam_tar)
        ctrl.set_up(self.cam_up_vector)
        
        # Render and capture
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(True)
        color = vis.capture_screen_float_buffer(True)
        
        time.sleep(1)  # Allow rendering to complete
        
        # Convert to Open3D image formats
        depth_o3d = o3d.geometry.Image(np.asarray(depth))
        color_o3d = o3d.geometry.Image((np.asarray(color) * 255).astype(np.uint8))
        
        # Determine depth scale based on the mesh filename
        filename_lower = os.path.basename(mesh_path).lower()
        if any(keyword in filename_lower for keyword in ["bunny", "armadillo", "happy"]):
            depth_scale = 1.0
            print("Using depth_scale=1.0")
        elif "sds" in filename_lower:
            depth_scale = 1000.0
            print("Using depth_scale=1000.0")
        else:
            depth_scale = 1.0
            print("Using default depth_scale=1.0")
        
        # Create RGBD image from color and depth
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=camera_distance*3,
            convert_rgb_to_intensity=False
        )
        
        # Generate point cloud from RGBD image using intrinsic parameters
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
        
        # Get the extrinsic matrix from the view and transform the point cloud
        view_param = ctrl.convert_to_pinhole_camera_parameters()
        extrinsic = view_param.extrinsic
        pcd.transform(np.linalg.inv(extrinsic))
        
        # Clean up
        vis.destroy_window()
        
        return pcd
    
    def capture(self, mesh_path):
        """
        Alias for capture_pointcloud_from_mesh to provide a simpler interface
        
        Args:
            mesh_path (str): Path to the mesh file
            
        Returns:
            o3d.geometry.PointCloud: Generated point cloud
        """
        return self.capture_pointcloud_from_mesh(mesh_path)
    
    def save_pointcloud(self, pcd, output_path=None):
        """
        Save the point cloud to a PLY file
        
        Args:
            pcd (o3d.geometry.PointCloud): Point cloud to save
            output_path (str, optional): Path to save the point cloud
            
        Returns:
            str: Path to the saved file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pointcloud_{timestamp}.ply"
            
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Point cloud saved as '{output_path}'")
        
        return output_path


# Example usage
if __name__ == "__main__":
    file =r"E:\BENBV\dataset_generation\open3d\stanford-bunny.obj"
    input_file = file
    # main_pcd 获取点云 需要的参数
    # front = [0, 0, -1]  # 相机朝向 - 看向z轴负方向
    # ctrl.set_lookat(mesh_center) >> mesh_center = mesh.get_center() # Camera target
    # ctrl.set_up([0, -1, 0]) 
    # size >>  width=640, height=480
    # fov >> 60
    
    # Create camera instance with parameters
    camera = MeshCamera(
        cam_pos=[0, 0, 3],        # Camera position
        cam_tar=[0, 0, 0],        # Camera target
        cam_up_vector=[0, -1, 0],  # Camera up vector
        near=0.1,                 # Near clipping plane
        far=10.0,                 # Far clipping plane
        size=(640, 480),          # Image size
        fov=60                    # Field of view
    )
    
    # Capture point cloud from mesh
    pcd = camera.capture(input_file)
    
    # Save the point cloud
    base_name = os.path.splitext(input_file)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{base_name}_{timestamp}_pcd.ply"
    camera.save_pointcloud(pcd, output_file)
    
    # Visualize
    # o3d.visualization.draw_geometries([pcd])