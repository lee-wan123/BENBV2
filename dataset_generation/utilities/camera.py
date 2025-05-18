import cv2
import pybullet as p
import numpy as np
import open3d as o3d
from env_3d import *


class Camera:
    def __init__(self, cam_pos, cam_tar, cam_up_vector, near, far, size, fov):
        self.width, self.height = size
        self.aspect = self.width / self.height
        self.fov = fov
        self.near, self.far = near, far

        self.view_matrix = p.computeViewMatrix(cam_pos, cam_tar, cam_up_vector)
        self.proj_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)

    def update_pose(self, camera_pos, target_pos, cam_up_vector):
        self.view_matrix = p.computeViewMatrix(camera_pos, target_pos, cam_up_vector)
        self.proj_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        return np.array(self.view_matrix).reshape(4, 4).T

    def rgb_2_Image(self, rgb):
        if isinstance(rgb, tuple):
            rgb = np.asarray(rgb)
        rgbImg = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgbImg

    def depth_2_Image(self, depth):
        if isinstance(depth, tuple):
            depth = np.asarray(depth).reshape(self.height, self.width).astype(np.float32)  # float
            # https://github.com/bulletphysics/bullet3/blob/e9c461b0ace140d5c73972760781d94b7b5eee53/examples/SharedMemory/SharedMemoryPublic.h#L460
        # depth_image = self.far * self.near / (self.far - (self.far - self.near) * depth)
        # Improve the depth conversion algorithm to handle edge cases more robustly.
        epsilon = 1e-6
        depth_image = self.far * self.near / (self.far - (self.far - self.near) * np.clip(depth, epsilon, 1 - epsilon))
        depth_image = (depth_image * 1000).astype(np.uint16)
        # depImg = Image.fromarray(depth_image) # save as png
        # depImg.save('./output/test.png')

        # depImg = (depth_image.astype(np.uint8)) # save as jpg
        # depImg = Image.fromarray(depImg)
        # depImg.save('./output/test.jpg')
        return depth_image

    def rgb_depth_2_Pointcloud(self, rgbImage, depth_image):
        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgbImage),
        # o3d.geometry.Image(depth_image),
        # convert_rgb_to_intensity=False)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            # o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )
        P_0, P_1 = self.proj_matrix[0], self.proj_matrix[5]
        P_2, P_3 = self.proj_matrix[2], self.proj_matrix[6]
        intrinsic.set_intrinsics(
            width=self.width,
            height=self.height,
            fx=P_0 * self.width / 2,
            fy=P_1 * self.height / 2,
            cx=(-P_2 * self.width + self.width) / 2,
            cy=(P_3 * self.height + self.height) / 2,
        )
        # cx = self.width / 2, cy = self.height / 2)
        # point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        # print(f"intrinsic_matrix:\n {intrinsic.intrinsic_matrix}\n,type: {type(intrinsic.intrinsic_matrix)}")
        # exit()

        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth_image),
            intrinsic,
            depth_scale=1000.0,
            depth_trunc=self.far - 0.1,
        )

        # o3d.visualization.draw_geometries([point_cloud])
        return point_cloud

    def shot(self):
        # Get depth values using the OpenGL renderer
        # TODO:
        # https://github.com/Stanford-TML/SpringGrasp_release/blob/4c644f669f61d635af5d25a1eea8994b62677b8a/tests/get_observable_pcd.py
        # !Attention: the z-buffer does not have enough accuracy
        _w, _h, rgb, depth, _ = p.getCameraImage(
            self.width,
            self.height,
            self.view_matrix,
            self.proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            # renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_NO_SEGMENTATION_MASK,
        )
        #    shadow = 0,
        #    lightAmbientCoeff = 1.0,
        #    lightColor = [1.0,1.0,1.0],
        #    lightDirection=[1, 1, 1])
        _rgbImage = None
        # _rgbImage = self.rgb_2_Image(rgb)
        _depthImage = self.depth_2_Image(depth)
        _pointcloud = self.rgb_depth_2_Pointcloud(_rgbImage, _depthImage)
        return _rgbImage, _depthImage, _pointcloud, _


if __name__ == "__main__":
    # test 
    camera = Camera(
        cam_pos=[0.3, 0, 0.3],
        cam_tar=[0, 0, 0],
        cam_up_vector=[0, 0, 1],
        near=0.01,
        far=1.0,
        size=(640, 480),
        fov=60,
    )
    print(f"projection_matrix:\n{camera.proj_matrix}")
    print(f"view_matrix:\n{camera.view_matrix}")

    rgb, depth, point_cloud, _ = camera.shot()
    # print("RGB Image:", rgb)
    # print("Depth Image:", depth)
    # print("Point Cloud:", point_cloud)