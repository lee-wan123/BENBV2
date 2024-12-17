import time
import numpy as np
import pybullet as p
import open3d as o3d
import pybullet_data

from pathlib import Path

from camera import Camera
from scipy.spatial.transform import Rotation as R
from common import unit_vector, angle_between, read_mesh_off, scale_mesh


class NBVScanEnv:

    SIMULATION_STEP_DELAY = 1 / 240.0

    def __init__(
        self,
        camera=None,
        vis=False,
        camera_model_path="./dataset_generation/urdf/small_sphere.urdf",
    ) -> None:
        self.vis = vis
        self.camera = camera

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setTimeStep(self.SIMULATION_STEP_DELAY)
        p.connect(p.SHARED_MEMORY)
        # p.setGravity(0, 0, -10)
        p.setGravity(0, 0, 0)

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1.0, 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi / 2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi / 2, np.pi / 2, np.pi / 2)
        # self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.planeID = p.loadURDF("cube.urdf", )
        # self.planeID = p.loadURDF("duck_vhacd.urdf")
        self.cameraID = p.loadURDF(camera_model_path, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]))

    def load_target_model(self, filename: str):
        try:
            self.model_ID, self.model_pts = self.add_3D_model(filename)
            if self.model_ID == -1 and self.model_pts == None:
                return False
            else:
                return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def remove_target_model(self):
        p.removeBody(self.model_ID)

    def compute_camera_rotation_matrix_ball(self, camera_pos, target_pos, theta, beta):
        theta, beta = np.deg2rad(theta), np.deg2rad(beta)

        to_camera_pos_vec = unit_vector(np.array(camera_pos) - np.array(target_pos))
        rot_axis_1 = np.cross(np.array([0, 0, 1]), to_camera_pos_vec)
        rot_axis_1 = unit_vector(rot_axis_1)
        r1 = R.from_rotvec((theta + np.pi) * rot_axis_1)
        quaternion = r1.as_quat()

        camera_vector_up = np.cross(to_camera_pos_vec, rot_axis_1)
        return quaternion, camera_vector_up

    def compute_camera_rotation_matrix_any(self, target_pos, camera_pos):
        try:
            camera_pos = np.array(camera_pos)
            target_pos = np.array(target_pos)

            # Compute the direction vector from target to camera
            to_camera_pos_vec = camera_pos - target_pos

            # Check if the vector is not zero
            if np.allclose(to_camera_pos_vec, 0):
                return np.array([0, 0, 0, 1]), np.array([0, 1, 0])

            to_camera_pos_vec = unit_vector(to_camera_pos_vec)

            # Check if the camera is directly above or below the target
            if np.allclose(np.abs(to_camera_pos_vec), [0, 0, 1]):
                # If camera is above target, use [1, 0, 0] as rot_axis
                # If camera is below target, use [-1, 0, 0] as rot_axis
                rot_axis = np.array([1, 0, 0]) if to_camera_pos_vec[2] > 0 else np.array([-1, 0, 0])
                up_vector = np.array([0, 1, 0]) if to_camera_pos_vec[2] > 0 else np.array([0, -1, 0])
            else:
                # Compute rotation axis
                rot_axis = np.cross([0, 0, 1], to_camera_pos_vec)
                rot_axis = unit_vector(rot_axis)

                # Compute up vector
                up_vector = np.cross(to_camera_pos_vec, rot_axis)

            # Compute rotation angle
            angle = angle_between([0, 0, 1], to_camera_pos_vec)

            # Create rotation object and get quaternion
            r = R.from_rotvec(angle * rot_axis)
            quaternion = r.as_quat()

            return quaternion, up_vector

        except Exception as e:
            print(f"Error: {e}")
            print(f"target_pos: {target_pos}, camera_pos: {camera_pos}")
            return np.array([0, 0, 0, 1]), np.array([0, 1, 0])

    def load_off_model(self, filename: str, points_max_length=100000):
        """
        Load the off model provided by the ModelNet
        """

        # The maximum size in Pybullet is 131072
        # define B3_MAX_NUM_VERTICES 131072
        # https://github.com/bulletphysics/bullet3/blob/e9c461b0ace140d5c73972760781d94b7b5eee53/examples/SharedMemory/SharedMemoryPublic.h#L1132C29-L1132C35

        vertices, faces = read_mesh_off(path=filename, scale=1.0)
        mesh_data = o3d.geometry.TriangleMesh()
        mesh_data.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_data.triangles = o3d.utility.Vector3iVector(faces)

        mesh_point_size = np.array(mesh_data.vertices).shape[0]
        if mesh_point_size > points_max_length:
            # mesh_data_list = split_mesh(mesh_data, mesh_point_size / (VERTICES_MAX / 3))
            return -1, None

        center = mesh_data.get_center()
        mesh_data, _ = scale_mesh(mesh_data=mesh_data, center=center)
        # o3d.io.write_triangle_mesh("./dataset_generation/output/entire.obj", mesh_data)

        shift = [0, 0, 0]
        meshScale = [1, 1, 1]
        indices = []
        for face in mesh_data.triangles:
            indices.extend(face)
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            vertices=np.array(mesh_data.vertices).tolist(),
            indices=indices,
            rgbaColor=[1.0, 1.0, 1.0, 1],
            specularColor=[0.0, 0.0, 0.0],
            visualFramePosition=shift,
            meshScale=meshScale,
        )

        model_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visualShapeId,
            basePosition=[0, 0, 0],
        )
        return model_id, mesh_data

    def load_obj_model(self, filename: str):
        """
        Load the obj model which is supported by PyBullet very well
        """
        mesh_data = o3d.io.read_triangle_mesh(filename)
        center = mesh_data.get_center()

        mesh_data, scaler = scale_mesh(mesh_data=mesh_data, center=center)
        # o3d.io.write_triangle_mesh("./dataset_generation/output/entire.obj", mesh_data)

        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=filename,
            rgbaColor=[1.0, 1.0, 1.0, 1],
            specularColor=[0.0, 0.0, 0.0],
            visualFramePosition=(-1 * center * scaler).tolist(),  # scale operation first and then move to the center, so scaler is needed
            meshScale=[scaler] * 3,
        )

        model_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visualShapeId,
            basePosition=[0.0, 0.0, 0.0],
        )

        return model_id, mesh_data

    def add_3D_model(self, filename: str, sample_number=60000):
        model_id, mesh_data = -1, None
        filename = Path(filename)

        if filename.suffix == ".obj":
            model_id, mesh_data = self.load_obj_model(str(filename))
            if model_id == -1:
                return -1, None

        elif filename.suffix == ".off":
            model_id, mesh_data = self.load_off_model(str(filename))
            if model_id == -1:
                return -1, None
        else:
            print("Not Support!")

        simple_suffix = f"simple{sample_number//1000}k.npy"

        sampled_filename = filename.parent / Path(f"{filename.stem}_{simple_suffix}")

        if sampled_filename.exists():
            model_pts = np.load(sampled_filename)
            print(f"{model_pts.shape} points loaded")
        else:
            print(f"The sampled file does not found, so run sample_points_poisson_disk ... ")
            pcd_sampled = mesh_data.sample_points_poisson_disk(number_of_points=sample_number)  # o3d.geometry.PointCloud() format
            model_pts = np.asarray(pcd_sampled.points, dtype=np.float64)
            print(f"{model_pts.shape} points loaded and sampled")
            np.save(sampled_filename, model_pts)
        return model_id, model_pts

    # def update_camera_ball(self, R, theta, beta, target_pos=[0, 0, 0]):
    #     """
    #     update the camera position and orientation given the R, theta and beta
    #     around a center point
    #     @theta: [0,90]
    #     @beta: [0,360]
    #     @return view_matrix: matrix from the camera to world frame
    #     """
    #     r_theta, r_beta = np.deg2rad(theta), np.deg2rad(beta)
    #     x = R * np.sin(r_theta) * np.cos(r_beta)
    #     y = R * np.sin(r_theta) * np.sin(r_beta)
    #     z = R * np.cos(r_theta)
    #     camera_pos = [x, y, z]
    #     qua_m, camera_vector_up = self.compute_camera_rotation_matrix_ball(camera_pos, target_pos, theta, beta)
    #     # camera_vector_up = [0, 0, 1.0]

    #     # p.addUserDebugLine(cameraPos, [0,0,0], [1, 0, 0], 1)
    #     p.addUserDebugLine(camera_pos, target_pos, lineColorRGB=[0, 0, 1], lifeTime=2)
    #     # p.addUserDebugLine(targetPos, [0,0,0], lineColorRGB=[1,0,0], lifeTime=0)

    #     p.resetBasePositionAndOrientation(self.cameraID, camera_pos, qua_m)
    #     return self.camera.update_pose(camera_pos, target_pos, camera_vector_up)

    def update_camera_any(self, target_pos, camera_pos):
        """
        update the camera position and orientation given any camera positions
        """
        qua_m, camera_vector_up = self.compute_camera_rotation_matrix_any(target_pos, camera_pos)
        # camera_vector_up = [0, 0, 1.0]
        p.addUserDebugLine(camera_pos, target_pos, lineColorRGB=[0, 0, 1], lifeTime=2)

        p.resetBasePositionAndOrientation(self.cameraID, camera_pos, qua_m)
        return self.camera.update_pose(camera_pos, target_pos, camera_vector_up)

    def step_simulation(self):
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)

    def read_debug_parameter(self):
        # read the value of task parameter
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)

        return x, y, z, roll, pitch, yaw

    def step(self):

        self.step_simulation()

        return self.get_observation()

    def get_observation(self):
        if isinstance(self.camera, Camera):
            obs = dict()
            rgb, depth, pointcloud, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, pc=pointcloud, seg=seg))
        else:
            assert self.camera is None
        return obs

    def reset(self):
        pass

    def close(self):
        p.disconnect(self.physicsClient)
