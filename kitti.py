import numpy as np, os.path as path, glob
from skimage import color
from skimage import io

class Util:
    @staticmethod
    def read_scales(pose_file_path):
        poses = Util.read_poses(pose_file_path)

        deltas = map(lambda prev_pose, pose: np.dot(np.linalg.inv(prev_pose), pose), poses[:-1], poses[1:])
        scales = map(lambda delta: np.linalg.norm(delta[:3,3]), deltas)
        return scales

    @staticmethod
    def read_poses(pose_file_path):
        poses = []
        with open(pose_file_path, 'r') as f:
            for line in f:
                pose = np.array([float(val) for val in line.split()]).reshape(3,4)
                pose = np.vstack((pose, [0, 0, 0, 1]))
                poses.append(pose)
        return poses

    @staticmethod
    def read_image(image_path):
        return color.rgb2gray(io.imread(image_path))
