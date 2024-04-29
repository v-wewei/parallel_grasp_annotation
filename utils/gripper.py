import os
import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel


class CustomGripperBase(GripperModel):
    """
    6-DoF Robotiq gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        CUR_DIR = os.path.dirname(__file__)
        super().__init__(os.path.join(CUR_DIR, "../asset/parallel_gripper/parallel_jaw_gripper_zhiyuan.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.05])

    @property
    def _important_geoms(self):
        return {}


class CustomGripper(CustomGripperBase):
    """
    1-DoF variant of RobotiqGripperBase.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1