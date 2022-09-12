from time import time
import mujoco
import numpy as np
from gym.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv


class HalfCheetahEnv_ChangeDynamics(HalfCheetahEnv):
    def __init__(
        self, 
        xml_file = "half_cheetah.xml", 
        forward_reward_weight = 1, 
        ctrl_cost_weight = 0.1, 
        reset_noise_scale = 0.1, 
        exclude_current_positions_from_observation = True
    ):
        super().__init__(xml_file, forward_reward_weight, ctrl_cost_weight, reset_noise_scale, exclude_current_positions_from_observation)

        self.original_body_mass = {}
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            self.original_body_mass[body_name] = getattr(self.model, 'body')(body_name).mass

        self.original_joint_damping = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            self.original_joint_damping[joint_name] = getattr(self.model, 'joint')(joint_name).damping


    def reset_model(self):
        for body_name, original_mass in self.original_body_mass.items():
            # self.model.body(body_name).mass = np.random.uniform(0.8, 1.2) * original_mass
            self.model.body(body_name).mass = 1.2 * original_mass

        for joint_name, original_damping in self.original_joint_damping.items():
            # self.model.joint(joint_name).damping = np.random.uniform(0.8, 1.2) * original_damping
            self.model.joint(joint_name).damping = 1.2 * original_damping

        return super().reset_model()


if __name__ == '__main__':
    import time
    from gym.wrappers.monitoring.video_recorder import VideoRecorder

    env = HalfCheetahEnv_ChangeDynamics()
    print(env.original_body_mass, env.original_joint_damping)

    # for _ in range(4):
    #     env.reset_model()
    #     print([env.model.body(body_name).mass for body_name, original_mass in env.original_body_mass.items()])

    done = False
    env.reset()
    video_recorder = VideoRecorder(env, './video.mp4', enabled = True)
    for i in range(500):
        _, _, done, _ = env.step(action = env.action_space.sample())
        # env.render()
        video_recorder.capture_frame()
        time.sleep(0.01)