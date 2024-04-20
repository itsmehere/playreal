from absl import app
import numpy as np
import torch
import time
import torch.jit
from tqdm import tqdm
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client

from motion_imitation.robots import a1_robot
from motion_imitation.robots import robot_config

from robot_interface import RobotInterface

BASE_FREQ = 100

# Config Values
dog_joint_stiffness = 45.0
dog_joint_damping = 0.7

tail_joint_stiffness = 0.6
tail_joint_damping = 0.1

num_tail_joints = 5

# Misc. Values
num_actions = 12 + num_tail_joints
num_envs = 1
max_episode_length = 5
obs_hist_len = 1

num_envs = 1
num_obs = 56

hist_len = 2
delay_len = 2


class State:
    def __init__(self):
        self.obs_buf = torch.zeros((1, 112))
        self.actions = torch.zeros((1, 12 + num_tail_joints))
        self.torques = torch.zeros((1, 12 + num_tail_joints))
        
        self.default_dof_pos = torch.tensor([0.05, 0.8, -1.4, -0.05, 0.8, -1.4, 0.05, 0.8, -1.4, -0.05, 0.8, -1.4, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.dof_pos = torch.zeros((1, 12 + num_tail_joints))
        self.dof_vel = torch.zeros((1, 12 + num_tail_joints))
        self.hist_obs = [torch.zeros((num_envs, num_obs // obs_hist_len), dtype=torch.float, requires_grad=False) for _ in range(hist_len + delay_len)]

        self.obs_scale_dof_pos = 1.0
        self.obs_scale_dof_vel = 0.05

        # Models
        with torch.no_grad():
            self.estimator = torch.jit.load('rma_estimator.pt')
            self.actor = torch.jit.load('rma_actor.pt')
            
        # Create Robot Instance
        p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = a1_robot.A1Robot(pybullet_client=p, action_repeat=1)
        
        # Torque Calculation
        self.clip_obs = 10.0
        self.clip_actions = 100.0
        
        self.action_scale = torch.tensor([0.5] * (12 + num_tail_joints))
        self.p_gains = torch.tensor([dog_joint_stiffness] * (12 + num_tail_joints))
        self.p_gains[-num_tail_joints:] = tail_joint_stiffness
        self.d_gains = torch.tensor([dog_joint_damping] * (12 + num_tail_joints))
        self.d_gains[-num_tail_joints:] = tail_joint_damping
        self.torque_limits = torch.tensor([23.7] * (12 + num_tail_joints))
        self.torque_limits[-num_tail_joints:] = 0.0

# Create robot state
state = State()

def main(_):
    # Estimator setup
    h = torch.zeros((1, num_envs, 256), dtype=torch.float, requires_grad=True)  # estimator num layers, ppo num envs, estimator hidden size
    c = torch.zeros((1, num_envs, 256), dtype=torch.float, requires_grad=True)  # estimator num layers, ppo num envs, estimator hidden size
    
    hidden_state = (h, c)
    mask = torch.ones((num_envs,))
    obs_hist = [torch.zeros_like(state.obs_buf) for _ in range(obs_hist_len)]

    # Play on real
    for i in range(max_episode_length):
        hidden_state = (torch.einsum("ijk,j->ijk", hidden_state[0], mask),
                        torch.einsum("ijk,j->ijk", hidden_state[1], mask))
        
        obs_hist.pop(0)
        obs_hist.append(state.obs_buf)
        estimator_obs = torch.cat(obs_hist, dim=-1).float()
        
        with torch.no_grad():
            zhat, hidden_state = estimate_latent(estimator_obs.unsqueeze(0), hidden_state)
        
        new_actions = estimate_actor_inference(state.obs_buf, zhat[0].detach())
        
        # Step updates the following state variables
        # - state.actions
        # - state.torques
        # - state.obs_buf (next iter)
        step(new_actions)
        
        # Play on real
        state.robot.Step(state.torques.flatten()[:12].detach().numpy(), robot_config.MotorControlMode.POSITION)
        
        print(f"\nIter: {i}")
        print(state.torques.shape)
        print(state.torques)
        
        time.sleep(1 / BASE_FREQ)


def estimate_latent(obs, hidden_states=None):
    latent, h, c = state.estimator(obs, hidden_states[0], hidden_states[1])
    return latent, (h, c)


def estimate_actor_inference(obs, latent):
    new_obs = torch.cat([obs, latent], dim=-1).float()
    return state.actor(new_obs)


def step(new_actions):
    # COMPUTE ACTIONS
    state.actions = torch.clip(new_actions, -state.clip_actions, state.clip_actions)
    
    # COMPUTE TORQUES
    actions_scaled = state.actions * state.action_scale
    torques = state.p_gains * (actions_scaled + state.default_dof_pos - state.dof_pos) - state.d_gains * state.dof_vel
    state.torques = torch.clip(torques, -state.torque_limits, state.torque_limits)
        
    compute_obs_buf()
    
    state.obs_buf = torch.clip(state.obs_buf, -state.clip_obs, state.clip_obs)


def compute_obs_buf():
    # Compute observations
    state.robot.ReceiveObservation()
    
    roll, pitch, _ = state.robot.GetBaseRollPitchYaw()
    
    if roll > torch.pi:
        roll -= 2 * torch.pi
        
    if pitch > torch.pi:
        pitch -= 2 * torch.pi
    
    state.dof_pos = torch.cat((torch.tensor(state.robot.GetMotorAngles()), torch.zeros(5)), dim=0).unsqueeze(0)
    state.dof_vel = torch.cat((torch.tensor(state.robot.GetMotorVelocities()), torch.zeros(5)), dim=0).unsqueeze(0)
    
    commands = torch.tensor([[1.5, 0.0, -0.2, 0.0]])
    commands_scale = torch.tensor([2.0, 2.0, 0.25])

    obs = torch.cat((
                    torch.tensor([[roll]]),
                    torch.tensor([[pitch]]),
                    commands[:, :3] * commands_scale, 
                    (state.dof_pos - state.default_dof_pos) * state.obs_scale_dof_pos,
                    state.dof_vel * state.obs_scale_dof_vel,
                    state.actions
                    ), dim=-1)
    
    state.hist_obs.pop(0)
    state.hist_obs.append(obs)
    
    obs_buf_delayed = torch.stack(state.hist_obs[:hist_len], dim=1)
    obs_buf_nodelay = torch.stack(state.hist_obs[-hist_len:], dim=1)
    
    state.obs_buf = torch.cat([obs_buf_delayed[...,:2], obs_buf_nodelay[...,2:5], obs_buf_delayed[...,5:(5 + num_actions * 2)], obs_buf_nodelay[...,(5 + num_actions * 2):]], dim=-1).view(num_envs, -1)


if __name__ == '__main__':
    app.run(main)

