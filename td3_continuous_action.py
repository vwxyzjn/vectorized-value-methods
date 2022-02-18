import argparse
import collections
import os
import random
import time
from distutils.util import strtobool
from typing import NamedTuple

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from celluloid import Camera
from matplotlib import pyplot as plt
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="Hopper-v2",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="vectorized-value-methods",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="vwxyzjn",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--visualize-timestep-distribution", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to visualize the distribution of timesteps of states")

    # Algorithm specific arguments
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--policy-noise", type=float, default=0.2,
        help="the scale of policy noise")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    args = parser.parse_args()
    # fmt: on
    return args


class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    timesteps: torch.Tensor


# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer:
    def __init__(self, buffer_limit, o, a, device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device

    def add(self, obs, real_next_obs, actions, rewards, dones, infos, timestep):
        transition = obs[0], real_next_obs[0], actions[0], rewards[0], dones[0], infos[0], timestep[0]
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, s_prime_lst, done_mask_lst, r_lst, t_lst = [], [], [], [], [], []

        for transition in mini_batch:
            s, s_prime, a, r, done_mask, _, timestep = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)
            t_lst.append(timestep)

        data = (
            torch.Tensor(np.array(s_lst)).to(self.device),
            torch.Tensor(np.array(a_lst)).to(self.device),
            torch.Tensor(np.array(s_prime_lst)).to(self.device),
            torch.Tensor(np.array(done_mask_lst)).to(self.device),
            torch.Tensor(np.array(r_lst)).to(self.device),
            torch.Tensor(np.array(t_lst)).to(self.device),
        )

        return ReplayBufferSamples(*data)


class TimestepStats(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.t += 1
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.t = 0
        return obs


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = TimestepStats(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x))


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, 0, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, device=device)
    loss_fn = nn.MSELoss()

    # Visualization:
    if args.visualize_timestep_distribution:
        fig, ax = plt.subplots()
        camera = Camera(fig)

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = envs.action_space.sample()
        else:
            actions = actor.forward(torch.Tensor(obs).to(device))
            actions = np.array(
                [
                    (
                        actions.tolist()[0]
                        + np.random.normal(0, max_action * args.exploration_noise, size=envs.action_space.shape[0])
                    ).clip(envs.single_action_space.low, envs.single_action_space.high)
                ]
            )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        timestep = torch.Tensor(np.array([env.t for env in envs.envs]))
        rb.add(obs, real_next_obs, actions, rewards, dones, infos, timestep)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            if args.visualize_timestep_distribution and global_step % 15 == 0:
                max_timestep = int(data.timesteps.max().long())
                histogram = data.timesteps.histc(bins=max_timestep, min=0, max=max_timestep)
                x = range(max_timestep)
                ax.bar(x, histogram.cpu())
                ax.set_xlabel("Visited timestep of the obs (higher the better)")
                ax.set_ylabel("Number of occurrences")
                ax.text(
                    0.2,
                    1.01,
                    f"global_step={global_step}, update={global_step - args.learning_starts}",
                    transform=ax.transAxes,
                )
                camera.snap()
            with torch.no_grad():
                clipped_noise = (torch.randn_like(torch.Tensor(actions[0])) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                )

                next_state_actions = (target_actor.forward(data.next_observations) + clipped_noise.to(device)).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target.forward(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target.forward(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1.forward(data.observations, data.actions).view(-1)
            qf2_a_values = qf2.forward(data.observations, data.actions).view(-1)
            qf1_loss = loss_fn(qf1_a_values, next_q_value)
            qf2_loss = loss_fn(qf2_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            qf2_loss.backward()
            nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), args.max_grad_norm)
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1.forward(data.observations, actor.forward(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()), args.max_grad_norm)
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
    if args.visualize_timestep_distribution:
        animation = camera.animate()
        animation.save(f"runs/{run_name}/animation.mp4")
        if args.track:
            wandb.log({"video.0": wandb.Video(f"runs/{run_name}/animation.mp4")})
    envs.close()
    writer.close()
