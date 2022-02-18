# Reference: https://arxiv.org/pdf/1509.02971.pdf
# https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
# https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py

import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool

import gym
import matplotlib.pyplot as plt
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from celluloid import Camera
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # Common arguments
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
    parser.add_argument("--total-timesteps", type=int, default=5000000,
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
    parser.add_argument("--asyncvec", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to use the async vectorized environments")
    parser.add_argument("--num-envs", type=int, default=32,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=9,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=1,
        help="the K epochs to update the policy")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--policy-noise", type=float, default=0.2,
        help="the scale of policy noise")
    parser.add_argument("--exploration-noise", type=float, default=0.2,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--start-e", type=float, default=1.0,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.8,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


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
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class QNetwork(nn.Module):
    def __init__(self, envs):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(
            np.array(envs.single_observation_space.shape).prod() + np.prod(envs.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, envs):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(envs.single_action_space.shape))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x))


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.tensorboard.patch(save=False)
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
    VecEnv = gym.vector.AsyncVectorEnv if args.asyncvec else gym.vector.SyncVectorEnv
    envs = VecEnv([make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # ALGO LOGIC: initialize agent here:
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
    loss_fn = nn.MSELoss()
    exploration_dist = torch.distributions.Normal(
        torch.zeros(envs.single_action_space.shape[0]).to(device),
        torch.zeros(envs.single_action_space.shape[0]).to(device) + args.exploration_noise,
    )
    cast_tensor = torch.ones((args.num_envs,) + envs.single_action_space.shape, device=device)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    timesteps = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=100)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    num_gradient_updates = 0

    # Visualization:
    if args.visualize_timestep_distribution:
        fig, ax = plt.subplots()
        camera = Camera(fig)

    for update in range(1, num_updates + 1):
        # ROLLOUTS
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        writer.add_scalar("charts/epsilon", epsilon, global_step)
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action = (
                    actor.forward(next_obs)
                    + torch.normal(
                        0,
                        max_action * args.exploration_noise,
                        size=(envs.num_envs, envs.single_action_space.shape[0]),
                        device=device,
                    )
                ).clamp(-max_action, max_action)
                random_action = torch.Tensor(envs.action_space.sample()).to(device)
                random_action_flag = (torch.rand((envs.num_envs, 1), device=device) > epsilon) * cast_tensor
                action = torch.where(random_action_flag.bool(), action, random_action)
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(done).to(device),
            )
            timesteps[step] = torch.Tensor(np.array([env.t for env in envs.envs]))

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    avg_returns.append(item["episode"]["r"])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    break

        if args.visualize_timestep_distribution:
            max_timestep = int(timesteps.max().long())
            histogram = timesteps.histc(bins=max_timestep, min=0, max=max_timestep)
            x = range(max_timestep)
            ax.bar(x, histogram.cpu())
            ax.set_xlabel("Visited timestep of the obs (higher the better)")
            ax.set_ylabel("Number of occurrences")
            ax.text(0.2, 1.01, f"global_step={global_step}, update={update}", transform=ax.transAxes)
            camera.snap()

        if global_step > args.learning_starts:
            # bootstrap value if not done
            with torch.no_grad():
                clipped_noise = (
                    (torch.randn_like(action) * args.policy_noise).clamp(-args.noise_clip, args.noise_clip).to(device)
                )
                next_state_actions = (target_actor.forward(next_obs) + clipped_noise).clamp(-max_action, max_action)
                qf1_next_target = qf1_target.forward(next_obs, next_state_actions)
                qf2_next_target = qf2_target.forward(next_obs, next_state_actions)
                next_value = torch.min(qf1_next_target, qf2_next_target).flatten()
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                # advantages = returns - values

            # TRAINING
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_rewards = rewards.reshape((-1,))
            b_dones = dones.reshape((-1,))
            b_returns = returns.reshape(-1)

            b_inds = np.arange(args.batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    num_gradient_updates += 1
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    qf1_a_values = qf1.forward(b_obs[mb_inds], b_actions[mb_inds]).view(-1)
                    qf2_a_values = qf2.forward(b_obs[mb_inds], b_actions[mb_inds]).view(-1)
                    qf1_loss = loss_fn(qf1_a_values, b_returns[mb_inds])
                    qf2_loss = loss_fn(qf2_a_values, b_returns[mb_inds])

                    writer.add_scalar("debug/q1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("debug/q2_values", qf2_a_values.mean().item(), global_step)

                    # optimize the model
                    q_optimizer.zero_grad()
                    (qf1_loss + qf2_loss).backward()
                    nn.utils.clip_grad_norm_(list(qf1.parameters()) + list(qf2.parameters()), args.max_grad_norm)
                    q_optimizer.step()
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)

                    if num_gradient_updates % args.policy_frequency == 0:
                        actor_loss = -qf1.forward(b_obs[mb_inds], actor.forward(b_obs[mb_inds])).mean()
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(list(actor.parameters()), args.max_grad_norm)
                        actor_optimizer.step()

                        writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                        # update the target network
                        for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.visualize_timestep_distribution:
        animation = camera.animate()
        animation.save(f"runs/{run_name}/animation.mp4")
        if args.track:
            wandb.log({"video.0": wandb.Video(f"runs/{run_name}/animation.mp4")})

    envs.close()
    writer.close()
