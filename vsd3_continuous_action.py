# Reference: https://arxiv.org/pdf/2010.09177.pdf
# Implementation: https://github.com/ling-pan/SD3

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    # Common arguments
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="HopperBulletEnv-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
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

    # Algorithm specific arguments
    parser.add_argument("--asyncvec", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to use the async vectorized environments")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--beta", type=float, default=0.05,
        help="softmax coefficient (default: 0.05)")
    parser.add_argument("--noise-samples", type=float, default=50,
        help="number of samples to estimate softmax of Q function (default: 50)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--policy-noise", type=float, default=0.2,
        help="the scale of policy noise")
    parser.add_argument("--exploration-noise", type=float, default=0.2,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
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
        if len(x.shape) == 3:
            x = torch.cat([x, a], 2)
        else:
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
    actor1 = Actor(envs).to(device)
    actor2 = Actor(envs).to(device)
    actor1_target = Actor(envs).to(device)
    actor2_target = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    actor1_target.load_state_dict(actor1.state_dict())
    actor2_target.load_state_dict(actor2.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q1_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    q2_optimizer = optim.Adam(list(qf2.parameters()), lr=args.learning_rate)
    actor1_optimizer = optim.Adam(list(actor1.parameters()), lr=args.learning_rate)
    actor2_optimizer = optim.Adam(list(actor2.parameters()), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    num_gradient_updates = 0

    for update in range(1, num_updates + 1):
        # ROLLOUTS
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action1 = actor1.forward(next_obs)
                action2 = actor2.forward(next_obs)

                q1 = qf1.forward(next_obs, action1)
                q2 = qf2.forward(next_obs, action2)

                action = torch.Tensor(
                    [action1[i].cpu().numpy() if q1[i] >= q2[i] else action2[i].cpu().numpy() for i in range(len(q1))]
                ).to(device)

                clipped_noise = (
                    (torch.randn_like(action) * args.exploration_noise).clamp(-args.noise_clip, args.noise_clip).to(device)
                )
                action = (action + clipped_noise).clamp(-max_action, max_action)
                
                action = action
            actions[step] = action

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(done).to(device),
            )

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # TRAINING
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_rewards = rewards.reshape((-1,))
        b_dones = dones.reshape((-1,))

        # next_obs index manipulation
        b_next_obs = torch.zeros_like(obs).to(device)
        b_next_obs[:-1] = obs[1:]
        b_next_obs[-1] = next_obs
        b_next_obs = b_next_obs.reshape((-1,) + envs.single_observation_space.shape)
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                num_gradient_updates += 1
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                for network in range(2):
                    if network == 0:
                        actor = actor1
                        target_actor = actor1_target
                        critic = qf1
                        target_critic = qf1_target
                        actor_opt = actor1_optimizer
                        q_opt = q1_optimizer
                        debug_value = "debug/q1_values"
                        debug_loss = "losses/qf1_loss"
                        debug_actor = "losses/actor1_loss"
                    else:
                        actor = actor2
                        target_actor = actor2_target
                        critic = qf2
                        target_critic = qf2_target
                        actor_opt = actor2_optimizer
                        q_opt = q2_optimizer
                        debug_value = "debug/q2_values"
                        debug_loss = "losses/qf2_loss"
                        debug_actor = "losses/actor2_loss"

                    with torch.no_grad():
                        clipped_noise = (
                            (
                                torch.randn(
                                    (b_actions[mb_inds].shape[0], args.noise_samples, b_actions[mb_inds].shape[1]),
                                    dtype=b_actions[mb_inds].dtype,
                                    layout=b_actions[mb_inds].layout,
                                    device=b_actions[mb_inds].device,
                                )
                                * args.policy_noise
                            )
                            .clamp(-args.noise_clip, args.noise_clip)
                            .to(device)
                        )
                        next_state_actions = target_actor.forward(b_next_obs[mb_inds])

                        next_state_actions = next_state_actions.unsqueeze(1)

                        next_state_actions = (next_state_actions + clipped_noise).clamp(-max_action, max_action)

                        next_states = b_next_obs[mb_inds].unsqueeze(1).repeat((1, args.noise_samples, 1))

                        next_q1 = qf1_target.forward(next_states, next_state_actions)
                        next_q2 = qf2_target.forward(next_states, next_state_actions)

                        next_q = torch.min(next_q1, next_q2).squeeze(2)
                        max_q = torch.max(next_q, 1, keepdim=True).values
                        norm_q = next_q - max_q
                        e_beta_norm_q = torch.exp(args.beta * norm_q)
                        e_times_q = next_q * e_beta_norm_q

                        sum_e_times_q = torch.sum(e_times_q, 1)
                        sum_e_beta_norm_q = torch.sum(e_beta_norm_q, 1)
                        next_q = (sum_e_times_q / sum_e_beta_norm_q).unsqueeze(1)

                        target_q = b_rewards[mb_inds].unsqueeze(1) + (1 - b_dones[mb_inds].unsqueeze(1)) * args.gamma * next_q


                    q = critic.forward(b_obs[mb_inds], b_actions[mb_inds])

                    q_loss = loss_fn(q, target_q)

                    q_opt.zero_grad()
                    q_loss.backward()
                    nn.utils.clip_grad_norm_(list(critic.parameters()), args.max_grad_norm)
                    q_opt.step()

                    writer.add_scalar(debug_value, q.mean().item(), global_step)

                    writer.add_scalar(debug_loss, q_loss.item(), global_step)

                    actor_loss = -critic.forward(b_obs[mb_inds], actor.forward(b_obs[mb_inds])).mean()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(list(actor.parameters()), args.max_grad_norm)
                    actor_opt.step()

                    writer.add_scalar(debug_actor, actor_loss.item(), global_step)

                for param, target_param in zip(actor1.parameters(), actor1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(actor2.parameters(), actor2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
