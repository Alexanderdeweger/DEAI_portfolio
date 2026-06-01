"""
ThijsDQN.py  -  Train an agent using Deep Q-Learning (DQN).

The agent plays the game, receives rewards, and improves over time.
No human gameplay data needed.

Usage:
    python ThijsDQN.py                        # train and save model
    python ThijsDQN.py --episodes 500         # custom episode count
    python ThijsDQN.py --watch                # watch the agent play after training
    python ThijsDQN.py --model model_dqn.pt --watch  # watch a pre-trained model

How DQN works (brief):
    - Agent observes game state (player pos + stone positions)
    - Agent picks an action (left / stay / right)
    - Game returns a reward (+1 survived, -10 hit)
    - Agent stores (state, action, reward, next_state) in replay memory
    - Every step, agent samples a random batch and trains the network
    - Over time the network learns which actions lead to more reward
"""

import os
import sys
import random
import argparse
import collections
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── headless pygame before import ────────────────────────────────────────────
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import pygame

try:
    from game import Game, Player, WIDTH, HEIGHT, MAX_STONES
except ImportError:
    print("ERROR: game.py not found in the same directory.")
    raise SystemExit(1)

# ── constants ─────────────────────────────────────────────────────────────────
STATE_DIM    = 1 + MAX_STONES * 3   # 43
NUM_ACTIONS  = 3                     # 0=left, 1=stay, 2=right

# DQN hyperparameters
MEMORY_SIZE  = 10_000   # how many transitions to remember
BATCH_SIZE   = 64       # how many to sample per training step
GAMMA        = 0.99     # discount factor (how much future rewards matter)
LR           = 1e-3     # learning rate
EPS_START    = 1.0      # starting exploration rate (100% random)
EPS_END      = 0.05     # minimum exploration rate (5% random)
EPS_DECAY    = 0.995    # how fast exploration decreases per episode
TARGET_UPDATE = 10      # update target network every N episodes
MAX_STEPS    = 2000     # max steps per episode

# ── neural network ────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Maps game state -> Q-value for each action.
    Q-value = expected total future reward for taking that action.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)

# ── replay memory ─────────────────────────────────────────────────────────────

class ReplayMemory:
    """
    Stores past experiences. We sample random batches to break correlations
    between consecutive steps, which stabilises training.
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,                dtype=torch.long),
            torch.tensor(rewards,                dtype=torch.float32),
            torch.tensor(np.array(next_states),  dtype=torch.float32),
            torch.tensor(dones,                  dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

# ── game state extraction ─────────────────────────────────────────────────────

def get_state(game):
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    vec[0] = game.player.rect.x / WIDTH
    for i, s in enumerate(game.stones[:MAX_STONES]):
        base = 1 + i * 3
        vec[base]     = s.rect.x / WIDTH
        vec[base + 1] = s.rect.y / HEIGHT
        vec[base + 2] = 1.0
    return vec

# ── DQN game environment ──────────────────────────────────────────────────────

class DQNGame(Game):
    """
    Wraps Game with a step() method for the DQN agent.
    Returns (next_state, reward, done).
    """
    def __init__(self):
        super().__init__(training=True)
        self.prev_lives = self.lives
        self.prev_points = self.points

    def reset(self):
        self.stones.clear()
        self.player = Player()
        self.points = 0
        self.lives  = 3
        self.spawn_interval = 1.0
        self.spawn_timer    = 0.0
        self.prev_lives     = self.lives
        pygame.event.pump()
        return get_state(self)

    def step(self, action):
        keys = {
            pygame.K_LEFT:  action == 0,
            pygame.K_RIGHT: action == 2,
        }

        dt = 1 / 60.0
        lives_before  = self.lives
        points_before = self.points

        self.update(dt, keys)
        pygame.event.pump()

        next_state = get_state(self)

        reward = 0.1
        done   = False

        if self.points > points_before:
            reward += 1.0
        if self.lives < lives_before:
            reward -= 10.0
            done = True  # end episode on hit

        return next_state, reward, done

# ── training loop ─────────────────────────────────────────────────────────────

def train(episodes, out_path, watch_after):
    env       = DQNGame()
    policy_net = QNetwork()   # network being trained
    target_net = QNetwork()   # stable copy used for Q-targets
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory    = ReplayMemory(MEMORY_SIZE)
    epsilon   = EPS_START

    print(f"Training for {episodes} episodes...")
    print(f"State dim: {STATE_DIM}  |  Actions: left / stay / right")
    print()

    best_reward = -float('inf')

    for ep in range(1, episodes + 1):
        state      = env.reset()
        total_reward = 0.0

        for step in range(MAX_STEPS):
            # ε-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)  # explore
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(state).unsqueeze(0))
                action = q_vals.argmax(1).item()              # exploit

            next_state, reward, done = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state         = next_state
            total_reward += reward

            # train when we have enough memory
            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                # current Q values
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # target Q values (Bellman equation)
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0]
                    q_targets  = rewards + GAMMA * max_next_q * (1 - dones)

                loss = nn.MSELoss()(q_values, q_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        # decay exploration
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # update target network periodically
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # save best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy_net.state_dict(), out_path)

        if ep % 10 == 0 or ep == 1:
            print(f"  Episode {ep:4d}/{episodes}  reward={total_reward:7.1f}  eps={epsilon:.3f}  best={best_reward:.1f}")

    print(f"\nTraining complete. Best model saved to {out_path}")

# ── watch agent play ──────────────────────────────────────────────────────────

def watch(model_path):
    # re-enable display
    if 'SDL_VIDEODRIVER' in os.environ:
        del os.environ['SDL_VIDEODRIVER']

    pygame.quit()
    pygame.init()

    from game import Game
    import pygame as pg

    class WatchGame(Game):
        def __init__(self, agent_net):
            super().__init__(training=False)
            self.agent_net = agent_net

        def run(self):
            while self.running:
                dt_ms = self.clock.tick(60)
                dt    = dt_ms / 1000.0

                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        self.running = False
                    elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                        self.running = False

                state = get_state(self)
                with torch.no_grad():
                    q_vals = self.agent_net(torch.tensor(state).unsqueeze(0))
                action = q_vals.argmax(1).item()

                keys = {pg.K_LEFT: action == 0, pg.K_RIGHT: action == 2}
                self.update(dt, keys)
                self.draw()

            pg.quit()

    net = QNetwork()
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()

    game = WatchGame(net)
    game.run()

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300,       help='Number of training episodes')
    parser.add_argument('--out',      default='model_dqn.pt',      help='Where to save the model')
    parser.add_argument('--model',    default='model_dqn.pt',      help='Model to load for --watch')
    parser.add_argument('--watch',    action='store_true',          help='Watch agent play (skips training if --model exists)')
    args = parser.parse_args()

    train(args.episodes, args.out, args.watch)
    if args.watch:
        watch(args.out)