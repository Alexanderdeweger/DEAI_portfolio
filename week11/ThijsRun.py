"""
run_agent.py  –  Load a trained model and let it play the game.

Usage:
    python run_agent.py                        # watch the agent play (visible window)
    python run_agent.py --train                # headless, logs to game_log.csv
    python run_agent.py --model my_model.pt    # custom model path
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import pygame

# ── re-use the same constants / classes from game.py ─────────────────────────
# (import them if game.py is in the same folder, otherwise we inline the needed bits)
try:
    from game import Game, Player, Stone, WIDTH, HEIGHT, PLAYER_W, PLAYER_H, STONE_W, STONE_H, MAX_STONES
except ImportError:
    print("ERROR: game.py not found in the same directory.")
    raise SystemExit(1)

# ── model (must match train_agent.py exactly) ─────────────────────────────────
STATE_DIM   = 1 + MAX_STONES * 3
NUM_ACTIONS = 3

class AgentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)

# ── state builder (must match train_agent.py exactly) ────────────────────────

def build_state(player_x, stones):
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    vec[0] = player_x / WIDTH
    for i, s in enumerate(stones[:MAX_STONES]):
        base = 1 + i * 3
        vec[base]     = s.rect.x / WIDTH
        vec[base + 1] = s.rect.y / HEIGHT
        vec[base + 2] = 1.0
    return vec

# ── agent wrapper ─────────────────────────────────────────────────────────────

class NeuralAgent:
    ACTION_LEFT  = 0
    ACTION_STAY  = 1
    ACTION_RIGHT = 2

    def __init__(self, model_path):
        self.model = AgentNet()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def choose_action(self, player_x, stones):
        state  = build_state(player_x, stones)
        tensor = torch.tensor(state).unsqueeze(0)   # (1, STATE_DIM)
        with torch.no_grad():
            logits = self.model(tensor)
        return logits.argmax(1).item()

# ── agent-controlled game loop ────────────────────────────────────────────────

class AgentGame(Game):
    def __init__(self, agent, training=False):
        super().__init__(training=training)
        self.agent = agent

    def run(self):
        while self.running:
            dt_ms = self.clock.tick(60)
            dt    = dt_ms / 1000.0 * 5

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False

            # let the neural network decide
            action = self.agent.choose_action(self.player.rect.x, self.stones)
            keys = {
                pygame.K_LEFT:  action == NeuralAgent.ACTION_LEFT,
                pygame.K_RIGHT: action == NeuralAgent.ACTION_RIGHT,
            }

            self.update(dt, keys)
            self.draw()

        pygame.quit()

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model.pt',  help='Path to trained model weights')
    parser.add_argument('--train', action='store_true', help='Run headless (no window)')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: model file '{args.model}' not found.")
        print("Train the model first:")
        print("  python train_agent.py")
        raise SystemExit(1)

    if args.train:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

    agent = NeuralAgent(args.model)
    game  = AgentGame(agent, training=args.train)
    game.run()