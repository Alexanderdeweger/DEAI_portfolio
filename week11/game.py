import os
import sys
import csv
import random
import time
import argparse

# If running in training mode we avoid opening a window
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    if '--train' in sys.argv:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame

# Constants
WIDTH, HEIGHT = 440, 440
PLAYER_W, PLAYER_H = 20, 40
STONE_W, STONE_H = 30, 30
MAX_STONES = 24

LOG_INTERVAL = 0.01  # seconds (10 ms)


class Player:
    def __init__(self):
        self.rect = pygame.Rect((WIDTH - PLAYER_W)//2, HEIGHT - PLAYER_H - 5, PLAYER_W, PLAYER_H)
        self.speed = 5

    def move_left(self):
        self.rect.x = max(0, self.rect.x - self.speed)

    def move_right(self):
        self.rect.x = min(WIDTH - PLAYER_W, self.rect.x + self.speed)


class Stone:
    def __init__(self, x, y, speed):
        self.rect = pygame.Rect(x, y, STONE_W, STONE_H)
        self.speed = speed

    def update(self, dt):
        self.rect.y += self.speed * dt


class Game:
    def __init__(self, training=False):
        pygame.init()
        self.training = training
        if not training:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption('Thijs Bullet Hell')
        else:
            # create a surface for logic even if not displayed
            self.screen = pygame.Surface((WIDTH, HEIGHT))

        self.clock = pygame.time.Clock()
        self.player = Player()
        self.stones = []
        self.points = 0
        self.lives = 3
        self.spawn_timer = 0.0
        self.spawn_interval = 1.0
        self.stone_speed = 150.0  # pixels per second
        self.running = True

        # Logging
        self.log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'game_log.csv')
        self.last_log = time.time()
        self.ensure_log_header()

    def ensure_log_header(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'player_x', 'points', 'lives', 'stones'])

    def spawn_stone(self):
        if len(self.stones) >= MAX_STONES:
            return
        x = random.randint(0, WIDTH - STONE_W)
        speed = self.stone_speed * (1 + random.random() * 0.5)
        self.stones.append(Stone(x, -STONE_H, speed))

    def reset_round(self):
        self.stones.clear()
        self.player = Player()
        self.points = 0
        self.lives = 3
        self.spawn_interval = 1.0
        self.stone_speed = 150.0

    def log_state(self):
        now = time.time()
        if now - self.last_log < LOG_INTERVAL:
            return
        self.last_log = now
        stones_flat = ';'.join(f"{s.rect.x}:{int(s.rect.y)}" for s in self.stones)
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"{now:.3f}", self.player.rect.x, self.points, self.lives, stones_flat])

    def update(self, dt, keys):
        # player movement
        if keys.get(pygame.K_LEFT):
            self.player.move_left()
        if keys.get(pygame.K_RIGHT):
            self.player.move_right()

        # spawn logic
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            # spawn a few stones depending on difficulty
            for _ in range(random.randint(1, 1 + int((1.0 - max(0.2, self.spawn_interval))/0.2))):
                self.spawn_stone()

        # update stones
        for s in list(self.stones):
            s.update(dt)
            # ground collision
            if s.rect.y >= HEIGHT:
                self.points += 10
                self.stones.remove(s)
                # increase difficulty slowly
                self.stone_speed += 0.5
                self.spawn_interval = max(0.2, self.spawn_interval - 0.01)
            # hit player
            elif s.rect.colliderect(self.player.rect):
                self.lives -= 1
                try:
                    self.stones.remove(s)
                except ValueError:
                    pass
                if self.lives <= 0:
                    # reset round
                    self.reset_round()
                    break

        # logging
        self.log_state()

    def draw(self):
        self.screen.fill((0, 0, 0))
        # draw player
        pygame.draw.rect(self.screen, (0, 200, 0), self.player.rect)
        # draw stones
        for s in self.stones:
            pygame.draw.rect(self.screen, (200, 0, 0), s.rect)
        # HUD
        font = pygame.font.Font(None, 24)
        txt = font.render(f"Points: {self.points} Lives: {self.lives}", True, (255,255,255))
        self.screen.blit(txt, (5,5))
        if not self.training:
            pygame.display.flip()

    def run(self):
        while self.running:
            dt_ms = self.clock.tick(60)
            dt = dt_ms / 1000.0 *5 # scale so speed is consistent

            # input
            keys_down = {}
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False

            pressed = pygame.key.get_pressed()
            keys_down[pygame.K_LEFT] = pressed[pygame.K_LEFT]
            keys_down[pygame.K_RIGHT] = pressed[pygame.K_RIGHT]

            # update
            self.update(dt, keys_down)

            # draw
            self.draw()

        pygame.quit()


if __name__ == '__main__':
    game = Game(training=args.train)
    game.run()
