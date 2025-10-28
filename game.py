# game.py
import pygame
from paddle import Paddle
from ball import Ball
from settings import *
import os

class Game:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()

        # Optional background (scaled to 0.5 size)
        bg_path = os.path.join(ASSET_DIR, "background.png")
        if os.path.exists(bg_path):
            bg = pygame.image.load(bg_path).convert()
            w, h = bg.get_size()
            self.background = pygame.transform.scale(bg, (int(w * 0.5), int(h * 0.5)))
        else:
            self.background = None


        # Font fallback
        font_path = os.path.join(ASSET_DIR, "font.TTF")
        if os.path.exists(font_path):
            self.font = pygame.font.Font(font_path, 48)
        else:
            self.font = pygame.font.SysFont(None, 48)

        self.reset_game()

    def reset_game(self):
        self.paddle_left = Paddle(50)
        self.paddle_right = Paddle(WIDTH - 65)
        self.ball = Ball()
        self.paddles = pygame.sprite.Group(self.paddle_left, self.paddle_right)
        self.all_sprites = pygame.sprite.Group(self.paddle_left, self.paddle_right, self.ball)
        self.score_left = 0
        self.score_right = 0
        self.game_over = False
        self.winner = None

    def handle_input(self):
        keys = pygame.key.get_pressed()
        self.paddle_left.speed = 0
        self.paddle_right.speed = 0

        if keys[pygame.K_w]:
            self.paddle_left.speed = -PADDLE_SPEED
        elif keys[pygame.K_s]:
            self.paddle_left.speed = PADDLE_SPEED

        if keys[pygame.K_UP]:
            self.paddle_right.speed = -PADDLE_SPEED
        elif keys[pygame.K_DOWN]:
            self.paddle_right.speed = PADDLE_SPEED

    def update(self):
        self.handle_input()
        self.paddle_left.update()
        self.paddle_right.update()
        result = self.ball.update(self.paddles)

        if result == "left":
            self.score_left += 1
            self.ball.reset()
        elif result == "right":
            self.score_right += 1
            self.ball.reset()


    def draw(self):
        if self.background:
            self.screen.blit(self.background, (0, 0))
        else:
            self.screen.fill(BLACK)

        self.all_sprites.draw(self.screen)
        score_text = self.font.render(f"{self.score_left}   {self.score_right}", True, WHITE)
        self.screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))
        pygame.display.flip()


