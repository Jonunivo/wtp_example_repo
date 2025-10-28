# ball.py
import pygame
import random
from settings import WIDTH, HEIGHT, BALL_SPEED, SPEED_MULTIPLIER

class Ball(pygame.sprite.Sprite):
    def __init__(self, size=20, color=(255, 0, 0)):
        super().__init__()
        # Create a colored square
        self.image = pygame.Surface((size, size))
        self.image.fill(color)
        self.rect = self.image.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        self.vx = random.choice([-BALL_SPEED, BALL_SPEED])
        self.vy = random.choice([-BALL_SPEED, BALL_SPEED])

    def update(self, paddles):
        self.rect.x += self.vx
        self.rect.y += self.vy

        # Bounce off top/bottom
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.vy *= -1

        # Bounce off paddles
        if pygame.sprite.spritecollideany(self, paddles):
            self.vx *= -1 * SPEED_MULTIPLIER

        # Check scoring
        if self.rect.left <= 0:
            return "right"
        elif self.rect.right >= WIDTH:
            return "left"
        return None

    def reset(self):
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.vx = random.choice([-BALL_SPEED, BALL_SPEED])

