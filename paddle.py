# paddle.py
import pygame
from settings import HEIGHT

class Paddle(pygame.sprite.Sprite):
    def __init__(self, x, width=15, height=100, color=(200, 200, 200)):
        super().__init__()
        # Create a simple colored surface instead of loading an image
        self.image = pygame.Surface((width, height))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.centery = HEIGHT // 2
        self.rect.x = x
        self.speed = 0

    def update(self):
        self.rect.y += self.speed
        # Prevent leaving screen
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT
