#!/usr/bin/env python

import random
import sys

import numpy as np
import pygame
from pygame.locals import *  # noqa


class FlappyBird:
    def __init__(self, agent):
        self.agent = agent
        self.screen_width = 400
        self.screen_high = 708
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_high))
        self.bird = pygame.Rect(65, 50, 50, 50)
        self.background = pygame.image.load("assets/background.png").convert()
        self.birdSprites = [pygame.image.load("assets/1.png").convert_alpha(),
                            pygame.image.load("assets/2.png").convert_alpha(),
                            pygame.image.load("assets/dead.png")]
        self.wallUp = pygame.image.load("assets/bottom.png").convert_alpha()
        self.wallDown = pygame.image.load("assets/top.png").convert_alpha()
        self.gap = 330
        self.wallx = 400
        self.birdY = 350
        self.jump = 0
        self.jumpSpeed = 10
        self.gravity = 5
        self.dead = False
        self.sprite = 0
        self.counter = 0
        self.offset = random.randint(-110, 110)
        self.isKeyDown = False

    def update_walls(self):
        self.wallx -= 2
        if self.wallx < -80:
            self.wallx = 400
            self.counter += 1
            self.offset = random.randint(-110, 110)

    def calculate_world_position_objets(self):
        self.world_positions = np.array([
            [self.wallx,
             360 + self.gap - self.offset + 10,
             self.wallUp.get_width() - 10,
             self.wallUp.get_height()
             ],
            [self.wallx,
             0 - self.gap - self.offset - 10,
             self.wallDown.get_width() - 10,
             self.wallDown.get_height()
             ],
            [
                self.bird[0],
                self.bird[1],
                self.bird[2],
                self.bird[3]
            ]
        ])
        return self.world_positions

    def get_world_position_objets(self):
        return self.world_positions

    def bird_update(self):
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            self.birdY += self.gravity
            self.gravity += 0.2
        self.bird[1] = self.birdY
        positions = self.calculate_world_position_objets()
        upRect = pygame.Rect(positions[0][0], positions[0][1], positions[0][2], positions[0][3])
        downRect = pygame.Rect(positions[1][0], positions[1][1], positions[1][2], positions[1][3])
        # toco el tubo de arriba?
        if upRect.colliderect(self.bird):
            self.dead = True
        # toco el tubo de abajo?
        if downRect.colliderect(self.bird):
            self.dead = True
        # toco el techo o toco el fondo?
        if not 0 < self.bird[1] < 720:
            self.dead = True

    def hold_key_down(self):
        self.isKeyDown = True

    def release_key(self):
        self.isKeyDown = False

    def do_action(self):
        self.jump = 17
        self.gravity = 0
        self.jumpSpeed = 10

    def each_cycle(self):
        clock = self.clock
        font = self.font
        clock.tick(3000)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.agent.save()
                sys.exit()
            if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
                self.do_action()

        if self.isKeyDown:
            self.do_action()

        self.screen.fill((255, 255, 255))
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.wallUp,
                         (self.wallx, 360 + self.gap - self.offset))
        self.screen.blit(self.wallDown,
                         (self.wallx, 0 - self.gap - self.offset))
        self.screen.blit(font.render(str(self.counter),
                                     -1,
                                     (255, 255, 255)),
                         (200, 50))
        if self.dead:
            self.sprite = 2
        elif self.jump:
            self.sprite = 1
        self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))
        if not self.dead:
            self.sprite = 0
        self.update_walls()
        self.bird_update()
        pygame.display.update()
        self.clock = clock
        self.font = font

    def init_game(self):
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 50)
        self.calculate_world_position_objets()

    def restart_game(self):
        self.bird[1] = 100
        self.birdY = 350
        self.dead = False
        self.counter = 0
        self.wallx = 400
        self.offset = random.randint(-110, 110)
        self.gravity = 5

    def run(self):
        self.init_game()
        while True:
            self.each_cycle()
