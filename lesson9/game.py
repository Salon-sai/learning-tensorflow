# -*- coding: utf-8 -*-

import pygame
import sys
from pygame.locals import *
import random
import math

from lesson9 import neuro_evolution

BACKGROUND = (255, 255, 255)
SCREEN_SIZE = (320, 480)

class Plane(object):

    def __init__(self, plane_image):
        self.plane_image = plane_image
        self.rect = plane_image.get_rect()

        self.width = self.rect[2]
        self.height = self.rect[3]
        self.x = SCREEN_SIZE[0] / 2 - self.width / 2
        self.y = SCREEN_SIZE[1] - self.height

        self.move_x = 0
        self.speed = 2
        self.alive = True

    def update(self):
        self.x += self.move_x * self.speed

    def draw(self, screen):
        screen.blit(self.plane_image, (self.x, self.y, self.width, self.height))

    def is_dead(self, enemes):
        if self.x < -self.width or self.x + self.width > SCREEN_SIZE[0] + self.width:
            return True

        for eneme in enemes:
            if self.collision(eneme):
                return True
        return False

    def collision(self, eneme):
        if not (self.x > eneme.x + eneme.width or
                self.x + self.width < eneme.x or
                self.y > eneme.y + eneme.height or
                self.y + self.height < eneme.y):
            return True
        else:
            return False

    def get_inputs_values(self, enemes, input_size=4):
        inputs = []
        for i in range(input_size):
            inputs.append(0.0)

        inputs[0] = (self.x * 1.0 / SCREEN_SIZE[0])
        index = 1
        for eneme in enemes:
            inputs[index] = eneme.x * 1.0 / SCREEN_SIZE[0]
            index += 1
            inputs[index] = eneme.y * 1.0 / SCREEN_SIZE[1]
            index += 1

        if len(enemes) > 0 and self.x < enemes[0].x:
            inputs[index] = -1.0
            index += 1
        else:
            inputs[index] = 1.0
        return inputs

class Enemy(object):
    def __init__(self, enemy_image):
        self.enemy_image = enemy_image
        self.rect = enemy_image.get_rect()

        self.width = self.rect[2]
        self.height = self.rect[3]
        self.x = random.choice(range(0, int(SCREEN_SIZE[0] - self.width / 2), 71))
        self.y = 0

    def update(self):
        self.y += 6

    def draw(self, screen):
        screen.blit(self.enemy_image, (self.x, self.y, self.width, self.height))

    def is_out(self):
        return True if self.y >= SCREEN_SIZE[1] else False

class Game(object):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Plane")

        self.ai = neuro_evolution.NeuroEvolution()
        self.generation = 0
        self.max_enemes = 1

        self.plane_image = pygame.image.load("./imgs/plane.jpg").convert_alpha()
        self.eneme_image = pygame.image.load("./imgs/missile.jpg").convert_alpha()

    def start(self):
        self.score = 0
        self.planes = []
        self.enemes = []

        self.gen = self.ai.next_generation()
        for i in range(len(self.gen)):
            plane = Plane(self.plane_image)
            self.planes.append(plane)

        self.generation += 1
        self.alives = len(self.planes)

    def update(self, screen):
        for i in range(len(self.planes)):
            if self.planes[i].alive:
                inputs = self.planes[i].get_inputs_values(self.enemes)
                res = self.gen[i].feed_forward(inputs)
                if res[0] < 0.5:
                    self.planes[i].move_x = -1
                elif res[0] > 0.55:
                    self.planes[i].move_x = 1

                self.planes[i].update()
                self.planes[i].draw(screen)

                if self.planes[i].is_dead(self.enemes) == True:
                    self.planes[i].alive = False
                    self.alives -= 1
                    self.ai.network_score(self.score, self.gen[i])
                    if self.is_ai_all_dead():
                        self.start()
        self.gen_enemes()

        for i in range(len(self.enemes)):
            self.enemes[i].update()
            self.enemes[i].draw(screen)
            if self.enemes[i].is_out():
                del self.enemes[i]
                break

        self.score += 1

        print("alive: {}, generation: {}, score {}".format(self.alives, self.generation, self.score))

    def run(self, FPS=100):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.screen.fill(BACKGROUND)
            self.update(self.screen)
            pygame.display.update()
            self.clock.tick(FPS)

    def gen_enemes(self):
        if len(self.enemes) < self.max_enemes:
            enemy = Enemy(self.eneme_image)
            self.enemes.append(enemy)

    def is_ai_all_dead(self):
        for plane in self.planes:
            if plane.alive:
                return False
        return True

game = Game()
game.start()
game.run()