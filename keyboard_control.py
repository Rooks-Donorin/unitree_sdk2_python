"""Drive the Go2 with the keyboard.

Usage:  .venv/bin/python keyboard_control.py eth0

A small window opens; it must have focus for keys to register.
  W / S      forward / backward
  A / D      turn left / right
  Q / E      strafe left / right
  Shift      hold for faster speed
  Space      stop
  R          stand up (into balance stand, ready to walk)
  F          lie down
  Esc        stop and quit

The dog only moves while keys are held. Releasing all keys, or the window
losing focus, sends StopMove.
"""
import sys
import time

import pygame

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

# Speeds (m/s, m/s, rad/s) for normal and Shift-held driving
NORMAL = (0.4, 0.3, 0.8)
FAST = (0.8, 0.5, 1.5)
SEND_HZ = 20


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <network_interface>   e.g. eth0")
        sys.exit(1)

    print("WARNING: make sure the area around the robot is clear.")
    ChannelFactoryInitialize(0, sys.argv[1])
    sport = SportClient()
    sport.SetTimeout(10.0)
    sport.Init()

    pygame.init()
    screen = pygame.display.set_mode((460, 300))
    pygame.display.set_caption("Go2 keyboard control")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    status = "Press R to stand up"
    moving = False
    last_send = 0.0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.WINDOWFOCUSLOST and moving:
                sport.StopMove()
                moving = False
                status = "Window lost focus - stopped"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    sport.StopMove()
                    moving = False
                    status = "Stopped"
                elif event.key == pygame.K_r:
                    status = "Standing up..."
                    sport.StandUp()
                    time.sleep(1.0)
                    sport.BalanceStand()
                    status = "Standing - ready to drive"
                elif event.key == pygame.K_f:
                    sport.StopMove()
                    moving = False
                    sport.StandDown()
                    status = "Lying down"

        keys = pygame.key.get_pressed()
        fast = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        vx_max, vy_max, vyaw_max = FAST if fast else NORMAL
        vx = vx_max * (keys[pygame.K_w] - keys[pygame.K_s])
        vy = vy_max * (keys[pygame.K_q] - keys[pygame.K_e])
        vyaw = vyaw_max * (keys[pygame.K_a] - keys[pygame.K_d])

        now = time.time()
        if vx or vy or vyaw:
            if now - last_send >= 1.0 / SEND_HZ:
                sport.Move(vx, vy, vyaw)
                last_send = now
                moving = True
        elif moving:
            sport.StopMove()
            moving = False

        screen.fill((25, 25, 30))
        lines = [
            "W/S fwd/back   A/D turn   Q/E strafe",
            "Shift fast   Space stop   Esc quit",
            "R stand up   F lie down",
            "",
            f"vx   {vx:+.2f} m/s",
            f"vy   {vy:+.2f} m/s",
            f"vyaw {vyaw:+.2f} rad/s",
            f"speed: {'FAST' if fast else 'normal'}",
            "",
            status,
        ]
        for i, line in enumerate(lines):
            screen.blit(font.render(line, True, (220, 220, 220)), (16, 14 + i * 26))
        pygame.display.flip()
        clock.tick(60)

    sport.StopMove()
    pygame.quit()


if __name__ == "__main__":
    main()
