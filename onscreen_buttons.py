"""Clickable on-screen buttons for lidar_control.py: every keyboard command, as a button.

A button does exactly what its key does, because it hands the program the same key press:
  * a tap button posts a KEYDOWN event (so the existing key handler runs; nothing is duplicated),
  * a hold button (driving, leaning) counts as its key being held down for as long as the mouse button is held,
    so letting go stops the dog just like letting go of the key,
  * a latch button (FAST) stays pressed until clicked again,
  * a command button runs a terminal command (the tricks: shake, stretch, ...).
This module only knows layout, drawing and hit-testing; lidar_control.py decides what a press does.
"""
from dataclasses import dataclass

import pygame

WIDTH = 300          # width of the button column, px
MARGIN = 8
GAP = 4
ROW_H = 22
HEAD_H = 20
TOP_H = 26           # the title bar
FOOT_H = 66          # hint text at the bottom (not scrolled)

COLORS = {
    "bg": (24, 24, 32), "title": (255, 220, 120), "group": (150, 200, 255), "text": (228, 228, 235),
    "base": (48, 50, 64), "hover": (70, 76, 100), "on": (30, 112, 72), "pressed": (205, 140, 30),
    "danger": (120, 42, 42), "border": (96, 98, 118), "hint": (170, 170, 185),
}


@dataclass
class Button:
    id: str
    label: str
    key: int = None          # pygame key it presses
    mod: int = 0             # modifier sent with it (e.g. KMOD_LSHIFT for Shift+U)
    hold: bool = False       # held down while the mouse button is held
    repeat: bool = False     # a hold button that also re-sends its key press while held (zoom)
    latch: bool = False      # stays on until clicked again
    command: str = None      # a terminal command to run instead of a key
    tip: str = ""            # what it does (shown at the bottom while the pointer is over it)
    danger: bool = False     # needs a second click (quit)


def default_groups():
    """The layout: [(title, [row, row, ...])], each row up to three Buttons (None leaves a gap)."""
    K, M = pygame, pygame.KMOD_LSHIFT
    b = Button
    return [
        ("DRIVE (hold to move)", [
            [b("strafe_l", "strafe L", K.K_q, hold=True, tip="Q  strafe left (hold)"),
             b("fwd", "FORWARD", K.K_w, hold=True, tip="W  walk forward (hold)"),
             b("strafe_r", "strafe R", K.K_e, hold=True, tip="E  strafe right (hold)")],
            [b("turn_l", "turn L", K.K_a, hold=True, tip="A  turn left (hold)"),
             b("back", "BACK", K.K_s, hold=True, tip="S  walk backward (hold)"),
             b("turn_r", "turn R", K.K_d, hold=True, tip="D  turn right (hold)")],
            [b("stop", "STOP", K.K_SPACE, tip="Space  stop, and cancel walks / laps"),
             b("stand", "stand", K.K_r, tip="R  stand up and balance"),
             b("lie", "lie down", K.K_f, tip="F  lie down")],
            [b("fast", "FAST", K.K_LSHIFT, latch=True, tip="Shift  faster speed (click to latch on / off)"),
             b("lean_dn", "nose down", K.K_UP, hold=True, tip="Up arrow  lean nose down (standing, hold)"),
             b("lean_up", "nose up", K.K_DOWN, hold=True, tip="Down arrow  lean nose up (standing, hold)")],
            [b("quit", "QUIT", K.K_ESCAPE, danger=True, tip="Esc  stop and quit (click twice)"),
             b("lean_l", "roll left", K.K_LEFT, hold=True, tip="Left arrow  roll left (standing, hold)"),
             b("lean_r", "roll right", K.K_RIGHT, hold=True, tip="Right arrow  roll right (standing, hold)")],
        ]),
        ("VIEW", [
            [b("view", "view", K.K_v, tip="V  top-down / 3D / first-person"),
             b("follow", "follow", K.K_m, tip="M  view follows the robot's heading / fixed"),
             b("help", "help", K.K_F1, tip="F1  every command, over the lidar pane")],
            [b("zoom_in", "zoom +", K.K_EQUALS, hold=True, repeat=True, tip="+  zoom in (hold to keep zooming)"),
             b("zoom_out", "zoom -", K.K_MINUS, hold=True, repeat=True, tip="-  zoom out (hold to keep zooming)"),
             b("fullscreen", "fullscreen", K.K_F11, tip="F11  fullscreen on / off")],
            [b("ground", "ground", K.K_g, tip="G  hide / show ground points"),
             b("colour", "colour", K.K_h, tip="H  colour by height / distance / camera"),
             b("mesh", "mesh", K.K_n, tip="N  block mesh view on / off (3D, first-person)")],
            [b("persist_less", "persist -", K.K_LEFTBRACKET, tip="[  forget points sooner"),
             b("persist_more", "persist +", K.K_RIGHTBRACKET, tip="]  keep points longer"),
             b("behind", "behind", K.K_F2, tip="F2  keep / drop the lidar points behind the dog")],
        ]),
        ("LIDAR + CAPTURE", [
            [b("lidar", "lidar", K.K_l, tip="L  lidar on / off"),
             b("snapshot", "snapshot", K.K_p, tip="P  save a snapshot (.ply)"),
             b("record", "record", K.K_o, tip="O  record lidar frames on / off")],
            [b("save_mesh", "save mesh", K.K_j, tip="J  save the block mesh as a coloured .ply"),
             b("forget", "forget map", K.K_k, tip="K  forget the point and obstacle maps")],
        ]),
        ("CAMERA", [
            [b("camera", "camera", K.K_i, tip="I  camera picture + lidar on / off (first-person)"),
             b("cam_size", "pic size", K.K_y, tip="Y  camera picture size"),
             b("clip", "rec clip", K.K_7, tip="7  record a camera clip on / off")],
            [b("depth", "depth map", K.K_1, tip="1  depth map on / off (Depth Anything)"),
             b("depth_filter", "depth filt", K.K_2, tip="2  drop lidar points the depth map disagrees with"),
             b("flow", "flow", K.K_3, tip="3  optical flow: tracks what moves, on / off")],
            [b("flow_view", "flow view", K.K_4, tip="4  flow view: movers / colour / arrows"),
             b("flow_less", "flow sens-", K.K_5, tip="5  flow less sensitive"),
             b("flow_more", "flow sens+", K.K_6, tip="6  flow more sensitive")],
            [b("mount_pick", "mount pick", K.K_SLASH, tip="/  pick a camera-mount setting"),
             b("mount_dn", "mount -", K.K_COMMA, tip=",  decrease that setting (saved)"),
             b("mount_up", "mount +", K.K_PERIOD, tip=".  increase that setting (saved)")],
            [b("mount_reset", "mount reset", K.K_BACKSLASH, tip="\\  reset the camera mount to defaults")],
        ]),
        ("PILLAR SCAN", [
            [b("pillar_ahead", "pick ahead", K.K_c, tip="C  pick the object straight ahead as the pillar"),
             b("lap_l", "lap left", K.K_z, tip="Z  circle the pillar to the left"),
             b("lap_r", "lap right", K.K_x, tip="X  circle the pillar to the right")],
            [b("tilt", "tilt scans", K.K_t, tip="T  tilt scans during a lap on / off"),
             b("pillar_clear", "clear", K.K_BACKSPACE, tip="Backspace  clear the pillar")],
        ]),
        ("WAYPOINTS", [
            [b("go", "GO / stop", K.K_RETURN, tip="Enter  walk the waypoints in order / stop"),
             b("wp_undo", "undo last", K.K_u, tip="U  remove the last waypoint"),
             b("wp_clear", "clear all", K.K_u, mod=M, tip="Shift+U  remove all waypoints")],
            [b("dodge", "dodge", K.K_b, tip="B  dodge things moving towards the dog, on / off")],
        ]),
        ("TRICKS (dog standing)", [
            [b("t_shake", "shake", command="shake", tip="paw wave (Hello)"),
             b("t_stretch", "stretch", command="stretch", tip="stretch"),
             b("t_happy", "happy", command="happy", tip="happy wiggle")],
            [b("t_heart", "heart", command="heart", tip="make a heart shape"),
             b("t_scrape", "scrape", command="scrape", tip="scrape the floor"),
             b("t_dance1", "dance 1", command="dance1", tip="dance routine 1")],
            [b("t_dance2", "dance 2", command="dance2", tip="dance routine 2"),
             b("t_sit", "sit", command="sit", tip="sit down like a dog"),
             b("t_rise", "rise", command="rise", tip="get up from sitting")],
            [b("t_recover", "recover", command="recover", tip="get up after a fall"),
             b("t_jump", "JUMP !", command="jump", danger=True, tip="throws the dog forward: clear 2 m. Click twice."),
             b("t_pounce", "POUNCE !", command="pounce", danger=True, tip="throws the dog forward: clear 2 m. Click twice.")],
        ]),
    ]


def all_buttons(groups):
    return [btn for _, rows in groups for row in rows for btn in row if btn is not None]


def layout(groups, rect, scroll=0):
    """Rectangles of every button for a column occupying `rect`: ([(Rect, Button)], total content height)."""
    cell_w = (rect.w - 2 * MARGIN - 2 * GAP) // 3
    y = rect.y + TOP_H - scroll
    items = []
    for _, rows in groups:
        y += HEAD_H
        for row in rows:
            for i, btn in enumerate(row):
                if btn is not None:
                    items.append((pygame.Rect(rect.x + MARGIN + i * (cell_w + GAP), y, cell_w, ROW_H), btn))
            y += ROW_H + GAP
        y += 4
    return items, y + scroll - rect.y


def max_scroll(groups, rect):
    return max(0, layout(groups, rect)[1] - (rect.h - FOOT_H))


def hit_test(items, pos):
    for r, btn in items:
        if r.collidepoint(pos):
            return btn
    return None


def draw(surface, font, rect, groups, is_active, mouse, held, armed, scroll=0, extra_hint=""):
    """Draw the column. is_active(id) -> bool lights a toggle; `held` is the Button being held; `armed` is the id
    of a danger button waiting for its second click. Returns the item list for hit-testing."""
    items, total = layout(groups, rect, scroll)
    area = pygame.Rect(rect.x, rect.y + TOP_H, rect.w, rect.h - TOP_H - FOOT_H)
    pygame.draw.rect(surface, COLORS["bg"], rect)
    old_clip = surface.get_clip()
    surface.set_clip(area)
    y = rect.y + TOP_H - scroll
    for title, rows in groups:                                   # group titles
        surface.blit(font.render(title, True, COLORS["group"]), (rect.x + MARGIN, y + 3))
        y += HEAD_H + sum(ROW_H + GAP for _ in rows) + 4
    hovered = None
    for r, btn in items:
        if r.bottom < area.top or r.top > area.bottom:
            continue
        over = r.collidepoint(mouse) and area.collidepoint(mouse)
        if over:
            hovered = btn
        if held is btn:
            color = COLORS["pressed"]
        elif armed == btn.id:
            color = COLORS["pressed"]
        elif is_active(btn.id):
            color = COLORS["on"]
        elif btn.danger:
            color = COLORS["hover"] if over else COLORS["danger"]
        else:
            color = COLORS["hover"] if over else COLORS["base"]
        pygame.draw.rect(surface, color, r, border_radius=4)
        pygame.draw.rect(surface, COLORS["border"], r, 1, border_radius=4)
        label = "again?" if armed == btn.id else btn.label
        text = font.render(label, True, COLORS["text"])
        surface.blit(text, (r.centerx - text.get_width() // 2, r.centery - text.get_height() // 2))
    surface.set_clip(old_clip)
    if total > rect.h - FOOT_H:                                  # scroll bar
        track = pygame.Rect(rect.right - 5, area.top, 3, area.h)
        thumb_h = max(20, int(area.h * area.h / total))
        thumb_y = area.top + int((area.h - thumb_h) * scroll / max(1, total - (rect.h - FOOT_H)))
        pygame.draw.rect(surface, (50, 52, 66), track)
        pygame.draw.rect(surface, (130, 134, 160), (track.x, thumb_y, 3, thumb_h))
    # title bar and footer
    pygame.draw.rect(surface, COLORS["bg"], (rect.x, rect.y, rect.w, TOP_H))
    surface.blit(font.render("BUTTONS   (F3 hides)", True, COLORS["title"]), (rect.x + MARGIN, rect.y + 6))
    foot = pygame.Rect(rect.x, rect.bottom - FOOT_H, rect.w, FOOT_H)
    pygame.draw.rect(surface, COLORS["bg"], foot)
    pygame.draw.line(surface, COLORS["border"], foot.topleft, foot.topright)
    lines = []
    if hovered is not None and hovered.tip:
        lines += wrap(hovered.tip, 40)[:2]
    elif extra_hint:
        lines += wrap(extra_hint, 40)[:2]
    lines += wrap("In the view: click = pick pillar, right-click = waypoint, drag = look / orbit, wheel = zoom", 40)
    for i, line in enumerate(lines[:4]):
        surface.blit(font.render(line, True, COLORS["text"] if i == 0 and hovered else COLORS["hint"]),
                     (rect.x + MARGIN, foot.y + 5 + i * 14))
    hide = Button("hide", "hide", tip="F3  hide the buttons")
    hide_rect = pygame.Rect(rect.right - 52, rect.y + 3, 44, 20)
    over = hide_rect.collidepoint(mouse)
    pygame.draw.rect(surface, COLORS["hover"] if over else COLORS["base"], hide_rect, border_radius=4)
    pygame.draw.rect(surface, COLORS["border"], hide_rect, 1, border_radius=4)
    text = font.render("hide", True, COLORS["text"])
    surface.blit(text, (hide_rect.centerx - text.get_width() // 2, hide_rect.centery - text.get_height() // 2))
    return items + [(hide_rect, hide)]


def wrap(text, width):
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return out + ([line] if line else [])


class VirtualKeys:
    """pygame.key.get_pressed() plus the keys that on-screen buttons are holding down."""

    def __init__(self, real, extra):
        self.real, self.extra = real, extra

    def __getitem__(self, key):
        return bool(self.real[key]) or key in self.extra
