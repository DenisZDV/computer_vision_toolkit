"""
mouse_aim.py
Smooth mouse movement with jitter effect — mimics human hand motion.

Use cases:
  - Computer vision aimbot research / education
  - RPA / UI automation where instant cursor jumps look unnatural
  - Testing anti-cheat bypass techniques (research purposes)

Backends supported:
  - win32api  (Windows, most natural, no elevation needed)
  - pynput    (cross-platform)
  - pyautogui (cross-platform, slower)

Usage:
    from utils.mouse_aim import MouseController, AimConfig

    ctrl = MouseController()
    ctrl.move_to(x=960, y=540)                  # smooth move to absolute coords
    ctrl.move_by(dx=50, dy=-30)                 # smooth relative move
    ctrl.snap_to(x=960, y=540)                  # instant teleport (no smoothing)

    # Custom config
    cfg = AimConfig(smoothing=0.15, jitter=1.5, steps=12)
    ctrl = MouseController(config=cfg)
"""

import time
import math
import random
from dataclasses import dataclass, field
from typing import Literal

# ── BACKEND DETECTION ──────────────────────────────────────────────────────

def _detect_backend() -> str:
    try:
        import win32api, win32con
        return 'win32'
    except ImportError:
        pass
    try:
        import pynput.mouse
        return 'pynput'
    except ImportError:
        pass
    try:
        import pyautogui
        return 'pyautogui'
    except ImportError:
        pass
    raise RuntimeError(
        'No mouse backend found. Install one of:\n'
        '  pip install pywin32         (Windows, recommended)\n'
        '  pip install pynput          (cross-platform)\n'
        '  pip install pyautogui       (cross-platform, slower)'
    )


# ── CONFIG ─────────────────────────────────────────────────────────────────

@dataclass
class AimConfig:
    """
    Mouse movement configuration.

    smoothing:  0.0 = instant snap, 1.0 = very slow creep
                Recommended: 0.05–0.25 for natural feel
    jitter:     Max random pixel offset per step (simulates hand tremor)
                0.0 = perfectly smooth, 2.0–4.0 = natural jitter
    steps:      Number of intermediate steps for smooth movement
                More steps = smoother but slightly more CPU
    min_delay:  Minimum sleep between steps (seconds)
    max_delay:  Maximum sleep between steps (seconds)
                Randomized interval simulates non-mechanical timing
    trigger_key: Key to hold for activation (None = always active)
    """
    smoothing:   float = 0.10
    jitter:      float = 1.5
    steps:       int   = 10
    min_delay:   float = 0.002
    max_delay:   float = 0.006
    trigger_key: str | None = None   # e.g. 'caps_lock', 'shift', None


# ── BACKENDS ───────────────────────────────────────────────────────────────

class _Win32Mouse:
    def __init__(self):
        import win32api, win32con
        self._api = win32api
        self._con = win32con

    def get_pos(self) -> tuple[int, int]:
        return self._api.GetCursorPos()

    def set_pos(self, x: int, y: int):
        self._api.SetCursorPos((x, y))

    def move_relative(self, dx: int, dy: int):
        x, y = self.get_pos()
        self.set_pos(x + dx, y + dy)

    def is_key_down(self, key: str) -> bool:
        key_codes = {
            'caps_lock': 0x14,
            'shift':     0x10,
            'ctrl':      0x11,
            'alt':       0x12,
            'lbutton':   0x01,
            'rbutton':   0x02,
            'xbutton1':  0x05,
        }
        code = key_codes.get(key.lower())
        if code is None:
            return False
        return bool(self._api.GetAsyncKeyState(code) & 0x8000)


class _PynputMouse:
    def __init__(self):
        from pynput import mouse, keyboard
        self._mouse = mouse.Controller()
        self._keyboard = keyboard

    def get_pos(self) -> tuple[int, int]:
        pos = self._mouse.position
        return (int(pos[0]), int(pos[1]))

    def set_pos(self, x: int, y: int):
        self._mouse.position = (x, y)

    def move_relative(self, dx: int, dy: int):
        self._mouse.move(dx, dy)

    def is_key_down(self, key: str) -> bool:
        # pynput doesn't have async key state; return False as safe default
        return False


class _PyAutoGUIMouse:
    def __init__(self):
        import pyautogui
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False
        self._pg = pyautogui

    def get_pos(self) -> tuple[int, int]:
        return self._pg.position()

    def set_pos(self, x: int, y: int):
        self._pg.moveTo(x, y, duration=0)

    def move_relative(self, dx: int, dy: int):
        self._pg.moveRel(dx, dy, duration=0)

    def is_key_down(self, key: str) -> bool:
        return False


# ── SMOOTHING ALGORITHMS ───────────────────────────────────────────────────

def _ease_out_quad(t: float) -> float:
    """Decelerate toward target — feels natural."""
    return 1 - (1 - t) ** 2


def _ease_in_out(t: float) -> float:
    """Accelerate then decelerate."""
    return t * t * (3 - 2 * t)


def _linear(t: float) -> float:
    return t


EASING = {
    'ease_out':    _ease_out_quad,
    'ease_in_out': _ease_in_out,
    'linear':      _linear,
}


# ── MAIN CONTROLLER ────────────────────────────────────────────────────────

class MouseController:
    """
    Smooth mouse controller with jitter simulation.

    Example:
        ctrl = MouseController()

        # Move to absolute screen position
        ctrl.move_to(960, 540)

        # Move relative
        ctrl.move_by(dx=100, dy=-50)

        # Used with bounding box from YOLO:
        box = (x1, y1, x2, y2)
        cx = (box[0] + box[2]) // 2
        cy = (box[1] + box[3]) // 2
        ctrl.move_to(cx, cy)
    """

    def __init__(
        self,
        config: AimConfig = None,
        backend: Literal['auto', 'win32', 'pynput', 'pyautogui'] = 'auto',
    ):
        self.config = config or AimConfig()

        if backend == 'auto':
            backend = _detect_backend()

        if backend == 'win32':
            self._mouse = _Win32Mouse()
        elif backend == 'pynput':
            self._mouse = _PynputMouse()
        elif backend == 'pyautogui':
            self._mouse = _PyAutoGUIMouse()
        else:
            raise ValueError(f'Unknown backend: {backend}')

        self.backend = backend

    def get_pos(self) -> tuple[int, int]:
        return self._mouse.get_pos()

    def snap_to(self, x: int, y: int):
        """Instant cursor teleport — no smoothing, no jitter."""
        self._mouse.set_pos(x, y)

    def move_to(
        self,
        x: int,
        y: int,
        easing: str = 'ease_out',
        override_steps: int = None,
    ):
        """
        Smooth move to absolute screen coordinates.

        Args:
            x, y:           Target screen coordinates
            easing:         'ease_out' | 'ease_in_out' | 'linear'
            override_steps: Override config.steps for this call
        """
        cfg = self.config
        if cfg.trigger_key and not self._mouse.is_key_down(cfg.trigger_key):
            return

        start_x, start_y = self.get_pos()
        dx = x - start_x
        dy = y - start_y
        dist = math.hypot(dx, dy)

        if dist < 1:
            return

        steps = override_steps or cfg.steps
        ease_fn = EASING.get(easing, _ease_out_quad)

        prev_x, prev_y = start_x, start_y

        for i in range(1, steps + 1):
            t = ease_fn(i / steps)

            # Interpolated target
            tx = int(start_x + dx * t)
            ty = int(start_y + dy * t)

            # Add jitter (decays near target)
            jitter_scale = (1 - t) * cfg.jitter
            jx = random.uniform(-jitter_scale, jitter_scale)
            jy = random.uniform(-jitter_scale, jitter_scale)

            nx = tx + int(jx)
            ny = ty + int(jy)

            # Move delta from previous position
            delta_x = nx - prev_x
            delta_y = ny - prev_y

            if delta_x != 0 or delta_y != 0:
                self._mouse.set_pos(nx, ny)

            prev_x, prev_y = nx, ny

            sleep_t = random.uniform(cfg.min_delay, cfg.max_delay)
            time.sleep(sleep_t)

        # Final snap to exact target
        self._mouse.set_pos(x, y)

    def move_by(
        self,
        dx: int,
        dy: int,
        easing: str = 'ease_out',
    ):
        """Smooth relative movement."""
        x, y = self.get_pos()
        self.move_to(x + dx, y + dy, easing=easing)

    def aim_at_box(
        self,
        box: tuple[int, int, int, int],
        target: Literal['center', 'top_third'] = 'center',
    ):
        """
        Aim at a YOLO bounding box.

        Args:
            box:    (x1, y1, x2, y2) in screen pixels
            target: 'center' = center of box
                    'top_third' = upper third (headshot zone)
        """
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2

        if target == 'top_third':
            # Upper third of box — approximates head position
            cy = y1 + (y2 - y1) // 3
        else:
            cy = (y1 + y2) // 2

        self.move_to(cx, cy)

    def is_trigger_held(self) -> bool:
        """Check if trigger key is currently held down."""
        if not self.config.trigger_key:
            return True
        return self._mouse.is_key_down(self.config.trigger_key)


# ── PRESET CONFIGS ─────────────────────────────────────────────────────────

def config_human_slow() -> AimConfig:
    """Slow, very human-like movement. Good for UI automation."""
    return AimConfig(smoothing=0.20, jitter=2.0, steps=20,
                     min_delay=0.005, max_delay=0.012)

def config_human_fast() -> AimConfig:
    """Fast but still smooth. Good for gaming scenarios."""
    return AimConfig(smoothing=0.08, jitter=1.5, steps=8,
                     min_delay=0.001, max_delay=0.004)

def config_precise() -> AimConfig:
    """Minimal jitter, precise targeting."""
    return AimConfig(smoothing=0.05, jitter=0.5, steps=6,
                     min_delay=0.001, max_delay=0.003)


# ── DEMO ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Mouse aim demo')
    p.add_argument('--mode', default='demo',
                   choices=['demo', 'circle', 'follow'],
                   help='Demo mode')
    p.add_argument('--backend', default='auto',
                   choices=['auto', 'win32', 'pynput', 'pyautogui'])
    args = p.parse_args()

    ctrl = MouseController(config=config_human_fast(), backend=args.backend)
    print(f'Backend: {ctrl.backend}')
    print('Starting in 2 seconds... Ctrl+C to stop')
    time.sleep(2)

    if args.mode == 'demo':
        # Move to corners
        positions = [(300, 300), (900, 300), (900, 600), (300, 600)]
        for x, y in positions:
            print(f'Moving to ({x}, {y})')
            ctrl.move_to(x, y)
            time.sleep(0.3)

    elif args.mode == 'circle':
        # Draw a circle
        cx, cy, r = 640, 400, 150
        steps = 60
        for i in range(steps + 1):
            angle = 2 * math.pi * i / steps
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            ctrl.move_to(x, y, override_steps=3)

    elif args.mode == 'follow':
        # Hover near current cursor position (jitter demo)
        print('Jitter demo — cursor will shake for 5 seconds')
        start = time.time()
        while time.time() - start < 5:
            x, y = ctrl.get_pos()
            jx = random.randint(-3, 3)
            jy = random.randint(-3, 3)
            ctrl.snap_to(x + jx, y + jy)
            time.sleep(0.016)
