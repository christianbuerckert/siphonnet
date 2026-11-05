"""Render the SiphonNet demo video (2D, 1920x1080, 30 fps) from video/build/data.npz.

    python video/render.py                      # full video -> video/out/siphonnet_<lang>.mp4
    python video/render.py --stills 5,12,40     # PNG stills at those seconds -> video/out/stills/
    python video/render.py --scale 0.5          # half-resolution preview
    python video/render.py --lang de            # German captions

Levels, pulses, pump patterns and readout scores are simulation data. Only the look of
the water streams (dashes, ripples, travelling dots) is decoration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
BUILD, OUT = HERE / "build", HERE / "out"
W, H, FPS = 1920, 1080, 30

BG = "#0a1120"
PANEL = "#111b2e"
GLASS = "#9fb6cf"
WATER = "#2f8fe8"
WATER_LIGHT = "#7cc4ff"
ACCENT = "#ffc857"
CYAN = "#5ee6d0"
TEXT = "#e8eef7"
MUTED = "#7f8ea6"
SANS, MONO = "Avenir Next", "Menlo"
NET_CX = 900  # horizontal centre of the network

TEXTS = {
    "en": {
        "title": "SiphonNet",
        "subtitle": "A computer made of self-emptying water containers",
        "one": "One siphon",
        "fill": "A container fills with water.",
        "prime": "At the trigger level the siphon primes and empties the container in one go.",
        "break": "At the break level air enters the bell. The siphon stops, the container refills.",
        "physics": "No sensor. No valve. No electronics. A threshold made of physics.",
        "h_on": "trigger level",
        "h_off": "break level",
        "bell": "bell",
        "standpipe": "standpipe",
        "outlet": "outlet",
        "inflow": "inflow",
        "level_axis": "fill level [cm]",
        "time_axis": "time [s]",
        "net_head": "{n} siphons · {pipes} pipes · 35 pumps · water only runs downhill",
        "connect": "Now stack {n} of them in {rows} rows. Each siphon pours into 4 containers in the row below.",
        "wiring": "35 pumps feed the containers. A readout watches every fill level.",
        "votes": "Gold lines: the containers whose levels pushed the winning letter up the most.",
        "input": "INPUT",
        "pumps": "35 pumps\n3 pixels flipped at random",
        "output": "RECOGNISED",
        "readout": "READOUT",
        "clock": "simulated {t:4.1f} s / {T:.0f} s",
        "pixels": "Every pixel of the input drives a pump.",
        "untrained": "The network itself is never trained. Only the readout on the right is.",
        "accuracy": "Test accuracy on noisy letters: {acc:.0f} %",
        "end1": "Every water level in this video comes from a physics simulation.",
        "end2": "Only the readout is trained.",
        "end3": "SiphonNet · open research code",
    },
    "de": {
        "title": "SiphonNet",
        "subtitle": "Ein Rechner aus Wasserbehältern, die sich selbst entleeren",
        "one": "Ein Siphon",
        "fill": "Ein Behälter füllt sich mit Wasser.",
        "prime": "Am Auslösepegel springt der Siphon an und leert den Behälter in einem Zug.",
        "break": "Am Abrisspegel kommt Luft unter die Glocke. Der Siphon reißt ab, der Behälter füllt sich neu.",
        "physics": "Kein Sensor. Kein Ventil. Keine Elektronik. Eine Schwelle aus reiner Physik.",
        "h_on": "Auslösepegel",
        "h_off": "Abrisspegel",
        "bell": "Glocke",
        "standpipe": "Standrohr",
        "outlet": "Ablauf",
        "inflow": "Zulauf",
        "level_axis": "Füllstand [cm]",
        "time_axis": "Zeit [s]",
        "net_head": "{n} Siphons · {pipes} Leitungen · 35 Pumpen · Wasser fließt nur bergab",
        "connect": "Jetzt {n} davon in {rows} Reihen stapeln. Jeder Siphon gießt in 4 Behälter der Reihe darunter.",
        "wiring": "35 Pumpen speisen die Behälter. Ein Ausleser beobachtet jeden Füllstand.",
        "votes": "Goldene Linien: die Behälter, deren Füllstände den Gewinner am stärksten nach oben drücken.",
        "input": "EINGABE",
        "pumps": "35 Pumpen\n3 Pixel zufällig vertauscht",
        "output": "ERKANNT",
        "readout": "AUSLESER",
        "clock": "simuliert {t:4.1f} s / {T:.0f} s",
        "pixels": "Jedes Pixel der Eingabe treibt eine Pumpe an.",
        "untrained": "Das Netz selbst wird nie trainiert. Nur der Ausleser rechts.",
        "accuracy": "Trefferquote bei verrauschten Buchstaben: {acc:.0f} %",
        "end1": "Jeder Wasserstand in diesem Video stammt aus einer Physik-Simulation.",
        "end2": "Trainiert wird nur der Ausleser.",
        "end3": "SiphonNet · offener Forschungscode",
    },
}

# timeline (seconds)
T_TITLE = 3.5
T_SINGLE = 20.0
T_INTRO = 9.0
L_TYPE, L_RUN, L_READ, L_RESET = 0.8, 5.0, 2.2, 0.6
L_DUR = L_TYPE + L_RUN + L_READ + L_RESET
T_OUTRO = 7.0
LETTERS_AZ = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def fade(t: float, start: float, end: float, ramp: float = 0.5) -> float:
    """1 inside [start, end], ramping in and out."""
    return ease((t - start) / ramp) * ease((end - t) / ramp)


class Video:
    def __init__(self, lang: str):
        self.d = {k: v for k, v in np.load(BUILD / "data.npz").items()}
        self.meta = json.loads((BUILD / "meta.json").read_text())
        self.tx = TEXTS[lang]
        self.word = self.meta["word"]
        self.t_single = T_TITLE
        self.t_intro = self.t_single + T_SINGLE
        self.t_letters = self.t_intro + T_INTRO
        self.t_outro = self.t_letters + len(self.word) * L_DUR
        self.duration = self.t_outro + T_OUTRO
        self._grid()
        self.fig = None

    # --- geometry of the network grid -----------------------------------------------
    def _grid(self):
        n = self.d["h_on"].size
        rows, cols = int(self.d["rows"]), int(self.d["cols"])
        self.cw, self.ch = 32, 52  # container size in px
        dx, dy = 56, 112
        x0 = NET_CX - (cols - 1) * dx / 2
        y_top = 660  # floor of the top row
        idx = np.arange(n)
        self.cx = (x0 + (idx % cols) * dx).astype(float)
        self.base = (y_top - (idx // cols) * dy).astype(float)
        # pumps: the 5x7 glyph grid above the network, hoses fanning out to their containers
        r, c = np.divmod(np.arange(35), 5)
        self.pump_x = NET_CX + (c - 2) * 34.0
        self.pump_y = 1012 - r * 29.0
        w_in = self.d["w_in"]
        self.hoses = [(p, i) for p in range(35) for i in np.nonzero(w_in[p])[0]]
        self.hose_seg = np.array([[(self.pump_x[p], self.pump_y[p] - 11), (self.cx[i], self.base[i] + self.ch + 15)]
                                  for p, i in self.hoses])
        self.hose_phase = np.random.default_rng(5).random(len(self.hoses))
        # readout rows A..Z on the right
        self.row_y = 922 - np.arange(26) * 29.5
        targets = self.d["targets"]
        self.active = self.d["fractions"] > 0  # (n, k) pipes that exist
        self.n_pipes = int(self.active.sum())
        curves = []
        s = np.linspace(0, 1, 16)[:, None]
        for i in range(n):
            for j in targets[i]:
                # S-bend downhill: out of the outlet, across, into the top of the lower container
                p0 = np.array([self.cx[i], self.base[i] - 3])
                p3 = np.array([self.cx[j], self.base[j] + self.ch + 3])
                ym = (p0[1] + p3[1]) / 2
                p1, p2 = np.array([p0[0], ym]), np.array([p3[0], ym])
                curves.append((1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s**2 * p2 + s**3 * p3)
        self.curves = np.array(curves).reshape(n, targets.shape[1], len(s), 2)

    def head(self) -> str:
        return self.tx["net_head"].format(n=self.d["h_on"].size, pipes=self.n_pipes)

    # --- drawing primitives ---------------------------------------------------------
    def text(self, x, y, s, size=28, color=TEXT, alpha=1.0, ha="center", va="center", font=SANS, weight="normal",
             rotation=0, box=False):
        if alpha <= 0.01:
            return
        bbox = dict(boxstyle="round,pad=0.35", fc=BG, ec="none", alpha=0.85 * alpha) if box else None
        self.ax.text(x, y, s, fontsize=size, color=color, alpha=alpha, ha=ha, va=va, family=font, weight=weight,
                     rotation=rotation, bbox=bbox, multialignment=ha if ha != "center" else "center")

    def caption(self, s, alpha=1.0, y=62):
        self.text(W / 2, y, s, size=27, alpha=alpha)

    def panel(self, x, y, w, h, alpha=1.0):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=18",
                                         fc=PANEL, ec="#22314d", lw=1.5, alpha=alpha))

    def stream(self, x, y_top, y_bottom, t, width=10, alpha=1.0, speed=260):
        """Falling water: a band with moving highlights (decoration)."""
        if alpha <= 0.01 or y_top <= y_bottom:
            return
        ax = self.ax
        ax.add_patch(Rectangle((x - width / 2, y_bottom), width, y_top - y_bottom, fc=WATER, ec="none", alpha=0.8 * alpha))
        phase = (t * speed) % 40
        ys = np.arange(y_top - phase, y_bottom, -40)
        segs = [[(x, yy), (x, max(y_bottom, yy - 14))] for yy in ys]
        ax.add_collection(LineCollection(segs, colors=WATER_LIGHT, linewidths=width * 0.18, alpha=0.8 * alpha))

    # --- scenes ---------------------------------------------------------------------
    def scene_title(self, t):
        a = fade(t, 0.2, T_TITLE, 0.8)
        self.text(W / 2, H / 2 + 40, self.tx["title"], size=96, alpha=a, weight="bold")
        self.text(W / 2, H / 2 - 60, self.tx["subtitle"], size=34, color=MUTED, alpha=a)

    def scene_single(self, t):
        d, tx, ax = self.d, self.tx, self.ax
        dt = float(d["single_dt"])
        lv, fl, st = d["single_level"], d["single_flowing"], d["single_starts"]
        k = min(len(lv) - 1, int(t / dt))
        h, flowing = lv[k], fl[k] > 0
        h_on, h_off, h_max = float(d["single_h_on"]), float(d["single_h_off"]), float(d["single_h_max"])
        a = fade(t, 0, T_SINGLE, 0.6)

        ppc, floor, x0, x1 = 55, 270, 250, 650
        xc = (x0 + x1) / 2
        y = lambda cm: floor + cm * ppc
        self.text(110, 1000, tx["one"], size=40, ha="left", alpha=a, weight="bold")

        # water with a gentle ripple
        xs = np.linspace(x0, x1, 60)
        ripple = (2.5 if not flowing else 4.0) * np.sin(xs / 45 + t * 4.0)
        top = y(h) + ripple
        ax.add_patch(Polygon(np.vstack([np.c_[xs, top], [[x1, floor], [x0, floor]]]), fc=WATER, ec="none", alpha=0.5 * a))
        ax.plot(xs, top, color=WATER_LIGHT, lw=2, alpha=0.8 * a)

        # standpipe (inner air or rushing water) and outlet below the floor
        pw, out_y = 26, floor - 170
        inside = WATER if flowing else BG
        ax.add_patch(Rectangle((xc - pw / 2, out_y), pw, y(h_on) - out_y, fc=inside, ec=GLASS, lw=2, alpha=a))
        if flowing:
            self.stream(xc, y(h_on), out_y, t, width=pw * 0.6, alpha=a, speed=420)
            self.stream(xc, out_y, 120, t, width=pw * 0.7, alpha=a, speed=420)
        # bell: inverted cup over the standpipe, rim at the break level
        bw, b_top = 96, y(h_on + 0.8)
        ax.plot([xc - bw / 2, xc - bw / 2, xc + bw / 2, xc + bw / 2], [y(h_off), b_top, b_top, y(h_off)],
                color=GLASS, lw=4, alpha=a, solid_capstyle="round")
        # container walls
        ax.plot([x0, x0, xc - pw / 2], [y(h_max + 0.6), floor, floor], color=GLASS, lw=4, alpha=a)
        ax.plot([xc + pw / 2, x1, x1], [floor, floor, y(h_max + 0.6)], color=GLASS, lw=4, alpha=a)
        # inflow tap and stream
        tap_y = y(h_max + 1.6)
        ax.plot([120, x0 + 70, x0 + 70], [tap_y, tap_y, tap_y - 18], color=GLASS, lw=10, alpha=a, solid_capstyle="butt")
        self.stream(x0 + 70, tap_y - 18, top[np.argmin(np.abs(xs - (x0 + 70)))], t, width=9, alpha=a)
        # level marks and labels
        for cm, col, label in ((h_on, ACCENT, tx["h_on"]), (h_off, CYAN, tx["h_off"])):
            ax.plot([x0 - 30, x1 + 10], [y(cm), y(cm)], color=col, lw=1.5, ls=(0, (6, 6)), alpha=0.9 * a)
            self.text(x0 - 40, y(cm), label, size=17, color=col, ha="right", alpha=a)
        self.text(xc + bw / 2 + 18, b_top - 8, tx["bell"], size=17, color=MUTED, ha="left", alpha=a)
        ax.plot([xc + pw / 2 + 2, x1 + 18], [y(4.5), y(4.5)], color=MUTED, lw=1, alpha=0.8 * a)
        self.text(x1 + 26, y(4.5), tx["standpipe"], size=17, color=MUTED, ha="left", alpha=a)
        self.text(xc + pw / 2 + 12, 150, tx["outlet"], size=17, color=MUTED, ha="left", alpha=a)
        self.text(120, tap_y + 28, tx["inflow"], size=17, color=MUTED, ha="left", alpha=a)

        # level over time
        px0, px1, py0, py1 = 930, 1820, 300, 760
        tt = np.arange(k + 1) * dt
        X = px0 + tt / T_SINGLE * (px1 - px0)
        Y = py0 + lv[: k + 1] / h_max * (py1 - py0)
        ax.plot([px0, px0, px1], [py1 + 20, py0, py0], color=MUTED, lw=1.5, alpha=a)
        for cm, col in ((h_on, ACCENT), (h_off, CYAN)):
            yy = py0 + cm / h_max * (py1 - py0)
            ax.plot([px0, px1], [yy, yy], color=col, lw=1.2, ls=(0, (6, 6)), alpha=0.7 * a)
        for sec in range(0, int(T_SINGLE) + 1, 5):
            xx = px0 + sec / T_SINGLE * (px1 - px0)
            self.text(xx, py0 - 26, f"{sec}", size=15, color=MUTED, alpha=a)
        for cm in (0, 5, 10):
            self.text(px0 - 18, py0 + cm / h_max * (py1 - py0), f"{cm}", size=15, color=MUTED, ha="right", alpha=a)
        self.text((px0 + px1) / 2, py0 - 62, tx["time_axis"], size=17, color=MUTED, alpha=a)
        self.text(px0 - 62, (py0 + py1) / 2, tx["level_axis"], size=17, color=MUTED, alpha=a, rotation=90)
        ax.plot(X, Y, color=WATER_LIGHT, lw=3, alpha=a)
        ax.add_patch(Circle((X[-1], Y[-1]), 7, fc=WATER_LIGHT, ec="none", alpha=a))
        for s in np.nonzero(st[: k + 1])[0]:
            ax.add_patch(Circle((px0 + s * dt / T_SINGLE * (px1 - px0), py0 + lv[s] / h_max * (py1 - py0)), 9,
                                fc=ACCENT, ec="none", alpha=a))

        # captions keyed to the simulated events
        starts = np.nonzero(st)[0] * dt
        flow = fl > 0
        breaks = np.nonzero(flow[:-1] & ~flow[1:])[0] * dt
        t_prime, t_break = starts[0], breaks[0]
        cues = [(0.0, t_prime - 0.3, tx["fill"]), (t_prime - 0.3, t_break + 0.6, tx["prime"]),
                (t_break + 0.6, t_break + 5.5, tx["break"]), (t_break + 5.5, T_SINGLE, tx["physics"])]
        for c0, c1, s in cues:
            self.caption(s, alpha=fade(t, c0, c1, 0.35))

    def draw_network(self, levels, flowing=None, starts_age=None, inflow=None, alpha=1.0, pipes=1.0, t=0.0,
                     pipe_alpha=0.07):
        """Containers, pipes and active pulses. ``starts_age`` maps container -> age in [0, 1)."""
        ax, d = self.ax, self.d
        n = levels.size
        h_max = float(d["h_max"])
        if pipes > 0:
            segs = self.curves[self.active]
            segs = segs[: int(pipes * len(segs))]
            ax.add_collection(LineCollection(segs, colors=GLASS, linewidths=0.8, alpha=pipe_alpha * alpha))
        hot, dots, fade_a = [], [], []
        for i, age in (starts_age or {}).items():
            for e in np.nonzero(self.active[i])[0]:
                hot.append(self.curves[i, e])
                dots.append(self.curves[i, e, min(self.curves.shape[2] - 1, int(age * self.curves.shape[2]))])
                fade_a.append(1 - age)
        if hot:
            fade_a = np.array(fade_a)
            cols = np.zeros((len(hot), 4))
            cols[:, :3] = matplotlib.colors.to_rgb(ACCENT)
            cols[:, 3] = 0.85 * fade_a * alpha
            ax.add_collection(LineCollection(hot, colors=cols, linewidths=2.2))
            dcols = cols.copy()
            dcols[:, 3] = np.minimum(1, 1.6 * cols[:, 3])
            dots = np.array(dots)
            ax.scatter(dots[:, 0], dots[:, 1], s=40, c=dcols, edgecolors="none")
        # drain gutter under the bottom row: whatever reaches it leaves the network
        y_g = self.base.min() - 24
        ax.plot([self.cx.min() - 40, self.cx.max() + 40], [y_g, y_g], color=GLASS, lw=3, alpha=0.35 * alpha,
                solid_capstyle="round")
        x = self.cx - self.cw / 2
        water = [Rectangle((x[i], self.base[i]), self.cw, self.ch * min(1, levels[i] / h_max)) for i in range(n)]
        ax.add_collection(PatchCollection(water, fc=WATER, ec="none", alpha=0.75 * alpha))
        glass = [Rectangle((x[i], self.base[i]), self.cw, self.ch) for i in range(n)]
        ec = np.tile(matplotlib.colors.to_rgba(GLASS, 0.55 * alpha), (n, 1))
        lw = np.full(n, 1.2)
        if starts_age:
            for i, age in starts_age.items():
                ec[i] = matplotlib.colors.to_rgba(ACCENT, alpha * (1 - 0.7 * age))
                lw[i] = 3.0
        ax.add_collection(PatchCollection(glass, fc="none", ec=ec, lw=lw))
        if flowing is not None:
            on = np.nonzero(flowing > 0)[0]
            segs = [[(self.cx[i], self.base[i] - 2), (self.cx[i], self.base[i] - 13)] for i in on]
            ax.add_collection(LineCollection(segs, colors=WATER_LIGHT, linewidths=3, alpha=alpha))
        if inflow is not None:
            rel = inflow / inflow.max()
            flicker = 0.75 + 0.25 * np.sin(t * 9 + np.arange(n))
            cols = np.zeros((n, 4))
            cols[:, :3] = matplotlib.colors.to_rgb(WATER_LIGHT)
            cols[:, 3] = rel * flicker * alpha * 0.9
            segs = [[(self.cx[i], self.base[i] + self.ch + 12), (self.cx[i], self.base[i] + self.ch + 2)] for i in range(n)]
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=2.5))

    def draw_hoses(self, pumps, alpha, t, droplets=False, base_alpha=0.025):
        """Pump hoses from the input grid to the containers; lit when their pump runs."""
        if alpha <= 0.01:
            return
        v = np.array([pumps[p] for p, _ in self.hoses])
        cols = np.zeros((len(self.hoses), 4))
        cols[:, :3] = matplotlib.colors.to_rgb(WATER_LIGHT)
        cols[:, 3] = (base_alpha + 0.12 * np.clip(v, 0, 1.2)) * alpha
        self.ax.add_collection(LineCollection(self.hose_seg, colors=cols, linewidths=0.75))
        on = np.nonzero(v > 0.05)[0]
        if droplets and on.size:  # decoration: water travelling down the active hoses
            f = (t * 0.9 + self.hose_phase[on]) % 1.0
            p0, p1 = self.hose_seg[on, 0], self.hose_seg[on, 1]
            pts = p0 + (p1 - p0) * f[:, None]
            self.ax.scatter(pts[:, 0], pts[:, 1], s=9, c=WATER_LIGHT, alpha=0.8 * alpha, edgecolors="none")

    def pump_grid(self, pumps, alpha, t):
        ax = self.ax
        for p in range(35):
            x, y = self.pump_x[p], self.pump_y[p]
            ax.add_patch(Circle((x, y), 11.5, fc="#0d1526", ec="#2a3b5c", lw=1.2, alpha=alpha))
            if pumps[p] > 0:
                glow = min(1.0, pumps[p]) * (0.85 + 0.15 * math.sin(t * 7 + p))
                ax.add_patch(Circle((x, y), 10.5, fc=WATER, ec=WATER_LIGHT, lw=1.2, alpha=alpha * glow))
        self.text(NET_CX - 112, 925, self.tx["pumps"], size=15, color=MUTED, ha="right", alpha=alpha, box=True)

    def left_panel(self, typed, done, alpha):
        self.panel(30, 110, 285, 900, alpha=alpha)
        self.text(55, 975, self.tx["input"], size=20, color=MUTED, ha="left", alpha=alpha, weight="bold")
        self.ax.add_patch(FancyBboxPatch((55, 870), 235, 70, boxstyle="round,pad=0,rounding_size=10",
                                         fc="#0b1322", ec="#33476b", lw=1.5, alpha=alpha))
        self.text(70, 905, typed, size=32, ha="left", font=MONO, alpha=alpha)
        self.text(55, 330, self.tx["output"], size=20, color=MUTED, ha="left", alpha=alpha, weight="bold")
        self.text(70, 255, done, size=44, ha="left", font=MONO, color=ACCENT, alpha=alpha, weight="bold")

    def readout_panel(self, scores, grow, win, highlight, alpha, clock=None):
        ax = self.ax
        self.panel(1490, 110, 400, 900, alpha=alpha)
        self.text(1520, 975, self.tx["readout"], size=20, color=MUTED, ha="left", alpha=alpha, weight="bold")
        rel = np.zeros(26) if scores is None else np.clip(scores - scores.min(), 0, None) / (np.ptp(scores) + 1e-9)
        for j, ch in enumerate(LETTERS_AZ):
            y = self.row_y[j]
            is_win = highlight and j == win
            self.text(1525, y, ch, size=14, color=ACCENT if is_win else MUTED, alpha=alpha, font=MONO,
                      weight="bold" if is_win else "normal")
            ax.add_patch(Rectangle((1545, y - 9), 320, 18, fc="#0d1526", ec="none", alpha=alpha))
            ax.add_patch(Rectangle((1545, y - 9), max(2, 320 * rel[j] * grow), 18,
                                   fc=ACCENT if is_win else "#3c5b8c", ec="none", alpha=alpha))
        if clock is not None:
            self.text(1690, 140, clock, size=17, color=MUTED, font=MONO, alpha=alpha)

    def vote_lines(self, contrib, win, alpha):
        """Gold lines from the containers that pushed the winning letter's score up the most."""
        c = contrib[win]
        top = np.argsort(c)[::-1][:24]
        top = top[c[top] > 0]
        if not top.size or alpha <= 0.01:
            return
        w = c[top] / c[top].max()
        end = np.array([1510.0, self.row_y[win]])
        s = np.linspace(0, 1, 24)[:, None]
        segs = []
        for i in top:
            p0 = np.array([self.cx[i] + self.cw / 2 + 2, self.base[i] + self.ch / 2])
            p1 = np.array([max(p0[0], end[0] - 120), p0[1]])
            segs.append((1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s**2 * end)
        cols = np.zeros((top.size, 4))
        cols[:, :3] = matplotlib.colors.to_rgb(ACCENT)
        cols[:, 3] = alpha * (0.25 + 0.65 * w)
        self.ax.add_collection(LineCollection(segs, colors=cols, linewidths=1 + 2.2 * w))
        boxes = [Rectangle((self.cx[i] - self.cw / 2 - 4, self.base[i] - 4), self.cw + 8, self.ch + 8) for i in top]
        self.ax.add_collection(PatchCollection(boxes, fc="none", ec=ACCENT, lw=1.6, alpha=0.8 * alpha))

    def scene_intro(self, t):
        a = fade(t, 0, T_INTRO + 1, 0.8)
        zero = np.zeros(35)
        wiring = ease((t - 4.0) / 1.5)
        self.draw_hoses(zero, a * wiring, t, base_alpha=0.12)
        self.draw_network(self.d["h_off"], alpha=a * ease(t / 1.2), pipes=ease((t - 1.0) / 2.5),
                          pipe_alpha=0.07 + 0.12 * fade(t, 1.0, T_INTRO + 1, 1.5))
        self.pump_grid(zero, a * ease((t - 4.0) / 1.0), t)
        self.readout_panel(None, 0, 0, False, a * ease((t - 5.5) / 1.0))
        self.text(NET_CX, 128, self.head(), size=18, color=MUTED, alpha=a * ease((t - 1.0) / 1.0))
        self.caption(self.tx["connect"].format(n=self.d["h_on"].size, rows=int(self.d["rows"])), alpha=fade(t, 0.6, 4.2, 0.4))
        self.caption(self.tx["wiring"], alpha=fade(t, 4.2, T_INTRO + 1, 0.4))

    def scene_letters(self, t):
        d, tx = self.d, self.tx
        n_letters = len(self.word)
        L = min(n_letters - 1, int(t / L_DUR))
        phi = t - L * L_DUR
        a = fade(t, 0, n_letters * L_DUR + 1, 0.6)
        steps = d["levels"].shape[1]
        t_read, t_reset = L_TYPE + L_RUN, L_TYPE + L_RUN + L_READ

        if phi < L_TYPE:
            levels, flowing, ages = d["h_off"], None, {}
        elif phi < t_read:
            k = min(steps - 1, int((phi - L_TYPE) / L_RUN * steps))
            levels, flowing = d["levels"][L, k], d["flowing"][L, k]
            look = 20  # steps of history shown as travelling pulses
            ages = {}
            for s in range(max(0, k - look + 1), k + 1):
                for i in np.nonzero(d["starts"][L, s])[0]:
                    ages[int(i)] = (k - s) / look
        else:
            final = d["levels"][L, steps - 1]
            r = ease((phi - t_reset) / L_RESET)
            levels, flowing, ages = final * (1 - r) + d["h_off"] * r, None, {}

        running = L_TYPE <= phi < t_read
        out = 1 - ease((phi - t_reset) / L_RESET)
        pump_a = ease((phi - 0.3) / 0.4) * out
        pumps = d["word_pumps"][L]
        self.draw_hoses(pumps * pump_a, a, t, droplets=running)
        self.draw_network(levels, flowing, ages, d["inflow"][L] if running else None, alpha=a, t=t)
        self.pump_grid(pumps * pump_a, a, t)
        self.text(NET_CX, 128, self.head(), size=18, color=MUTED, alpha=a)

        typed = self.word[:L] + (self.word[L] if phi > 0.3 else "")
        cursor = "▏" if (t * 2) % 2 < 1 else " "
        done = self.word[:L] + (self.word[L] if phi > t_read + 0.8 else "")
        self.left_panel(typed + cursor, done, a)

        sc = d["scores"][L]
        win = int(np.argmax(sc))
        grow = ease((phi - t_read) / 0.8) * out
        sim_t = 0.0 if phi < L_TYPE else min(1.0, (phi - L_TYPE) / L_RUN) * steps * float(d["net_dt"])
        self.readout_panel(sc, grow, win, phi > t_read + 0.6, a,
                           clock=tx["clock"].format(t=sim_t, T=steps * float(d["net_dt"])))
        self.vote_lines(d["contributions"][L], win, a * ease((phi - t_read - 0.4) / 0.6) * out)

        total = n_letters * L_DUR
        cues = [(0, L_DUR, tx["pixels"]), (L_DUR, 2 * L_DUR, tx["votes"]), (2 * L_DUR, 4 * L_DUR, tx["untrained"]),
                (4 * L_DUR, total + 1, tx["accuracy"].format(acc=100 * self.meta["accuracy"]))]
        for c0, c1, s in cues:
            self.caption(s, alpha=a * fade(t, c0, c1, 0.4))

    def scene_outro(self, t):
        a = fade(t, 0, T_OUTRO + 1, 0.8)
        self.draw_network(self.d["h_off"], alpha=0.15 * a)
        self.text(W / 2, 640, self.tx["end1"], size=34, alpha=fade(t, 0.3, T_OUTRO + 1, 0.6))
        self.text(W / 2, 570, self.tx["end2"], size=34, color=ACCENT, alpha=fade(t, 1.6, T_OUTRO + 1, 0.6))
        self.text(W / 2, 430, self.tx["end3"], size=44, alpha=fade(t, 3.0, T_OUTRO + 1, 0.6), weight="bold")

    # --- frame assembly -------------------------------------------------------------
    def setup(self, scale: float):
        self.fig = plt.figure(figsize=(W / 100, H / 100), dpi=100 * scale, facecolor=BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1])

    def draw(self, t: float) -> np.ndarray:
        ax = self.ax
        ax.cla()
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_facecolor(BG)
        ax.axis("off")
        if t < self.t_single:
            self.scene_title(t)
        elif t < self.t_intro:
            self.scene_single(t - self.t_single)
        elif t < self.t_letters:
            self.scene_intro(t - self.t_intro)
        elif t < self.t_outro:
            self.scene_letters(t - self.t_letters)
        else:
            self.scene_outro(t - self.t_outro)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]


def render_chunk(args):
    lang, scale, f0, f1, path = args
    v = Video(lang)
    v.setup(scale)
    w, h = int(W * scale), int(H * scale)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        proc.stdin.write(np.ascontiguousarray(v.draw(f / FPS)).tobytes())
    proc.stdin.close()
    return proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=sorted(TEXTS))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--stills", default="")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.stills:
        v = Video(args.lang)
        v.setup(args.scale)
        (OUT / "stills").mkdir(exist_ok=True)
        for s in args.stills.split(","):
            img = v.draw(float(s))
            plt.imsave(OUT / "stills" / f"still_{args.lang}_{float(s):06.2f}.png", img)
        print(f"duration {v.duration:.1f} s")
        return
    v = Video(args.lang)
    frames = int(round(v.duration * FPS))
    bounds = np.linspace(0, frames, args.workers + 1).astype(int)
    chunks = [(args.lang, args.scale, int(a), int(b), BUILD / f"chunk_{args.lang}_{i:02d}.mp4")
              for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
    with ProcessPoolExecutor(args.workers) as pool:
        codes = list(pool.map(render_chunk, chunks))
    if any(codes):
        raise SystemExit(f"ffmpeg failed: {codes}")
    listing = BUILD / f"chunks_{args.lang}.txt"
    listing.write_text("".join(f"file '{c[4]}'\n" for c in chunks))
    suffix = "" if args.scale == 1 else f"_{args.scale:g}x"
    out = OUT / f"siphonnet_{args.lang}{suffix}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    print(f"{out}  ({frames} frames, {v.duration:.1f} s)")


if __name__ == "__main__":
    main()
