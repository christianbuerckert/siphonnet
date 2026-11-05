"""A 5x7 pixel font: every lit pixel is one pump."""

from __future__ import annotations

import numpy as np

_FONT = """
A .###. #...# #...# ##### #...# #...# #...#
B ####. #...# #...# ####. #...# #...# ####.
C .###. #...# #.... #.... #.... #...# .###.
D ####. #...# #...# #...# #...# #...# ####.
E ##### #.... #.... ####. #.... #.... #####
F ##### #.... #.... ####. #.... #.... #....
G .###. #...# #.... #.### #...# #...# .####
H #...# #...# #...# ##### #...# #...# #...#
I .###. ..#.. ..#.. ..#.. ..#.. ..#.. .###.
J ..### ...#. ...#. ...#. ...#. #..#. .##..
K #...# #..#. #.#.. ##... #.#.. #..#. #...#
L #.... #.... #.... #.... #.... #.... #####
M #...# ##.## #.#.# #.#.# #...# #...# #...#
N #...# #...# ##..# #.#.# #..## #...# #...#
O .###. #...# #...# #...# #...# #...# .###.
P ####. #...# #...# ####. #.... #.... #....
Q .###. #...# #...# #...# #.#.# #..#. .##.#
R ####. #...# #...# ####. #.#.. #..#. #...#
S .#### #.... #.... .###. ....# ....# ####.
T ##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..
U #...# #...# #...# #...# #...# #...# .###.
V #...# #...# #...# #...# #...# .#.#. ..#..
W #...# #...# #...# #.#.# #.#.# #.#.# .#.#.
X #...# #...# .#.#. ..#.. .#.#. #...# #...#
Y #...# #...# .#.#. ..#.. ..#.. ..#.. ..#..
Z ##### ....# ...#. ..#.. .#... #.... #####
"""

GLYPHS: dict[str, np.ndarray] = {}
for _line in _FONT.strip().splitlines():
    _ch, *_rows = _line.split()
    GLYPHS[_ch] = np.array([[c == "#" for c in r] for r in _rows], dtype=float)

LETTERS = "".join(GLYPHS)
ROWS, COLS = 7, 5


def noisy(letter: str, rng: np.random.Generator, flips: int = 3, pump_tolerance: float = 0.2) -> np.ndarray:
    """Pump drive (35,) for a letter: ``flips`` random pixels inverted, each pump ±tolerance."""
    g = GLYPHS[letter].ravel().copy()
    idx = rng.choice(g.size, size=flips, replace=False)
    g[idx] = 1.0 - g[idx]
    return g * rng.uniform(1 - pump_tolerance, 1 + pump_tolerance, g.size)
