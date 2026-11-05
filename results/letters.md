# Letters A–Z (5×7 pixels, 3 flipped, ±20 % pump tolerance; 40 train / 20 test per letter, 3 seeds)

Test accuracy in %. Features: pulse counts and mean levels in 3 windows of a 30 s trial. 'linear on pixels' does not depend on N.

| model | N=36 | N=64 | N=100 | N=144 |
|---|---|---|---|---|
| physical, gravity cascade | 90.5 ± 0.8 | 92.4 ± 0.9 | 94.3 ± 1.1 | 95.2 ± 0.4 |
| physical, random graph (needs pumps) | 89.2 ± 1.8 | 92.7 ± 0.9 | 94.4 ± 1.2 | 94.7 ± 0.2 |
| physical SiphonNet, W=0 | 93.0 ± 1.8 | 94.3 ± 0.4 | 96.1 ± 0.9 | 96.5 ± 0.4 |
| linear on pixels | 93.7 ± 0.2 | 93.7 ± 0.2 | 93.7 ± 0.2 | 93.7 ± 0.2 |
