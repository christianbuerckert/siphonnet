# Letters A–Z (5×7 pixels, 3 flipped, ±20 % pump tolerance; 40 train / 20 test per letter, 3 seeds), cascade with 5 rows

Test accuracy in %. Features: pulse counts and mean levels in 3 windows of a 30 s trial. 'linear on pixels' does not depend on N.

| model | N=100 |
|---|---|
| physical, gravity cascade | 95.5 ± 0.2 |
| physical, random graph (needs pumps) | 94.4 ± 1.2 |
| physical SiphonNet, W=0 | 96.1 ± 0.9 |
| linear on pixels | 93.7 ± 0.2 |
