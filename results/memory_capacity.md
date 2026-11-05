# Linear memory capacity (N=100, 2500 ticks, delays 1..40, 3 seeds)

Features per tick: `levels` = mean fill level of every container (N), `both` = levels + discharge counts (2N), `states` = ESN activations (N). Abstract model: 4 steps per tick. Physical model: 2 s per tick unless stated.

| model | features | MC | delay at which r² < 0.5 |
|---|---|---|---|
| ESN (tanh) | states | 19.0 ± 1.0 | 20 |
| abstract SiphonNet | levels | 0.2 ± 0.0 | 1 |
| abstract SiphonNet | both | 0.3 ± 0.1 | 1 |
| abstract SiphonNet, W=0 | levels | 1.0 ± 0.2 | 1 |
| abstract SiphonNet, W=0 | both | 1.1 ± 0.2 | 2 |
| physical SiphonNet | levels | 1.2 ± 0.3 | 1 |
| physical SiphonNet | both | 1.2 ± 0.3 | 1 |
| physical SiphonNet, W=0 | levels | 0.2 ± 0.1 | 1 |
| physical SiphonNet, W=0 | both | 0.2 ± 0.1 | 1 |
| physical, sealed (no bleed) | both | 1.5 ± 0.4 | 1 |
| physical, displacers 0-0.7 | both | 2.1 ± 0.4 | 1 |
| physical, tick 0.5 s | both | 0.3 ± 0.1 | 1 |
| physical, tick 8 s | both | 0.9 ± 0.2 | 1 |
