# Memory capacity vs. routed share (N=100, features = levels + counts, 3 seeds)

`rate` = discharges per container per tick. `never empties` = share of siphons without a new pulse in the last 200 ticks although they are well filled (saturated through-flow or stuck at overflow).

| share | abstract MC | abstract rate | physical MC | physical rate | physical never empties |
|---|---|---|---|---|---|
| 0.00 | 1.1 ± 0.2 | 0.199 | 0.2 ± 0.1 | 0.082 | 0 % |
| 0.30 | 0.7 ± 0.1 | 0.287 | 0.5 ± 0.2 | 0.111 | 0 % |
| 0.55 | 0.3 ± 0.1 | 0.444 | 1.2 ± 0.3 | 0.154 | 0 % |
| 0.80 | 0.5 ± 0.2 | 0.957 | 2.3 ± 0.3 | 0.208 | 0 % |
| 0.90 | 0.5 ± 0.1 | 1.742 | 3.3 ± 0.2 | 0.125 | 21 % |
| 1.00 | 0.0 ± 0.0 | 4.000 | 3.3 ± 0.4 | 0.067 | 50 % |
