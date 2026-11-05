"""Quick look at firing statistics of both models on two-moons inputs."""
import time
import numpy as np
from sklearn.datasets import make_moons
from siphonnet import *

rng = np.random.default_rng(0)
X, y = make_moons(400, noise=0.2, random_state=0)
ch = ComplementEncoder().fit_transform(X)
n = 300
W_in = random_input_matrix(ch.shape[1], n, rng)

top = random_topology(n, rng)
ab = AbstractSiphonNet(top, rng)
t0 = time.time()
rec = ab.run(pump_inflow(ch, W_in, ab.params.input_gain), hold=42, record_every=14)
rec0 = AbstractSiphonNet(top.without_recurrence(), np.random.default_rng(0)).run(pump_inflow(ch, W_in, 0.15), hold=42, record_every=14)
print(f"abstract  spikes/neuron/trial: full {rec.counts.sum(1).mean():.2f}  W=0 {rec0.counts.sum(1).mean():.2f}  silent {np.mean(rec.counts.sum(1)==0):.2f}  ({time.time()-t0:.1f}s)")

p = PhysicalParams()
top_p = random_topology(n, rng, delay_steps=(1, 5))
ph = PhysicalSiphonNet(top_p, rng, p)
steps = int(60 / p.dt)
t0 = time.time()
rec = ph.run(pump_inflow(ch, W_in, p.input_gain), hold=steps, record_every=steps // 3)
rec0 = PhysicalSiphonNet(top_p.without_recurrence(), np.random.default_rng(1), p).run(pump_inflow(ch, W_in, p.input_gain), hold=steps, record_every=steps // 3)
print(f"physical  pulses/siphon/60s: full {rec.counts.sum(1).mean():.2f}  W=0 {rec0.counts.sum(1).mean():.2f}  silent {np.mean(rec.counts.sum(1)==0):.2f}  ({time.time()-t0:.1f}s)")
print(f"pulse volume {ph.pulse_volume.mean():.0f} mL, siphon flow at h_on {ph.c_siphon*np.sqrt(8+10):.1f} mL/s, bleed at 5cm {ph.c_bleed*np.sqrt(5):.2f} mL/s")
print(f"mean pump inflow {pump_inflow(ch, W_in, p.input_gain).mean():.2f} mL/s")
