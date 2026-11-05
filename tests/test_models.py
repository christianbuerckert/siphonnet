import numpy as np
import pytest

from siphonnet import (
    AbstractParams,
    AbstractSiphonNet,
    PhysicalParams,
    PhysicalSiphonNet,
    Topology,
    random_topology,
)


def single(params=None, displacer=None):
    """One isolated physical siphon without fabrication spread."""
    p = params or PhysicalParams(area_spread=0.0, level_spread=0.0, bleed_diameter=0.0)
    top = Topology(np.zeros((1, 0), int), np.zeros((1, 0)), np.ones((1, 0), int))
    return PhysicalSiphonNet(top, np.random.default_rng(0), p, displacer=displacer)


def run_constant(net, q, seconds):
    steps = int(seconds / net.params.dt)
    inflow = np.full((1, 1, net.topology.n), q)
    return net.run(inflow, hold=steps, record_every=steps)


# --- topology -------------------------------------------------------------------------


def test_topology_has_no_self_loops_and_routes_the_share():
    top = random_topology(50, np.random.default_rng(1), out_degree=7, share=0.55)
    assert not np.any(top.targets == np.arange(50)[:, None])
    assert all(len(set(row)) == 7 for row in top.targets)
    assert np.allclose(top.fractions.sum(axis=1), 0.55)
    total = sum(M.sum(axis=1).A.ravel() for M in top.routing_matrices().values())
    assert np.allclose(total, 0.55)


def test_ablation_routes_nothing():
    top = random_topology(20, np.random.default_rng(2)).without_recurrence()
    assert top.routing_matrices() == {}


def test_share_above_one_is_rejected():
    with pytest.raises(ValueError):
        random_topology(10, np.random.default_rng(0), share=1.2)


# --- abstract model -------------------------------------------------------------------


def test_abstract_first_spike_time_matches_closed_form():
    u, leak = 0.1, 0.96
    top = Topology(np.zeros((1, 0), int), np.zeros((1, 0)), np.ones((1, 0), int))
    net = AbstractSiphonNet(top, np.random.default_rng(0), AbstractParams(leak=leak, threshold_range=(1.0, 1.0)))
    rec = net.run(np.full((1, 1, 1), u), hold=100, record_every=1)
    first = int(np.argmax(rec.counts[0, :, 0])) + 1  # steps until the first spike
    # content after t steps: u * (1 - leak^t) / (1 - leak)
    expected = int(np.ceil(np.log(1 - (1 - leak) / u) / np.log(leak)))
    assert first == expected


# --- physical model -------------------------------------------------------------------


def test_pulse_count_follows_mass_balance():
    net = single()
    q, seconds = 0.5, 6000.0  # slow filling, so discharge time is negligible
    pulses = run_constant(net, q, seconds).counts.sum()
    assert abs(pulses - q * seconds / net.pulse_volume[0]) <= 2


def test_hysteresis_drains_to_break_level_not_to_zero():
    net = single()
    steps = int(120 / net.params.dt)
    rec = net.run(np.full((1, 1, 1), 3.0), hold=steps, record_every=1)
    levels = rec.levels[0, :, 0]
    assert rec.counts.sum() >= 2
    assert levels.min() >= net.h_off[0] - 1e-9
    assert levels.max() <= net.h_on[0] + 0.1


def test_siphon_saturates_when_inflow_exceeds_its_capacity():
    net = single()
    capacity_at_break = net.c_siphon * np.sqrt(net.h_off[0] + net.params.drop)
    rec = run_constant(net, 1.2 * capacity_at_break, 300.0)
    assert rec.counts.sum() == 1  # primes once, then never breaks


def test_displacer_shrinks_pulses_and_raises_their_rate():
    q, seconds = 3.0, 600.0
    plain = run_constant(single(), q, seconds).counts.sum()
    half = single(displacer=0.5)
    assert np.isclose(half.pulse_volume[0], single().pulse_volume[0] / 2)
    assert abs(run_constant(half, q, seconds).counts.sum() - 2 * plain) <= 3


def test_bleed_hole_makes_weak_inputs_go_silent():
    p = PhysicalParams(area_spread=0.0, level_spread=0.0, bleed_diameter=0.1)
    net = single(p)
    weak = 0.5 * net.c_bleed * np.sqrt(net.h_on[0])  # bleed outruns the pump before h_on
    assert run_constant(net, weak, 600.0).counts.sum() == 0


def test_network_conserves_water():
    """With everything routed (share 1), no bleed and no overflow, water is only pumped in."""
    rng = np.random.default_rng(3)
    p = PhysicalParams(bleed_diameter=0.0, h_max=1e4)
    n, q, seconds = 30, 2.0, 300.0
    net = PhysicalSiphonNet(random_topology(n, rng, share=1.0, delay_steps=(1, 3)), rng, p)
    steps = int(seconds / p.dt)
    rec = net.run(np.full((1, 1, n), q), hold=steps, record_every=1)
    stored = rec.levels[0] @ net.area_eff  # (steps,)
    supplied = net.area_eff @ net.h_off + q * n * p.dt * np.arange(1, steps + 1)
    in_transit_max = 3 * p.dt * net.c_siphon * np.sqrt(rec.levels[0].max(axis=0) + p.drop).sum()
    assert np.all(stored <= supplied + 1e-6)
    assert np.all(stored >= supplied - in_transit_max)


def test_gravity_topology_only_flows_downhill():
    from siphonnet.topology import gravity_topology

    rows, cols = 10, 10
    top = gravity_topology(rows, cols, np.random.default_rng(4))
    src = np.repeat(np.arange(rows * cols), top.targets.shape[1]).reshape(top.targets.shape)
    active = top.fractions > 0
    assert np.all(top.targets[active] // cols > src[active] // cols)  # strictly lower rows
    assert np.all(np.abs(top.targets[active] % cols - src[active] % cols) <= 2)
    assert not np.any(active[(rows - 1) * cols :])  # bottom row drains away
    assert np.allclose(top.fractions[: (rows - 1) * cols].sum(axis=1), 0.55)
