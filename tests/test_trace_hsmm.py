"""The count-exact decoder recovers planted events and honours K exactly."""
import numpy as np

from tracesed.hsmm import decode, lognormal_lp


def _posterior(T, events, fps, hi=0.9, lo=0.05, seed=0):
    rng = np.random.default_rng(seed)
    p = np.full(T, lo) + rng.uniform(0, 0.03, T)
    for a, b in events:
        p[int(a * fps):int(b * fps)] = hi
    return p


def test_recovers_planted_events_exactly():
    fps, T = 50.0, 250
    ev = [(0.40, 1.10), (1.60, 1.90), (3.00, 4.40)]
    scores, seg = decode(_posterior(T, ev, fps), fps, K=6, bias=0.0, ev_cost=1.0)
    k = int(np.argmax(scores))
    assert k == 3
    np.testing.assert_allclose(seg(3), np.array(ev), atol=1.0 / fps)


def test_k_is_exact_when_given():
    fps, T = 50.0, 250
    ev = [(0.40, 1.10), (1.60, 1.90), (3.00, 4.40)]
    p = _posterior(T, ev, fps)
    _, seg = decode(p, fps, K=2)
    s = seg(2)
    assert s.shape == (2, 2)
    # forced to two events it keeps the two with most evidence, in order
    np.testing.assert_allclose(s, np.array([ev[0], ev[2]]), atol=1.0 / fps)
    _, seg4 = decode(p, fps, K=4)
    s4 = seg4(4)
    assert s4.shape == (4, 2) and np.all(np.diff(s4[:, 0]) > 0)
    assert np.all(s4[1:, 0] >= s4[:-1, 1])                    # non-overlapping


def test_class_channels_enforce_order():
    fps, T = 50.0, 200
    P = np.full((T, 2), 0.05)
    P[25:75, 1] = 0.9          # class 1 first
    P[120:170, 0] = 0.9        # class 0 second
    _, seg = decode(P, fps, K=2, ev_channel=np.array([1, 0]))
    np.testing.assert_allclose(seg(2), [[0.5, 1.5], [2.4, 3.4]], atol=1.0 / fps)


def test_min_gap_and_duration_prior():
    fps, T = 50.0, 200
    ev = [(0.5, 1.0), (1.04, 1.5)]                            # 40 ms apart
    p = _posterior(T, ev, fps)
    _, seg = decode(p, fps, K=2, min_gap=0.1)
    s = seg(2)
    assert s[1, 0] - s[0, 1] >= 0.1 - 1e-9
    lp = lognormal_lp(T, fps, mu=np.log(0.5), sigma=0.3)
    assert np.argmax(lp) / fps < 0.5 and lp[0] < -1e17
