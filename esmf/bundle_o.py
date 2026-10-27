"""G0-O matched causal-sensory controls for flat and curved connections.

The bounded synthetic task checks transport arithmetic and matched work only;
it is not evidence that curvature improves learning or cognition.
"""
import json
import random

from .bundle_p import (
    commutator, exponential, generators, identity, matmul, matvec, maxdiff,
    plus, scaled,
)
from .causal_bundle_s import path, physical_world, wrap

D = 4
TERMS = 24


def fixed_exponential(A):
    """Fixed-work Taylor approximation to exp(A), shared by both controls."""
    result = identity(D)
    term = identity(D)
    for k in range(1, TERMS + 1):
        term = scaled(matmul(term, A), 1. / k)
        result = plus(result, term)
    return result


def sensory_paths(seed, episodes=12):
    rng = random.Random(seed)
    params = (.61, -.52)
    traces = []
    for _ in range(episodes):
        state = (rng.uniform(-1.5, 1.5), rng.uniform(-.7, .7))
        actions = path(rng, rng.randint(4, 8))
        segments = []
        for action in actions:
            later = physical_world(state, (action,), params, "noncommuting")
            segments.append((wrap(later[0] - state[0]), later[1] - state[1]))
            state = later
        traces.append(tuple(segments))
    return tuple(traces)


def transport(segments, vector, connection):
    Ax, Ay = connection
    result = identity(D)
    for dx, dy in segments:
        for amount, A in ((dx, Ax), (dy, Ay)):
            step = fixed_exponential(scaled(A, -amount))
            result = matmul(step, result)
    return matvec(result, vector)


def positive_error(actual, expected):
    return sum((a - b) ** 2 for a, b in zip(actual, expected))


def compare_controls(seed=721991, episodes=12):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    X, Y = generators()
    controls = {"flat": (X, X), "curved": (X, Y)}
    flat_curvature = maxdiff(commutator(*controls["flat"]),
                             scaled(identity(D), 0.))
    curved_curvature = maxdiff(commutator(*controls["curved"]),
                               scaled(identity(D), 0.))
    traces = sensory_paths(seed, episodes)
    initial = (.3, -.2, .7, 1.)
    errors = {}
    steps = 0
    action_cost = 0.
    for trace in traces:
        steps += len(trace)
        action_cost += sum(dx * dx + dy * dy for dx, dy in trace)
    for world, truth in controls.items():
        targets = [transport(trace, initial, truth) for trace in traces]
        errors[world] = {}
        for name, candidate in controls.items():
            errors[world][name] = sum(
                positive_error(transport(trace, initial, candidate), target)
                for trace, target in zip(traces, targets)
            ) / episodes
    # Each candidate executes two fixed 24-term exponentials and two matrix
    # products per observed base segment, plus one final vector product.
    operations = steps * (2 * TERMS + 2)
    return {
        "task": "causal sensory paths to local-vector transport",
        "seed": seed,
        "episodes": episodes,
        "sensory_segments": steps,
        "identical_sensory_paths": True,
        "equal_matrix_work": True,
        "nonnegative_sensory_action_cost": action_cost,
        "4x4_matrix_multiplications_per_candidate": operations,
        "flat_curvature_norm": flat_curvature,
        "curved_curvature_norm": curved_curvature,
        "mean_squared_transport_error": errors,
        "exponential_truncation_gap": maxdiff(
            fixed_exponential(scaled(Y, -.7)), exponential(scaled(Y, -.7))),
        "claim_boundary": "matched synthetic transport control; no learning or cognition claim",
    }


def selfcheck():
    result = compare_controls()
    error = result["mean_squared_transport_error"]
    assert result["flat_curvature_norm"] == 0.
    assert result["curved_curvature_norm"] > .1
    assert result["4x4_matrix_multiplications_per_candidate"] > 0
    assert error["flat"]["flat"] < 1e-24
    assert error["curved"]["curved"] < 1e-24
    assert error["flat"]["curved"] > 1e-8
    assert error["curved"]["flat"] > 1e-8
    assert result["exponential_truncation_gap"] < 1e-12
    return result


if __name__ == "__main__":
    print(json.dumps(selfcheck(), indent=2))
