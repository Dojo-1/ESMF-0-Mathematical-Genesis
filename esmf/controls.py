"""Paired G0-B topology controls with a strictly local readout.

This bounded synthetic experiment is a falsification harness, not evidence of
semantic emergence. Diagnostics are computed after learning and never fed back
to the learner.
"""
import argparse
import json
import math
import random
from pathlib import Path


N_REGIONS = 12
READOUT = 1
PROPOSAL_SENDERS = ((READOUT - 1) % N_REGIONS, (READOUT + 1) % N_REGIONS)
FEATURES = 9
EXPLORATION_PERIOD = 20
STRATEGIES = ("fixed_informative", "random_one_edge", "scripted_edge",
              "residual_driven")
COMPOSITIONS = ("and", "xor")
INTERVENTIONS = ("observe", "do_a", "do_b")


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def sample(rng, composition="and", intervention="observe", appearance=None):
    """Create local observations and an outcome; latent causes stay hidden."""
    if composition not in COMPOSITIONS or intervention not in INTERVENTIONS:
        raise ValueError("unknown composition or intervention")
    latent = rng.randrange(2)
    a = latent ^ int(rng.random() < 0.2)
    b = latent ^ int(rng.random() < 0.2)
    if intervention == "do_a":
        a = rng.randrange(2)
    elif intervention == "do_b":
        b = rng.randrange(2)
    if appearance is None:
        appearance = rng.randrange(2)
    if appearance not in (0, 1):
        raise ValueError("appearance must be binary")

    cues = [rng.randrange(2) for _ in range(N_REGIONS)]
    cues[0] = a ^ appearance
    cues[READOUT] = b ^ appearance
    label = int((a and b) if composition == "and" else (a != b))
    return cues, label, appearance


def _pairs():
    return [(i, j) for i in range(N_REGIONS) for j in range(i + 1, N_REGIONS)]


class LocalReadout:
    """One-edge learner whose output sees only local state and received messages."""

    def __init__(self, strategy, seed):
        if strategy not in STRATEGIES:
            raise ValueError("unknown topology strategy")
        self.strategy = strategy
        self.rng = random.Random(seed)
        pairs = _pairs()
        random_edge = random.Random(seed + 50000).choice(pairs)
        initial = {
            "fixed_informative": (0, READOUT),
            "random_one_edge": random_edge,
            "scripted_edge": (0, READOUT),
            "residual_driven": random_edge,
        }[strategy]
        self.edges = {tuple(sorted(initial))}
        self.weights = [0.0] * FEATURES
        self.residual_scores = {i: 0.0 for i in PROPOSAL_SENDERS}
        self.proposal_messages = {i: 0.0 for i in PROPOSAL_SENDERS}
        self.learning_rate = 0.12
        self.updates = 0
        self.edge_exposures = 0
        self.proposal_events = 0
        self.candidate_evaluations = 0
        self.message_count = 0

    def _local_features(self, cues, appearance):
        received = 0.0
        has_message = 0.0
        for a, b in self.edges:
            if a == READOUT:
                received, has_message = float(cues[b]), 1.0
            elif b == READOUT:
                received, has_message = float(cues[a]), 1.0
        local_b = float(cues[READOUT])
        style = float(appearance)
        return [
            1.0, local_b, received, style, local_b * received,
            received * style, local_b * style,
            local_b * received * style, has_message,
        ], has_message

    def predict(self, cues, appearance):
        features, has_message = self._local_features(cues, appearance)
        score = sum(w * x for w, x in zip(self.weights, features))
        return sigmoid(score), has_message

    def _explore(self):
        senders = list(PROPOSAL_SENDERS)
        scores = [self.proposal_messages[i] for i in senders]
        self.candidate_evaluations += len(PROPOSAL_SENDERS)
        self.proposal_events += 1
        if self.strategy == "residual_driven":
            best_score = max(scores)
            source = self.rng.choice([sender for sender, score in zip(senders, scores)
                                      if score == best_score])
            self.edges = {tuple(sorted((source, READOUT)))}
        elif self.strategy == "scripted_edge":
            source = senders[(self.proposal_events - 1) % len(senders)]
            self.edges = {tuple(sorted((source, READOUT)))}

    def update(self, cues, label, appearance):
        prediction, has_message = self.predict(cues, appearance)
        error = label - prediction
        features, _ = self._local_features(cues, appearance)
        for i, value in enumerate(features):
            self.weights[i] = max(-4.0, min(4.0,
                                          self.weights[i] + self.learning_rate * error * value))
        for i in PROPOSAL_SENDERS:
            cue = cues[i]
            self.residual_scores[i] = (0.95 * self.residual_scores[i]
                                       + error * (float(cue) - 0.5))
            self.proposal_messages[i] = abs(self.residual_scores[i])
        self.updates += 1
        self.edge_exposures += len(self.edges)
        self.message_count += int(has_message)
        if self.updates % EXPLORATION_PERIOD == 0:
            self._explore()
        return (prediction - label) ** 2

    def valid(self):
        if len(self.edges) != 1 or any(not 0 <= a < b < N_REGIONS
                                       for a, b in self.edges):
            return False
        return (len(self.weights) == FEATURES
                and all(math.isfinite(x) and -4.0 <= x <= 4.0
                        for x in self.weights)
                and all(math.isfinite(x) for x in self.residual_scores.values())
                and all(math.isfinite(x) for x in self.proposal_messages.values())
                and self.edge_exposures == self.updates)


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _variance(values):
    mean = _mean(values)
    return _mean([(x - mean) ** 2 for x in values])


def observation_gram(observations):
    """External uncentered observation Gram; never called during learning."""
    if not observations or any(len(row) != N_REGIONS for row in observations):
        raise ValueError("observations must be a nonempty region-by-sample matrix")
    count = float(len(observations))
    return [[sum(row[i] * row[j] for row in observations) / count
             for j in range(N_REGIONS)] for i in range(N_REGIONS)]


def signed_laplacian(edge, observations):
    """External signed Laplacian, with edge sign from observed covariance."""
    if not observations or any(len(row) != N_REGIONS for row in observations):
        raise ValueError("observations must be a nonempty region-by-sample matrix")
    a, b = edge
    if not 0 <= a < b < N_REGIONS:
        raise ValueError("invalid edge")
    means = [_mean(row[i] for row in observations) for i in range(N_REGIONS)]
    covariance = _mean([(row[a] - means[a]) * (row[b] - means[b])
                        for row in observations])
    sign = -1.0 if covariance < 0 else 1.0
    matrix = [[0.0] * N_REGIONS for _ in range(N_REGIONS)]
    matrix[a][a] = matrix[b][b] = 1.0
    matrix[a][b] = matrix[b][a] = -sign
    return matrix


def _evaluate(model, examples):
    predictions = []
    labels = []
    message_count = 0
    for cues, label, appearance in examples:
        prediction, has_message = model.predict(cues, appearance)
        predictions.append(prediction)
        labels.append(label)
        message_count += int(has_message)
    return {
        "n": len(examples),
        "brier": round(_mean([(p - y) ** 2 for p, y in zip(predictions, labels)]), 7),
        "prediction_variance": round(_variance(predictions), 10),
        "label_classes": sorted(set(labels)),
        "message_rate": round(message_count / len(examples), 7),
    }


def _diagnostics(model, observations):
    """Run after training; these values are deliberately not model inputs."""
    return {
        "observation_gram": observation_gram(observations),
        "signed_laplacian": signed_laplacian(next(iter(model.edges)), observations),
        "used_for_learning": False,
    }


def paired(seed, episodes=600, eval_per_stratum=40):
    if not 1 <= episodes <= 100000 or not 1 <= eval_per_stratum <= 1000:
        raise ValueError("episodes or evaluation budget out of range")
    train_rng = random.Random(seed + 10000)
    training = [sample(train_rng) for _ in range(episodes)]

    test_rng = random.Random(seed + 20000)
    holdout = {}
    for composition in COMPOSITIONS:
        for intervention in INTERVENTIONS:
            for appearance in (0, 1):
                key = f"{composition}/{intervention}/appearance-{appearance}"
                holdout[key] = [
                    sample(test_rng, composition, intervention, appearance)
                    for _ in range(eval_per_stratum)
                ]
    diagnostic_observations = [
        sample(test_rng, "and", "observe", appearance)[0]
        for appearance in (0, 1)
        for _ in range(eval_per_stratum)
    ]

    results = {}
    for strategy in STRATEGIES:
        model = LocalReadout(strategy, seed)
        for cues, label, appearance in training:
            model.update(cues, label, appearance)
            assert model.valid(), "topology, contact, or parameter invariant violated"

        groups = {key: _evaluate(model, examples) for key, examples in holdout.items()}
        all_examples = [example for examples in holdout.values() for example in examples]
        overall = _evaluate(model, all_examples)
        appearance_gaps = {
            f"{composition}/{intervention}": round(abs(
                groups[f"{composition}/{intervention}/appearance-0"]["brier"]
                - groups[f"{composition}/{intervention}/appearance-1"]["brier"]
            ), 7)
            for composition in COMPOSITIONS for intervention in INTERVENTIONS
        }
        state_scalars = (len(model.weights) + len(model.residual_scores)
                         + len(model.proposal_messages) + 2)
        results[strategy] = {
            **overall,
            "groups": groups,
            "appearance_brier_gaps": appearance_gaps,
            "training_message_rate": round(model.message_count / episodes, 7),
            "edges": [list(edge) for edge in sorted(model.edges)],
            "active_contacts": len(model.edges),
            "active_contact_exposures": model.edge_exposures,
            "updates": model.updates,
            "exploration_events": model.proposal_events,
            "candidate_evaluations": model.candidate_evaluations,
            "logical_operation_units_estimate": (
                episodes * (3 * FEATURES + 3 * len(PROPOSAL_SENDERS))
                + model.candidate_evaluations * 3
            ),
            "state_scalars": state_scalars,
            "memory_bytes_estimate": state_scalars * 8,
            "diagnostics": _diagnostics(model, diagnostic_observations),
        }
    return {
        "seed": seed,
        "episodes": episodes,
        "eval_per_stratum": eval_per_stratum,
        "matched_budget": {
            "updates_per_strategy": episodes,
            "active_contacts_per_update": 1,
            "exploration_events_per_strategy": episodes // EXPLORATION_PERIOD,
            "candidate_evaluations_per_event": len(PROPOSAL_SENDERS),
        },
        "training_distribution": "AND, observational",
        "holdout_compositions": list(COMPOSITIONS),
        "holdout_interventions": list(INTERVENTIONS),
        "z_plus": {
            "implemented": False,
            "status": "deferred: no separate possibility/probability channels or specificity gate",
        },
        "models": results,
    }


def _summary(rows):
    return {
        strategy: {
            "mean_brier": round(_mean([row["models"][strategy]["brier"] for row in rows]), 7),
            "paired_delta_vs_fixed_informative": round(_mean([
                row["models"][strategy]["brier"]
                - row["models"]["fixed_informative"]["brier"] for row in rows
            ]), 7),
        }
        for strategy in STRATEGIES
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--eval-per-stratum", type=int, default=40)
    parser.add_argument("--output", help="optional path for the full JSON record")
    args = parser.parse_args()
    try:
        seeds = [int(value) for value in args.seeds.split(",")]
    except ValueError:
        parser.error("seeds must be comma-separated integers")
    if (not seeds or len(seeds) > 128 or len(set(seeds)) != len(seeds)
            or not 1 <= args.episodes <= 100000
            or not 1 <= args.eval_per_stratum <= 1000):
        parser.error("require 1-128 unique seeds, 1-100000 episodes, and 1-1000 eval samples")
    rows = [paired(seed, args.episodes, args.eval_per_stratum) for seed in seeds]
    output = {
        "protocol": "G0-B matched local topology falsification (no semantic claim)",
        "resource_governor": {
            "execution": "standard-library CPU only",
            "remote_job_scheduled": False,
            "local_kaggle_accessed": False,
        },
        "paired_seed_summary": _summary(rows),
        "runs": rows,
        "claim_gate": "no semantic-emergence promotion without labeled, fixed-size, and randomized controls",
    }
    rendered = json.dumps(output, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
