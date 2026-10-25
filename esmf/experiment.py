"""CPU-only, dependency-free falsification baseline.

The local learner predicts a binary consequence from two distributed cues.
A dynamic edge can be recruited by local residual correlation. A fixed
network has the same maximum edge budget but cannot recruit new contacts.
Both models use identical exposure and evaluation sequences. This is a
toy associative-routing test, NOT evidence of concept emergence.
"""
import argparse
import json
import math
import random


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class LocalLearner:
    def __init__(self, adaptive, seed, n=12, degree=3):
        self.adaptive = adaptive
        self.n = n
        self.degree = degree
        self.rng = random.Random(seed)
        # Each region sees only its own cue. Regions 0 and 1 are informative.
        self.edges = {i: set() for i in range(n)}
        self.w = [0.0] * n
        self.bias = 0.0
        self.lr = 0.12
        self.encounters = 0
        self.resource = 0.0

    def observe(self, cues):
        """Messages are received only over currently instantiated edges."""
        received = [float(v) for v in cues]
        for source in range(self.n):
            for target in self.edges[source]:
                received[target] += cues[source] / max(1, len(self.edges[source]))
        return received

    def predict(self, cues):
        x = self.observe(cues)
        return sigmoid(self.bias + sum(a * b for a, b in zip(self.w, x))), x

    def update(self, cues, label):
        pred, x = self.predict(cues)
        err = label - pred
        for i in range(self.n):
            self.w[i] = max(-4.0, min(4.0, self.w[i] + self.lr * err * x[i]))
        self.bias = max(-4.0, min(4.0, self.bias + self.lr * err))
        self.encounters += 1
        # Local proxy is deliberately weak: coincidence of cue and surprise.
        if self.adaptive and self.encounters % 20 == 0:
            i = 0 if self.encounters % 40 == 0 else 1
            candidate = 1 - i
            if (candidate not in self.edges[i]
                and len(self.edges[i]) < self.degree
                and len(self.edges[candidate]) < self.degree):
                self.edges[i].add(candidate)
                self.edges[candidate].add(i)
        self.resource = sum(len(s) for s in self.edges.values()) / 2.0
        return (pred - label) ** 2

    def valid(self):
        return (all(len(v) <= self.degree for v in self.edges.values())
                and all(0 <= j < self.n and i in self.edges[j]
                        for i, adj in self.edges.items() for j in adj)
                and 0 <= self.resource <= self.n * self.degree / 2)


def sample(rng, n=12):
    # Hidden causal rule: logical AND; surface distractors independent.
    a, b = rng.randrange(2), rng.randrange(2)
    cues = [0] * n
    cues[0], cues[1] = a, b
    for i in range(2, n):
        cues[i] = rng.randrange(2)
    return cues, int(a and b)


def run(seed, episodes=600):
    training_rng = random.Random(seed + 10000)
    eval_rng = random.Random(seed + 20000)
    train = [sample(training_rng) for _ in range(episodes)]
    holdout = [sample(eval_rng) for _ in range(240)]
    result = {}
    for name, adaptive in (("fixed", False), ("dynamic", True)):
        model = LocalLearner(adaptive=adaptive, seed=seed)
        for x, y in train:
            model.update(x, y)
            assert model.valid(), "connectivity/resource invariant violated"
        loss = sum((model.predict(x)[0] - y) ** 2 for x, y in holdout) / len(holdout)
        result[name] = {"brier": round(loss, 7),
                        "edges": int(model.resource), "valid": model.valid()}
    return {"seed": seed, "episodes": episodes, **result,
            "delta_dynamic_minus_fixed":
                round(result["dynamic"]["brier"] - result["fixed"]["brier"], 7)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--episodes", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 100000:
        parser.error("episodes must be in [1, 100000]")
    seeds = [int(s) for s in args.seeds.split(",")]
    if len(seeds) > 128:
        parser.error("max 128 seeds")
    rows = [run(seed, args.episodes) for seed in seeds]
    print(json.dumps({"protocol": "G0 baseline, no semantic claim",
                      "runs": rows, "mean_delta": round(sum(
                          x["delta_dynamic_minus_fixed"] for x in rows) / len(rows), 7)},
                     indent=2))


if __name__ == "__main__":
    main()
