"""Paired G0-B control: same informative edge, fixed from the start.

This is a falsification control, NOT a test of endogenous growth.
The matched fixed graph has the edge throughout training, while the
scripted-growth model acquires it later. Active final degree is matched;
training exposure to the edge is NOT matched.
"""
import argparse
import json
import random
from .experiment import LocalLearner, sample


def paired(seed, episodes=600):
    train_rng = random.Random(seed + 10000)
    test_rng = random.Random(seed + 20000)
    train = [sample(train_rng) for _ in range(episodes)]
    test = [sample(test_rng) for _ in range(240)]
    result = {}
    for name in ("fixed_empty", "fixed_matched", "scripted_growth"):
        model = LocalLearner(name == "scripted_growth", seed)
        if name == "fixed_matched":
            model.edges[0].add(1)
            model.edges[1].add(0)
        for cues, label in train:
            model.update(cues, label)
            assert model.valid()
        brier = sum((model.predict(c)[0] - y) ** 2 for c, y in test) / len(test)
        result[name] = {"brier": round(brier, 7),
                        "edges": sum(map(len, model.edges.values())) // 2}
    return {"seed": seed, **result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--episodes", type=int, default=600)
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if not seeds or len(seeds) > 128 or not 1 <= args.episodes <= 100000:
        parser.error("require 1-128 seeds and 1-100000 episodes")
    rows = [paired(s, args.episodes) for s in seeds]
    means = {key: round(sum(row[key]["brier"] for row in rows)/len(rows), 7)
             for key in ("fixed_empty", "fixed_matched", "scripted_growth")}
    print(json.dumps({"protocol": "G0-B edge-parity falsification", "means": means,
                      "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
