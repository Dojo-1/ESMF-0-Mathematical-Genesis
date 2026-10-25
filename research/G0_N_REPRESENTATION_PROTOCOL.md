# G0-N · First real/neutral/complex representational comparison

**Question:** Does using a fixed O(2,2) transformation or a complex-unitary block encoding of the *same* four real features improve prediction when readout capacity is matched?

A tiny binary environment supplies independent latent b and action a; y = a AND b; nuisance appearance bit is ignored (deliberately disclosed). Features [1,a,b,ab] are already sufficient, making this a **representation and numerical-conditioning test**, not concept discovery. All models use 4 real trainable weights, identical training records, SGD, step count, and Euclidean L2 regularization. Elapsed time is noisy and not energy.

**Important:** the neutral boost is nonorthogonal in Euclidean norm. Although it preserves signature (2,2), L2 regularization and SGD gradients are **not covariant under it**. A difference in Brier is therefore not an intrinsic advantage of neutral-signature geometry. The complex2 encoding is just a real unitary rotation of two complex channels: without complex-valued trainable operations, it is not a bona fide complex learning model. This test should expose rather than hide those issues.

**Prediction before observing data:** complex/Euclidean scores should be close, apart from finite SGD details; neutral differs due to conditioning. Any apparent neutral improvement should not be promoted to H1 because training procedure and regularization are not gauge-matched.

**Further work:** metric-adjusted natural gradient and properly transformed regularization for gauge parity; genuine Clifford/split-quaternion operations, ablations, multiple tasks and interventions. Only then ask about efficiency or adaptive geometry.
