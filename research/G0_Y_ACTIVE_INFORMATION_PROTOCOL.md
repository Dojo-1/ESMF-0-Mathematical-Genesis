# G0-Y · Complementary observations and resource-bounded active epistemics

Date: 2026-10-25. **Pre-registered before G0-Y code execution.** Experimental inheritance: [G0-X](results/G0_X_2026-10-25.md), neutral determinant fiber Mat₂(R), orientation twist by transposition, and its antisymmetric off-diagonal contrast D=F₀₁−F₁₀.

## First: correct an overlooked conditional assumption [P]

Let T∈{+1,−1} be a balanced hidden monodromy bit and D~N(0,v) an *unobserved* fiber contrast independent of T. For an odd-winding probe Y_o=T D+ε_o; for an even/zero-winding probe Y_e=D+ε_e, with independent centered Gaussian noises of finite strictly positive variances s_o²,s_e².

Both individual marginal laws are independent of T: I(T;Y_o)=I(T;Y_e)=0. The **joint** law differs by covariance sign:

Sigma_T = [[v+s_o², T v],[T v,v+s_e²]].

For v>0, joint observations distinguish T; explicitly
`log P(T=+1|yo,ye)/P(T=-1|yo,ye) = 2 v yo ye/[v(s_o²+s_e²)+s_o² s_e²]` for equal priors.

Thus I(T;(Y_o,Y_e))>0, and I(T;Y_e|Y_o)>0. The older G0-X corollary I(T;Y_even | F0,intervention)=0 **remains correct** under its conditional-known-fiber premise. With latent D and histories that couple D and T, an even measurement can be essential. Neither the odd nor even channel is intrinsically informative in isolation.

More sharply: any sequence using only one parity is non-identifying under symmetric D prior, regardless of its length and independent Gaussian noise. Mixed parity is necessary here, not sufficient if D≈0 or noise overwhelms it.

## Data-generating process [A]

A minimal embodied boundary proxy with TWO scalar sensors, not a privileged 4D frame:
- a hidden fixed-per-episode D sampled from N(0,1);
- hidden balanced T∈{±1};
- somatic/locomotor probe parity 0 or 1, standing for even or odd winding in an orientation bundle;
- one extracellular scalar sensor y=s(T,parity)D+N(0,σ²), not F0 or whole tangent bundle;
- enteric-like finite local energy reserve (initial 5.0, floor 0.25), with bounded charges for every action, observation and maintenance;
- autonomic-like quiet/focused sensing, changing observation variance at a real resource cost;
- posterior cognition as an exact two-component Gaussian mixture over (T,D) with explicit online conditioning, not an LLM;
- no real neural tissue, environmental motion or physical power measurement.

σ varies with sensing mode and the current reserve according to a declared model; the controller knows its own energy state and noise law, but not T or D. Ground truth is withheld until evaluation.

## Competing policies and targets

Common finite probe menu: even/quiet, even/focus, odd/quiet, odd/focus, with deterministic resource charges and known noise levels.

- **Anticipatory:** when all single-probe information gains vanish, use expected two-probe mutual information per combined cost; otherwise choose positive one-step information gain per cost. Bayes uncertainty is genuine and can cause abstention.
- **Greedy stop:** myopic expected single-probe gain, stops without probing if all candidates have zero gain.
- **Alternating:** fixed even/odd alternation with quiet or focused measurement, same posterior and stopping rule.
- **Random:** uniformly chosen affordable probes with same posterior, criterion and resource ceiling.
- **Odd-only / even-only:** negative identifiability controls.
- **Conventional real Bayesian baseline:** same likelihood, posterior and anticipatory policy in ordinary scalar arrays; *must tie* any geometric reading. Here the geometry only supplies the parity/contrast interpretation.
- **Oracle latent D (analytic only):** identify twist after odd-winding observations when D is supplied; not a fair practical comparator.

Metrics: true hidden-class selective accuracy; classification coverage; false confident choices; posterior Brier and log score; mean energy spent; mean number and types of probes; total abstentions; mean entropy reduction. Reject any efficiency claim if higher classification quality comes solely from more energy or privileged observations.

## Numerical and logical falsifiers

1. For a symmetric latent prior, posterior P(T) must remain exactly 1/2 (within numerical error) after arbitrary all-even or all-odd sequences.
2. Two opposite-parity Gaussian observations must reproduce the exact analytical posterior log-odds above.
3. Conditioning on a *known* contrast D makes even probes uninformative about T as per original theorem; unknown D opens synergistic information.
4. Conditional two-step mutual information must be positive for mixed parity and zero for same parity at the initial symmetric belief.
5. Posterior normalization, variance positivity, energy nonnegativity, reproducibility, Gaussian quadrature convergence; no confidence declaration if evidence absent.
6. If anticipatory policy fails to beat the one-step-stop policy on coverage or calibrated quality at matched energy, no proactive-epistemic advantage.
7. If naive fixed alternation ties or dominates the lookahead policy, record null: optimum may be a simple circuit.
8. No unique advantage of fiber bundles or reduced hardware energy may be inferred against an equivalently informed real Bayesian controller.
9. Resource costs must include focus sensor cost, not only probe motion; no artificial infinite nutrient restoration.
10. Balance results over both hidden T classes and across independently drawn D; report class-conditioned findings and narrow uncertainty.

## Evidence status
[P] Conditional Gaussian identifiability and covariance calculation under explicit assumptions. [A] Surrogate organs and sensory/metabolic laws. [M] Only after remote execution of frozen tests. [H] Adaptive experimental design produces useful generalization or compute-energy efficiency in richer worlds. [S] 'Maieutic organism' is architectural language, not sentience.
