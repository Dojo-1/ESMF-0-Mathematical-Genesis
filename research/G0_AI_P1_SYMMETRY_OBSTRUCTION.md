# G0-AI P1 · An exact obstruction to Genesis by growth alone

**Status:** P, conditional algebraic/software statement for the finite G0-AI operators; M for an exhaustive four-arm counterworld regression *at its tested seeds and horizon*; H for the proposed generative escape. No claim about biological neural tissue, AI consciousness or asymptotic intelligence.

## The overlooked invariant

The G0-AI D0 experiment [G0_AI_D0_RESULTS.md](G0_AI_D0_RESULTS.md) gave a counterintuitive result: locally differentiated regulatory coefficients, stochastic contact growth, lost-contact provenance and closed integer resource ledgers all occurred, yet their adaptive tissue did worse than its fixed geometric reference. It might be tempting to add more neurites, algebraic dimensions or regulome coefficients. Before that, we inspected an exact symmetry preserved by all present primitives.

Let the external stimulus be a history \(X=(x_1,\ldots,x_t)\), with bipolar inputs \(x_k\in\{-1,+1\}^4\). The exchangeable local sites have signed activities \(h_i\), signed fast/slow traces \(f_i,s_i\), and *neutral* quantities consisting of their plasticity rates, gain, regulatory mixture weights, lagged pair correlations, directed contact weights and fitness, topology, resource ledger, typed-loss log and random generator states. There is a common involution

\[
\mathcal J: (X,h,f,s,v)\mapsto(-X,-h,-f,-s,-v),
\]

where \(v\) is the external readout weight vector; the readout intercept \(b\), all neutral tissue variables and the externally witnessed labels \(y\) are unchanged.

### Proposition P1 (conditional sign-equivariance)

For two G0-AI runs with identical initial local projections, coordinates, contacts and PRNG states, receiving the stimuli \(X\) and \(-X\), but the *same sequence of externally witnessed labels*, the resulting states are related at every step by \(\mathcal J\). Their **predictions are identical**, as are all topology decisions, energetic and material balances and typed contact remainders. This holds for `adaptive`, `plastic_no_growth`, `fixed` and `shuffled` modes.

**Proof (induction over updates).** At zero, all signed local activities and traces and readout weights are zero. Suppose the relation holds at step \(t-1\). Because tanh is odd, the local response \(h_i'=\tanh(g_i(\langle a_i,x_t\rangle+\sum_j w_{ij}h_j))\) reverses sign while its gain, projections and edge weights remain identical. Its absolute novelty, neighbor disagreement magnitude and resource pressure are even; hence the slow regulome, mixture coefficients, gain and learning rates are identical. Each lagged correlation \(h_i(t)h_j(t-1)\) is even, so synaptic adaptation, contact ranking, event hazard, pruning and resource updates coincide. Both runs consume the same pseudorandom draws and preserve the same typed remainders. In the external readout \(\sigma(b+\sum_i v_i h_i)\), the weights and features both reverse, giving exactly the same prediction. Its post-label update \(v_i\gets v_i+\alpha(y-p)h_i\) likewise reverses, and its intercept update does not. This closes induction. \(\square\)

The finite Python regression [tests/test_g0_ai_symmetry.py](../tests/test_g0_ai_symmetry.py) verifies all these equivalences, including morphology, regulatory weights and predictions, under paired counterworlds over 180 updates for all four arms. This regression is **not a formal theorem checker**; the proof above is conditional on the exact stated update equations.

### Corollary (odd-feature readout cannot express a nonconstant even target exactly)

Fix a tissue representation \(\phi\) for which \(\phi(-X)=-\phi(X)\), and use a logistic affine readout \(F(X)=\sigma(b+v^\top\phi(X))\). If \(F(X)=F(-X)\) for every sign-paired history, injectivity of \(\sigma\) implies \(v^\top\phi(X)=0\) for every history, so \(F\) is constant. Therefore such a readout cannot exactly express a *nonconstant* sign-invariant target over the whole sign-paired domain. In particular the two-input parity target of G0-AI D0 is even under simultaneous sign reversal, whereas the accessible tissue features are odd.

This **does not prove** that sign-equivariance accounts for the entire measured Brier deficit: the source world has a finite trajectory, a nuisance Markov bit and changing latent phases; stochastic finite-sample readout learning need not realize the population optimum. Nor does it prove that all recurrent neural networks, adaptive membranes or hypercomplex architectures share this obstruction. It identifies a specific missing parity class in *our* present representation.

## A more productive mathematical next move

Instead of manually making an XOR neuron, let local interactions have access to a graded, bounded composition *closure*:

\[
V=V_{\rm even}\oplus V_{\rm odd}, \qquad
V_{\rm odd}\otimes V_{\rm odd}\longrightarrow V_{\rm even}.
\]

A co-observation of two odd signals can form an even local relation. The system may propose, transport, retain or prune such **typed relational sections** under resource and provenance constraints, rather than having an engineer install a privileged parity detector. Do not force every local Worldlet into the same carrier; a conventional real-valued second-order feature map and a compute-matched recurrent baseline must be matched controls, not dismissed as flat.

One must distinguish: (a) physical feasibility of a local composition, (b) its evidence of temporal usefulness, (c) whether it becomes durable without being prescribed an anatomical role, (d) its transport/attribution remainder, and (e) whether it improves genuine future behavior under fair budgets. New typed operators need an independent instrument and cannot promote their own evidence class.

A fundamental engineering trade-off remains. Letting an organism generate every mathematically possible composition without a capacity/morphological cost causes trivial combinatorial explosion; fixing the only admissible relations in advance recreates G0-AH's limitation. A resource-bounded *proposal–consolidation–retirement ecology*, with generic local relation composition and typed witnesses, is the research hypothesis **G0-AJ**, not an achieved cognitive property.

**Decision:** retire (R) the claim that greater neuritic growth *within G0-AI's strictly odd observation algebra* can by itself solve parity-like even relational tasks. Preserve the finite D0 negative experiment, the exact P1 counterworld, and this open compositional remainder as separate pieces of evidence.
