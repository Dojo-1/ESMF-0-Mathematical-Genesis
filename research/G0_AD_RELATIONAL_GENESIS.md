# G0-AD — Endogenous relational genesis: Rosenblatt × Noether × Llull × Trasgo

Status: **implemented mathematical/behavioural contract and tests; NOT an empirical performance finding**. This branch is additive to G0-AC. No held-out G0-AD result, biological validation, semantic certification, conserved biological charge, or claimed superiority over a conventional real-valued model.

## Why this belongs inside ESMF-0

ESMF-0 asks whether a bounded, evolving organism can develop distinctions through encounters, not learn a supplied vocabulary of latent classes. The prior geometry already gives partial and loss-aware correspondence (G0-L), rank-stratified observation (G0-R), global and local seam transport (G0-Q/W/X), constrained causal learning (G0-S/T/U), resource-aware adaptive clocks (G0-AB/AC), and provenance-gated temporal repair (G0-Z/AA). Those are foundations, not proved cognition.

The missing connection is an interrogative mechanism which:
- proposes candidate equivalences rather than assigning all of them in advance;
- learns context-dependent, response-relevant relations from experience;
- never silently absorbs a witnessed behavioural difference;
- reports geometric transport defects without upgrading them into semantic truth;
- allows uncertain or contradictory evidence to motivate (but not authorize) new Worldlets.

This contribution does not replace developmental chemistry, autonomic/enteric budgets, or rank-four neutral geometry with a global scalar classifier. The present interface is deliberately bounded and does not purport to implement the still-open couplings.

## Historical and conceptual bridge

- **Frank Rosenblatt (psychological optic):** generalization is acquired by an organism through exposure, response and adaptation. The problem is not a pretrained list of geometric equivalence classes.
- **Emmy Noether (structure and conditional symmetry):** specify what maps preserve, in which category and under which assumptions. G0-L covariance checks and transport/contrast diagnostics are not applications of Noether's first theorem: no variational action or conserved charge is defined here. Noetherian finite generation is a separate memory hypothesis, not an automatic stabilization theorem for changing memory.
- **Ramon Llull (generative interrogation):** difference, concordance, contrariety and mediating questions generate questions, not certify answers. The multi-register letter combinations in Pedro de Guevara's 1586 rendering of the Ars motivate explicit syntactic registers, not theological authority. That book is a later compilation of Llull, not a thirteenth-century autograph.
- **IGBundle/Trasgo/§-LANG:** carry selected invariants across contexts, declaring transformed, lost and unresolved parts. The grammar is L_syn/L_sem + S; G0-AD's executable bounded classifier and audits are L_op, but no externally validated M performance exists.

This proposal does not claim those historically distinct programmes were secretly the same theory.

## The object implemented

For every context c and declared transformation g, store a separate memory cell:

    M(c,g) = (p_eff, d_eff, n_preserved, n_changed, provenance).

A feedback encounter supplies actual observed behavioural responses before and after g, not hidden causal labels. Exact response equality is only a *task surrogate* for relation preservation. It does not assert sameness of persons, semantics, neural states or topology.

Fast observations increment effective and raw evidence. Slow dissipation with 0 < rho <= 1 updates effective evidence by rho^k, but never deletes raw counterexample witness counts. The bounded provenance ledger either accepts an encounter or refuses it before silent eviction. The context-transformation table cannot exceed configured limits. An explicit context fork records lineage, does not copy the parent's unverified symmetry, and requires later experiences to populate the child. This is computational contextual plasticity, not physical neurite growth.

A decision is provisional:
- preserve if sufficient current positive observations and no counterexample for that context and transformation;
- distinguish if sufficient current negative observations and no positive witness;
- contested whenever both were observed; never average a contradiction into a confident statement;
- abstain if evidence is insufficient, decayed or missing.

Any contested state may *propose* a hypothetical Worldlet split. It cannot self-authorize a split, rewrite its trace or promote its evidence status. This conservative rule may over-abstain under noisy feedback: measurable limitation, not an asymptotic learning guarantee.

Llull's generator emits four typed questions about concordance, difference, contrariety and mediation, all marked S. Contrariety asks what would refute a proposed invariance; an ordinary changed response is never automatically a logical contradiction.

## Noether-aware geometry, with correct scope

The bridge uses G0-L Correspondence (source/target Worldlets, rectangular forward T, declared return R). For states x,y it reports:

    [R]_transport,target = y - T x
    [R]_transport,roundtrip = x - R T x
    q_target = (y-Tx)^T H_target (y-Tx).

Under frame changes B_s, B_t and metric co-transformation, the target defect changes by B_t while q_target stays invariant. This is a coordinate-covariance identity, not a Noether conservation law. With indefinite H_target the quadratic may be negative or vanish for nonzero defects; it is never called positive loss or physical resource.

The contrast auditor separately accepts an *external* witness that two source situations must remain distinguishable. It detects whether T identifies them. This oracle is excluded from the learner's inputs. Rank-nullity explains why a rectangular map from higher to lower dimension can erase a contrast; that need not be a failure when the task never needs that direction.

The behavioural [R]_sem remains distinct from [R]_transport and all other named remainders in the repository.

## A small impossibility result — formalization target

Let f(c,g) be a deterministic classifier whose only input is a declared context c and transformation g, returning whether the relevant response is preserved. If a dataset contains two encounters with the same (c,g), one requiring preservation and one requiring distinction, f cannot classify both correctly. On this equally weighted pair, its accuracy is at most 1/2.

Proof: f(c,g) takes one value, whereas the two required outputs differ. Exactly one prediction must be mistaken. No neural, geometric or Noetherian assumption is needed.

**P:** this elementary conditional statement. **H:** context refinement, informative sensing and provenance can resolve some conflicts in the complete ESMF ecology without false merging. **Not established:** that conflicts reflect context rather than noise or nonstationarity; that splits help; or that the proposed system outperforms a well-informed conventional model.

## Concrete tests in this branch

The deterministic demonstration uses the same half-turn transformation for two contexts. Under numeral reading, the observed response six → nine changes. For tracking an identified physical object, the response to the rotated object is preserved. This is a test fixture; it is not a general organism trained without labels.

Tests cover: the finite typed interrogative grammar; distinct decisions in two contexts for one transformation; contested evidence surviving dissipation; explicitly unapproved split proposal; independent child context; bounded provenance and pair capacity; G0-L rectangular transport under independent nonorthogonal frames; an invisible-kernel contrast; and a genuinely negative quadratic on an indefinite metric.

Run:

    python -m esmf.relational_genesis_ad
    python -m unittest discover -s tests -p 'test_relational_genesis_ad.py' -v

These are executable-definition checks and finite fixtures, NOT empirical generalization. CI separately tests Python 3.11/3.12/3.13 and ruff after a PR is opened.

## Next experimental bridge: G0-AD1, not included here

Pre-register held-out G0-AD1 *before running it* (see G0_AD_PREREGISTRATION.json). Couple G0-AB/AC sensing and positive resource governors to the G0-AD question/witness layer through explicit causal observation and action adapters. Never give the learner evaluator-only contrast labels, hidden symmetry truth, source continuity or external G0-L diagnostics.

The test should learn the same surface transformation under different contextual meanings, evaluate new examples/new contexts, and include noisy/inconsistent feedback, drift, context splits and finite operation/energy budgets. Compare context-aware grammar with conventional context-aware inference, same-capacity context-shuffled conditions, frozen grammar and equivalent-real implementations. Report false invariant merges, distinctions preserved, abstention/coverage, drift recovery, split precision, real CPU and simulated resource expenditure. Zero false merges from abstaining on everything is not success.

Freeze seeds, scoring and capacity before looking at held-out outcomes. If improvement vanishes against an equivalently informed real implementation, the geometric lift remains an explanatory diagnostic and not a demonstrated computational advantage.

## Architectural boundary

This is **partial integration**, not perfection or completion of ESMF-0. There is no new physical growth, integrated G0-Q/AC0+Worldlet learner, online sheaf descent, general group discovery, theorem about neural memory, or claim of sentience. Frozen held-out measurements and the reproduction guard are untouched. The next claim-promotion step is independent held-out evaluation, including negative results when observed.
