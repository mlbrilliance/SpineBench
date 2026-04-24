# Prior-art digest (read before Week 7)

These six papers define the sycophancy-eval landscape. Your Related Work section has to position SpineBench against each one. 30-60 min per paper; pull the full PDFs for the ones marked ⭐ as methodological neighbors — those are the ones reviewers will cite against us.

---

## 1. ⭐ ELEPHANT — Cheng, Yu, Lee, Khadpe, Ibrahim, Jurafsky (ICLR 2026)

**One-liner:** "Social sycophancy" = excessive preservation of the user's face (desired self-image). Measured across 11 models; finds LLMs preserve user face **45 percentage points more than humans**.

**Method:** Preference/affirmation framing. Not factual ground truth — whether the model validates the user's self-image.

**How SpineBench differs:**
- ELEPHANT uses social/preference axes. SpineBench uses factual ground truth with known-correct answers — auditable without preference labelling.
- ELEPHANT scores one composite "face preservation" number. SpineBench has 10 failure modes decomposed.
- ELEPHANT evaluated 11 models (mostly frontier). SpineBench is 50+ open-only.

**Must cite as:** the closest framing-level neighbor; acknowledge we inherit the "models validate users" framing but operationalize via factual correctness.

URL: https://openreview.net/forum?id=igbRHKEiAs

---

## 2. ⭐ Kim & Khashabi (2024) — "When Models Fold Under Follow-up Pushback"

**One-liner:** Models correctly judge both sides of an argument *in parallel*, but flip their evaluation under follow-up pushback. The flip rate is the sycophancy signal.

**Method:** Judge-response-pushback-judge-again protocol. Measures delta.

**How SpineBench differs:**
- Kim & Khashabi focus on pairwise evaluation flipping. SpineBench generalizes to 10 pressure types beyond pushback.
- They use preference judgments. We use factual correctness.
- Their testbed is arguments/opinions. Ours is facts.

**Must cite as:** the closest methodological neighbor. The pushback-flip design is essentially the `direct_pushback` and `escalating_insistence` failure modes in SpineBench. Frame SpineBench as a generalization across 10 pressure axes.

---

## 3. SycEval — Fanous, Goldberg, et al. (AAAI/ACM AIES 2025)

**One-liner:** Sycophancy eval focused on factual QA under pressure. Covered closed models; measured flip rates under user pushback.

**Method:** Curated factual QA + user pushback; measures rate of sycophantic flips.

**How SpineBench differs:**
- SycEval was primarily closed-model. SpineBench is open-models-only — makes HF Pro load-bearing and ensures reproducibility.
- SycEval has 2-3 pressure modes. SpineBench has 10.
- SycEval doesn't publish a leaderboard Space or held-out set with rotation. We do.

**Must cite as:** direct methodological sibling for the factual-pressure axis. Our contribution is decomposition + open-only + rotation.

---

## 4. SYCON-Bench (May 2025)

**One-liner:** Multi-turn sycophancy under extended conversational pressure.

**Method:** Longer dialogues, measures drift over multiple turns.

**How SpineBench differs:**
- SYCON-Bench's pressure is free-form conversational. SpineBench's pressure is structured via 10 named templates — enables ablation ("which mode is each model vulnerable to").
- SYCON-Bench doesn't fully decompose failure modes. SpineBench does.

**Must cite as:** concurrent work on multi-turn pressure; we go narrower-but-deeper on template structure.

---

## 5. Sharma et al. 2023 (Anthropic) — "Towards understanding sycophancy in language models"

**One-liner:** The foundational sycophancy paper. Identifies that LLMs across providers systematically match user beliefs over truthful responses. Introduces synthetic-data mitigation.

**Method:** Crafted prompt templates where correct answer conflicts with what user states; measures model compliance with user.

**How SpineBench differs:**
- Sharma et al. establish *that* sycophancy exists; SpineBench quantifies it across 50 open models with statistical rigor.
- They have ~5 tasks; we have 10 pressure modes with hundreds of scenarios each.

**Must cite as:** foundational context, not competitor. This is "the paper that named the problem."

URL: https://arxiv.org/abs/2310.13548

---

## 6. Perez et al. 2022 — "Discovering Language Model Behaviors with Model-Written Evaluations"

**One-liner:** Introduced the methodology of using LMs to generate evaluation items at scale. Found sycophancy among other emergent behaviors.

**Method:** Model-written evaluation scenarios.

**How SpineBench differs:**
- Perez et al. rely on model-generated scenarios, which invites "generator model bias." SpineBench uses hand-crafted pressure templates + external factual ground truth.
- SpineBench validates with human audit + adversarial probe; Perez et al. rely on LM-generated labels.

**Must cite as:** methodological precedent for scalable eval construction; we intentionally avoid the model-written-eval pitfall.

---

## Honorable mentions (cite but don't dwell)

- **DarkBench** (Mar 2025) — deceptive/dark-pattern behaviors; narrower than sycophancy.
- **Syco-Bench** (May 2025) — single-turn sycophancy; narrower than our multi-turn + multi-mode setup.
- **Bullshit Eval** (Jul 2025) — bullshit-detection axis; adjacent but different phenomenon.
- **Denison et al. (2024)** — "reward hacking as a spectrum from flattery to subterfuge"; useful for the discussion/future-work section.
- **Hong et al.** — multi-turn sycophancy + finding that alignment tuning amplifies and scaling reduces.
- **LiveBench** (ICLR 2025) — contamination-limited benchmark; our quarterly rotation is explicitly modeled after theirs.

---

## Writing the Related Work section (Week 7 recipe)

Structure as four sub-sections, ~3/4 page each:

1. **Foundational work on LLM sycophancy** — Sharma et al., Perez et al. (context, not competitors).
2. **Factual sycophancy under pressure** — SycEval, Kim & Khashabi (our closest methodological neighbors; explicit three-way comparison table).
3. **Social/preference sycophancy** — ELEPHANT, SYCON-Bench (adjacent framings we inherit from).
4. **Benchmark construction methodology** — LiveBench (rotation), MMLU-Pro / TruthfulQA / SimpleQA (our GT sources), JudgeBench / CalibraEval (our judge-bias-aware methodology).

End with a tight **"What's new in SpineBench"** paragraph hitting the six defensible claims from the plan: factual GT, cross-product, 10-mode decomposition, open-only, full statistical rigor, quarterly rotated held-out.

---

## Things to have printed and at hand during the write-up

- The six ELEPHANT / SycEval / SYCON-Bench / Kim & Khashabi / Sharma / Perez abstracts (first 500 words each).
- The LiveBench methodology section (contamination-limited design).
- A three-row comparison table: (This paper | ELEPHANT | SycEval) across columns [axis, ground truth, pressure modes, model count, open/closed, rotation, statistical rigor].
