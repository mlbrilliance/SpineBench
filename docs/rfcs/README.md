# Architecture RFCs

Design proposals for SpineBench module refactors, following the deep-module philosophy
(small interface, large implementation).

## Active RFCs

| # | Title | Status | Priority |
|---|---|---|---|
| [0001](0001-evaluator.md) | Deepen the evaluation pipeline into a single `Evaluator` | Accepted | high (Week 3 blocker) |
| [0002](0002-corpus-builder.md) | Deepen the data pipeline into a `CorpusBuilder` | Accepted | medium |
| [0003](0003-model-runtime.md) | Introduce a `ModelRuntime` layer for provider orchestration | Accepted | medium (Week 4 prereq) |

## Implementation order

1. **0001 first** — Week-3 features (CoT judge prompts, leave-one-judge-out variance, adversarial probe set) will land on top of the deepened Evaluator, so it must ship before the pilot run.
2. **0002 second** — Before the quarterly held-out rotation; can slot into any gap in Week 3 or end of Week 3.
3. **0003 third** — Before Week 4's full 50-model run. SHA pinning becomes mandatory at that point.

All three refactors are structural (no behavior changes) and are tested via deleted-then-replaced boundary tests. TDD discipline: red → green → refactor on every step. Implementation work routes through multi-model-router per the standing rule in ~/.claude/CLAUDE.md.
