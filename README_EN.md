# Structural LLM Minimal Lab

[中文](README.md) | [English](README_EN.md)

This is a minimal lab for a structural-intelligence LLM. Instead of starting with a larger model, it asks a smaller question first:

> Can a model extract objects, roles, events, state, time, conditions, causality, beliefs, sources, intent, and questions from text, then reason over those structures to answer?

The current path is neural input understanding plus explicit structural reasoning: Query and Statement parsing use PyTorch character-level bidirectional GRU models, while world state, rules, and answers remain inspectable structural logic.

## Design Principles

The runtime is centered on training data, intermediate structures, and replaceable learned capabilities. New capabilities follow:

```text
Observe phenomenon -> Remove secondary factors -> Build ideal model -> Mathematical expression -> Experimental verification
```

In practice: record failing examples as JSONL training or feedback data first, then evaluate whether the missing piece is a structural label, slot, state projection, rule, or answer. Only change parser/state/query/inference code when the data proves the structure is insufficient.

Complex scenarios should be mapped into these target semantic layers:

- `ENTITY`: objects and concepts such as people, items, containers, places, organizations, times, and abstract topics.
- `TYPE/ATTRIBUTE`: categories and attributes such as color, access state, existence, preference, identity, quantity, and units.
- `FRAME/ROLE`: events, actions, relations, narrative frames, and roles such as `actor/theme/goal/source/recipient/result`.
- `STATE`: current world state, profile state, long-term memory state, and overwriteable state.
- `TIME/ORDER`: event order, before/after relations, historical slices, initial/latest/current views.
- `MODALITY/POLARITY`: negation, correction, possibility, counterfactuals, actual and hypothetical views.
- `SOURCE/BELIEF`: who said something, who believes it, evidence source, and isolation between factual and personal belief worlds.
- `CONDITION/CAUSE`: if/then, because/therefore, conditional triggers, causal explanation, and rule expansion.
- `QUANTITY/COMPARISON`: counts, sets, containment closure, exclusions, and comparisons.
- `REFERENCE/FOCUS`: pronouns, demonstratives, former/latter references, dialogue focus, and context carryover.
- `QUERY`: location, contents, count, historical event, source, belief, contradiction, causal, counterfactual, and compound questions.
- `INTENT/DIALOG_ACT`: user intent, greetings, summaries, capability questions, preferences, and learning feedback.
- `CONFIDENCE/FEEDBACK`: confidence, confirmation candidates, low-confidence queues, and trusted answer sources.
- `RULE/ANSWER`: computable rule hits, answer candidates, answer ordering, and final natural-language output.

`REL` and `EVENT` may still appear in CLI output, but they are compatibility/readability views. New reasoning should land in the structural layers above rather than adding a one-off answer path.

## Data Flow

```text
raw text
  -> perception/lexer.py keeps statement and query candidates
  -> perception/normalizer.py removes surface variation and cleans slots
  -> perception/reference.py resolves references, focus, and context carryover
  -> my_neural.py loads the Statement neural model and parses Entity + FRAME/ROLE
  -> world/event_schema.py / world/state.py projects FRAME into current STATE
  -> world/causal.py expands conditions, causality, and hypothetical views
  -> my_neural.py loads the Query neural model and parses QUERY
  -> memory/long_term.py injects verified long-term STATE
  -> belief/source selectors isolate facts, claims, and personal belief worlds
  -> intent_analyzers infer INTENT
  -> reasoning/rules/* infer RULE from QUERY + FRAME/STATE
  -> reasoning/answers/* generate an answer
  -> kernel.py returns Prediction
```

## Quick Start

```bash
uv sync
make train
make check
make ask TEXT="研究员把芯片放进托盘。托盘被带到实验室。芯片在哪里？"
```

Common commands:

```bash
make ask TEXT="芯片在哪里？"
make remember TEXT="我叫小王"
make knowledge QUESTION="为什么天是蓝的？" ANSWER="因为短波长蓝光更容易被大气散射。"
make train
make test
make check
```

The unified CLI entrypoint is:

```bash
uv run struct <command> [args...]
```

Current subcommands include `ask`, `train`, `eval-query`, `eval-statement`, `eval-intent`, `add-memory`, `add-knowledge`, and `compile-*`.

## Project Structure

```text
data/                         # JSONL training/feedback data and model artifacts
docs/
  development_workflow_en.md   # Training-first workflow and sample formats
  symbolic_baseline_capabilities_en.md
src/struct_llm/
  capabilities.py              # Pluggable cognitive capability interfaces
  kernel.py                    # Public prediction entrypoint and default assembly
  kernel_flow.py               # Perception -> comprehension -> state -> reasoning -> output
  dataset_io.py                # Shared JSONL line I/O and file fingerprints
  structure.py                 # Entity/Frame/Role/State/Query/Intention
  cli.py                       # struct command dispatcher
  cli_commands/                # ask, learning, memory command implementations
  perception/                  # lexing, normalization, reference resolution
  comprehension/               # Query/Statement/Intent schemas and evaluation
  neural/                      # Query/Statement neural models and shared training helpers
  world/                       # event schema, causal expansion, state projection
  reasoning/
    selection/                 # shared structure selectors
    rules/                     # domain-specific rule inference
    answers/                   # domain-specific answer generation
  memory/                      # long-term memory and knowledge models
  motor/                       # feedback, self-learning, dialog answers
tests/
  test_learning.py
  test_state.py
  test_queries.py
  test_dialogue.py
  support.py
```

## Development Entry Points

When adding a capability, read [AGENTS.md](AGENTS.md) and [docs/development_workflow_en.md](docs/development_workflow_en.md) first. The capability catalog lives in [docs/symbolic_baseline_capabilities_en.md](docs/symbolic_baseline_capabilities_en.md).

Test layout:

- `tests/test_learning.py`: loaders, feedback writes, neural training, runtime loading.
- `tests/test_state.py`: event projection, state overwrite, negation, and correction.
- `tests/test_queries.py`: location, contents, counts, historical events, and reference queries.
- `tests/test_dialogue.py`: dialog, memory, causality, belief, contradictions, and counterfactuals.

Every change should add the relevant end-to-end tests. For surface variation, tests should prove that multiple phrasings map to the same `FRAME/ROLE/STATE/QUERY`, not merely that one sentence can be answered.
