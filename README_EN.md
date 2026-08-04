# Structural LLM Minimal Lab

[中文](README.md) | [English](README_EN.md)

This repository is a minimal experimental skeleton for a "structural intelligence" LLM. It does not start by scaling up a large neural model. Instead, it first asks a smaller question:

> Can a model extract entities, relations, events, and rules from text, then reason over those structures to produce an answer?

The project currently has two layers:

- `symbolic`: an explicit structural reasoning baseline that runs with the Python standard library.
- `neural`: a reserved PyTorch tiny Transformer entry point for learning to generate structure tokens and answer tokens.

## Design Philosophy

The goal is structural intelligence, not enumerating every possible natural-language phrasing with end-to-end regex templates. Every new capability should follow this chain:

```text
Observe phenomenon -> Remove secondary factors -> Build ideal model -> Mathematical expression -> Experimental verification
```

In code, this means:

- Observe phenomenon: inspect the failed input, current structure output, and expected answer. Decide whether the failure is in sentence splitting, normalization, statement parsing, Query parsing, state updates, rule inference, or answer generation.
- Remove secondary factors: normalize surface variation such as `放进/放到/放入`, `盒子里/盒子里面`, active/passive voice, particles, and conversational filler before these details pollute entity slots.
- Build ideal model: map natural language into intermediate structures such as `ENTITY`, `FRAME/ROLE`, `STATE`, and `QUERY`.
- Mathematical expression: make reasoning depend on computable mechanisms: frame-role matching, current-state lookup, relation closure, and state overwrite.
- Experimental verification: every new capability needs tests showing that different surface forms map to the same structural model.

### Core Structures

`FRAME/ROLE/STATE/QUERY` is the core semantic layer of the symbolic baseline:

- `ENTITY`: recognized objects, such as people, items, containers, and places.
- `FRAME`: historical events preserved in source order, such as `put_in`, `move`, `give`, and `paint`.
- `ROLE`: event roles, such as `actor`, `theme`, `goal`, and `recipient`.
- `STATE`: the current world state projected from events, such as `in(芯片,盒子)` or `at(托盘,实验室)`.
- `QUERY`: a computable abstraction of the user question, such as `actor_for_event(put_in,item=芯片,holder=盒子)`.
- `RULE`: an inferred reasoning rule, such as `event_actor_matches` or `container_moves_contents`.

`REL` and `EVENT` are still displayed by the CLI, but they are compatibility and readability views. `REL` is materialized from current `STATE`; `EVENT` is materialized from historical `FRAME`. New reasoning capabilities should be built on `FRAME/ROLE/STATE/QUERY`, not on `REL/EVENT` as the primary model.

### Data Flow

```text
raw text
  -> text_processing.split_sentences
  -> normalization: surface cleanup and slot normalization
  -> frame_parser: statements -> Entity + FRAME/ROLE
  -> state_engine: FRAME -> current STATE
  -> query_parser: query candidates -> QUERY
  -> inference: QUERY + FRAME/STATE -> RULE + answer
  -> reasoner: orchestration and Prediction
```

Order matters. Statements are processed in source order. Later `put_in`, `move`, `give`, and `paint` events overwrite the old current state for the same object. Historical events remain available as `FRAME`, so "where is it now?" and "who previously put X into Y?" can both be answered.

### Capability Composition

To prevent `reasoner.py` from growing back into a large if/regex file, the symbolic pipeline is composed with `StructuralCapabilities`:

```python
capabilities = default_capabilities().with_query_parsers(parse_keeper_query)
prediction = predict(text, capabilities)
```

There are six pluggable capability types:

- `statement_parsers`: parse a statement into `Entity + Frame`.
- `state_projectors`: project a `Frame` into `State`.
- `state_reducers`: decide how a new state updates the current world.
- `query_parsers`: parse a normalized question candidate into `Query`.
- `rule_inferers`: infer a rule name from a complete `Structure`.
- `answerers`: generate natural-language answers from a ruled `Structure`.

When adding a capability, first decide which layer owns it. Then add a small function to the corresponding module and register it in the default capability list. Use `.with_query_parsers(...)` and similar methods only when an external caller wants to inject a temporary extension.

## Project Structure And Configuration

```text
.
  AGENTS.md            # Agent constraints: structural-intelligence principles and module boundaries
  README.md            # Chinese project documentation
  README_EN.md         # English project documentation
  Makefile             # Common command entry points
  pyproject.toml       # Python package metadata, dependencies, and CLI entry points
  uv.toml              # uv configuration
  uv.lock              # uv lockfile
  data/
    train.jsonl        # symbolic/neural training data
    test.jsonl         # test data
    tiny_model.pt      # tiny Transformer artifact, if trained
src/struct_llm/
  world.py              # Tiny world: people, items, containers, places, task templates
  structure.py          # Structure types: entity, relation, event, frame, state, query, rule
  dataset.py            # Dataset generation and compositional train/test split
  text_processing.py    # Sentence splitting and query candidate retention
  normalization.py      # Particles, question frames, and slot-boundary normalization
  capabilities.py       # Pluggable capability interfaces
  frame_parser.py       # Statement -> Entity + FRAME/ROLE
  state_engine.py       # FRAME -> current STATE
  query_parser.py       # query candidates -> QUERY
  inference.py          # QUERY + FRAME/STATE -> rules and answers
  reasoner.py           # Lightweight orchestration layer
  vocab.py              # Character-level vocabulary for the neural model
  model.py              # Optional PyTorch tiny Transformer
scripts/
  make_dataset.py       # Generate JSONL data
  run_symbolic_demo.py  # Run the symbolic reasoning demo
  train_tiny_model.py   # Train the tiny Transformer; requires torch
tests/
  test_reasoner.py      # Standard-library tests
```

### Key File Responsibilities

`AGENTS.md` is the project constraint file. It states that new capabilities must follow "Observe phenomenon -> Remove secondary factors -> Build ideal model -> Mathematical expression -> Experimental verification", and it defines module ownership.

`pyproject.toml` defines package metadata, dependencies, optional neural dependencies, and CLI commands:

```text
struct-demo = struct_llm.cli:run_symbolic_demo
struct-ask = struct_llm.cli:ask_symbolic
struct-ask-neural = struct_llm.cli:ask_neural
struct-make-dataset = struct_llm.cli:make_dataset
struct-train-tiny = struct_llm.cli:train_tiny_model
```

`Makefile` is the daily command surface. `make ask` calls `uv run struct-ask "$(TEXT)"`; `make test` runs standard-library unittest.

`structure.py` defines all intermediate structures. Prefer extending this structural model instead of encoding semantics in strings.

`normalization.py` removes surface variation. For example, it currently normalizes `放到/放入/放进` into the same containment action and normalizes `盒子里面/盒子里` into the container slot `盒子`.

`frame_parser.py` extracts `FRAME/ROLE` from statements. New event types, such as "take out", "open", or "close", should start here as statement parsers.

`state_engine.py` projects historical events into the current world state. If a new event changes the world, add a state projector or state reducer here.

`query_parser.py` abstracts user questions into `QUERY`. A new phrasing should not answer directly; it should first normalize into a structure such as `location(芯片)` or `actor_for_event(...)`.

`inference.py` owns rule inference and answer generation. New reasoning behavior usually adds both a `rule_inferer` and an `answerer`.

`reasoner.py` only orchestrates: split text, call capabilities, assemble structures, and return answers. Business rules should not be placed here.

### Where To Add New Capabilities

When a new failing example appears, first identify the failing layer:

- Punctuation, tails, or conversational question fragments are not preserved: edit `text_processing.py`.
- Particles, synonym actions, or slot boundaries are polluted: edit `normalization.py`.
- A new event is not extracted, such as "take out", "open", or "close": edit `frame_parser.py`.
- A new event changes the current world, such as taking something out of a container: edit `state_engine.py`.
- The user changed the wording but the meaning is the same: edit `query_parser.py`.
- `QUERY` and `FRAME/STATE` already exist, but no rule is inferred: edit the rule inferers in `inference.py`.
- A rule is inferred, but the final answer wording is wrong: edit the answerers in `inference.py`.

Every change should add an end-to-end test in `tests/test_reasoner.py`. For surface variation, the test should prove that multiple phrasings map to the same `FRAME/ROLE/STATE/QUERY`, not merely that one sentence can be answered.

## Setup

If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync the environment:

```bash
uv sync
```

To use the optional neural model:

```bash
uv sync --extra neural
```

## Common Commands

Most common commands:

```bash
make demo
make ask
make data
make test
```

Ask a custom question:

```bash
make ask TEXT="研究员把芯片放进托盘。托盘被带到实验室。芯片在哪里？"
```

Start interactive symbolic mode:

```bash
make chat
```

Train and run the tiny Transformer:

```bash
make train
make ask-neural TEXT="研究员把芯片放进托盘。托盘被带到实验室。芯片在哪里？"
make chat-neural
```

## Minimal Task

Example:

```text
小明把钥匙放进盒子。盒子被带到厨房。钥匙在哪里？
```

Structure:

```text
ENTITY person=小明
ENTITY item=钥匙
ENTITY container=盒子
ENTITY place=厨房
REL in(钥匙,盒子)
REL at(盒子,厨房)
EVENT put_in(小明,钥匙) WITH holder=盒子
EVENT handle(小明,钥匙)
EVENT move(盒子,厨房)
FRAME f1 type=put_in time=1
ROLE f1 actor=小明
ROLE f1 theme=钥匙
ROLE f1 goal=盒子
FRAME f2 type=handle time=2
ROLE f2 actor=小明
ROLE f2 theme=钥匙
FRAME f3 type=move time=3
ROLE f3 theme=盒子
ROLE f3 goal=厨房
RULE container_moves_contents
QUERY location(钥匙)
```

Answer:

```text
钥匙在厨房的盒子里。
```

This is the minimal shape of structural intelligence: natural language is not mapped directly to an answer. It is first mapped to an inspectable, composable, and transferable intermediate structure. The symbolic baseline extracts historical events and current state sentence by sentence, then extracts `QUERY` and infers rules. It no longer requires the full text to match one enumerated template.

It also supports more open content queries. For example, "实验室里至少有什么？" performs closure over movement and containment relations.

### Abstract Modeling Rule

Implementation follows the fixed chain:

```text
Observe phenomenon -> Remove secondary factors -> Build ideal model -> Mathematical expression -> Experimental verification
```

For example, these questions:

```text
谁把芯片放进托盘？
芯片是谁放进托盘的？
芯片被谁放进托盘的？
谁把芯片放到托盘里面的？
```

are surface variants. They should all normalize to the same event-role model:

```text
FRAME type=put_in
ROLE actor=?
ROLE theme=芯片
ROLE goal=托盘
```

Then they become a computable query:

```text
QUERY actor_for_event(put_in,item=芯片,holder=托盘)
```

And finally the answer is verified against existing events and relations:

```text
EVENT put_in(小郭,芯片) WITH holder=托盘
REL in(芯片,托盘)
```
