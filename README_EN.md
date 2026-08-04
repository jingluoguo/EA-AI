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

To prevent `reasoner.py` from growing back into a large if/regex file, the cognitive kernel is composed with `CognitiveCapabilities`. `StructuralCapabilities` remains as a compatibility alias for older callers:

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

### Modular Extension Principle

Modules should have boundaries before they have power; they should be pluggable before they become stronger.

In this project, `CognitiveCapabilities` is the inner structural-reasoning kernel. Planning, embodiment, emotion, self-modeling, and continual learning should first exist as empty but swappable slots, then grow concrete implementations without pushing that logic into `reasoner.py`.

See [docs/modular_architecture.md](docs/modular_architecture.md) for the rollout plan.

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
  docs/
    modular_architecture.md  # Pluggable module architecture and rollout plan
  data/
    train.jsonl        # symbolic/neural training data
    test.jsonl         # test data
    tiny_model.pt      # tiny Transformer artifact, if trained
src/struct_llm/
  world.py              # Tiny world: people, items, containers, places, task templates
  structure.py          # Structure types: entity, relation, event, frame, state, query, rule
  event_schema.py       # Event schemas: role aliases, state effects, and event-query matching
  dataset.py            # Dataset generation and compositional train/test split
  cognitive/
    capabilities.py     # Cognitive kernel capability registry: statements, states, queries, rules, and answers
    kernel.py           # Single cognitive-kernel pipeline that connects parsing, state, query, and inference modules
    text_processing.py  # Sentence splitting and query candidate retention
    normalization.py    # Particles, question frames, and slot-boundary normalization
    frame_parser.py     # Statement -> Entity + FRAME/ROLE
    state_engine.py     # FRAME -> current STATE
    query_parser.py     # query candidates -> QUERY
    inference.py        # QUERY + FRAME/STATE -> rules and answers
  capabilities.py       # Compatibility exports; real capability registration lives in cognitive/capabilities.py
  text_processing.py    # Compatibility exports; real implementation lives in cognitive/
  normalization.py      # Compatibility exports; real implementation lives in cognitive/
  frame_parser.py       # Compatibility exports; real implementation lives in cognitive/
  state_engine.py       # Compatibility exports; real implementation lives in cognitive/
  query_parser.py       # Compatibility exports; real implementation lives in cognitive/
  inference.py          # Compatibility exports; real implementation lives in cognitive/
  reasoner.py           # Lightweight orchestration layer
  modules/              # Outer pluggable modules; cognitive mounts the kernel and is not a second parser system
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

`event_schema.py` defines event role aliases and state effects. For example, `put_in` projects to `in(theme, goal)`, `move` projects to `at(theme, goal)`, `open/close` projects to `access(theme, result)`, and `create/destroy` projects to `exists(theme, result)`. Event-query matching and counterfactual event exclusion reuse the same role aliases.

`cognitive/normalization.py` removes surface variation. For example, it currently normalizes `放到/放入/放进` into the same containment action and normalizes `盒子里面/盒子里` into the container slot `盒子`.

`cognitive/frame_parser.py` extracts `FRAME/ROLE` from statements. New event types, such as "take out", "open", or "close", should start here as statement parsers.

`cognitive/state_engine.py` projects historical events into the current world state. If a new event changes the world, add a state projector or state reducer here.

`cognitive/query_parser.py` abstracts user questions into `QUERY`. A new phrasing should not answer directly; it should first normalize into a structure such as `location(芯片)` or `actor_for_event(...)`.

`cognitive/inference.py` owns rule inference and answer generation. New reasoning behavior usually adds both a `rule_inferer` and an `answerer`.

`reasoner.py` only orchestrates: split text, call capabilities, assemble structures, and return answers. Business rules should not be placed here.

### Where To Add New Capabilities

When a new failing example appears, first identify the failing layer:

- Punctuation, tails, or conversational question fragments are not preserved: edit `cognitive/text_processing.py`.
- Particles, synonym actions, or slot boundaries are polluted: edit `cognitive/normalization.py`.
- A new event is not extracted, such as "take out", "open", or "close": edit `cognitive/frame_parser.py`.
- A new event changes the current world, such as taking something out of a container: edit `cognitive/state_engine.py`.
- The user changed the wording but the meaning is the same: edit `cognitive/query_parser.py`.
- `QUERY` and `FRAME/STATE` already exist, but no rule is inferred: edit the rule inferers in `cognitive/inference.py`.
- A rule is inferred, but the final answer wording is wrong: edit the answerers in `cognitive/inference.py`.

Every change should add an end-to-end test in `tests/test_reasoner.py`. For surface variation, the test should prove that multiple phrasings map to the same `FRAME/ROLE/STATE/QUERY`, not merely that one sentence can be answered.

### Currently Supported Capabilities

The symbolic baseline currently supports these structural capabilities:

- Current-state tracking: putting in, moving, giving, and painting update current `STATE`.
- Attribute-state tracking: object states such as open/closed can be queried, for example "what state is the box in?"
- Existence tracking: creation and destruction events are represented as current state; destruction clears the object's current location, owner, color, and access state.
- State invalidation: taking out and negated statements delete current containment, such as "托盘里没有芯片".
- State correction: one statement can remove an old state and write a new one, such as "芯片不在托盘里而在盒子里".
- Historical event retention: after the current state changes, the system can still answer who previously put an item into a container.
- Temporal queries: "where was it at the beginning?", "where was it before someone's action?", and "what happened after an event?"
- Conditional reasoning: "if ... then ..." and "as long as ..." style rules.
- Causal explanation: "because ... so ..." and "why ..." queries.
- Nested closure: multi-level containers and movement propagation, such as "the chip is in the small box inside the big box in the lab".
- Set queries: contents of a holder, contents except X, and places visited.
- Count queries: questions such as "how many things are in this place?" return the known lower bound from the current structural closure, filtering destroyed objects.
- Count comparison: questions such as "which place has more things?" and "do these two places have the same amount?" compare the current known-content closures.
- Polar yes/no queries: questions such as "does the chip exist?", "is the chip in the tray?", and "is there a chip in the lab?" reuse the same structural state checks.
- Same-location queries: questions such as "are the chip and the bottle in the same place?" compare the current location keys of two targets.
- Role queries: latest actor for an item, actions by each actor, and current inventories by owner.
- Reference resolution: context-based follow-up phrases such as "this chip" and "here" are mapped back to known entities.
- Source tracking: questions like "who said the chip is in the tray?" are separated from factual world state.
- Belief worlds: personal-view queries such as "where does Xiao Wang think the chip is?" and "who believes the chip is in the box?", without rewriting factual world state.
- Contradiction detection: questions such as "is there any contradiction?" compare claims or beliefs against the current factual state.
- Counterfactual replay: questions such as "where would X be if someone had not done an event?" are answered by excluding the target event and replaying `FRAME -> STATE`.

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
