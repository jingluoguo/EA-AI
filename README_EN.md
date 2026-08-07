# Structural LLM Minimal Lab

[中文](README.md) | [English](README_EN.md)

This repository is a minimal experimental skeleton for a "structural intelligence" LLM. It does not start by scaling up a large neural model. Instead, it first asks a smaller question:

> Can a model extract entities, relations, events, and rules from text, then reason over those structures to produce an answer?

The current day-to-day path combines explicit structural reasoning with a PyTorch neural Query boundary model.

## Design Philosophy

The goal is structural intelligence, with no wording tables, regex parsers, or string fallbacks in the runtime. Future work is centered on training data and replaceable learned capabilities. Every new capability should follow this chain:

```text
Observe phenomenon -> Remove secondary factors -> Build ideal model -> Mathematical expression -> Experimental verification
```

In code, this means:

- Observe phenomenon: inspect the failed input, current structure output, and expected answer. Decide whether the failure is in sentence splitting, normalization, statement parsing, Query parsing, state updates, rule inference, or answer generation.
- Training first: write the failure as data first, including observation, context, world_state, belief_state, expected structures, or answer. Run evaluation before changing code; change parser, state, query, or inference code only when the data shows that structure labels or model slots are missing.
- Remove secondary factors: normalize surface variation such as `放进/放到/放入`, `盒子里/盒子里面`, active/passive voice, particles, and conversational filler before these details pollute entity slots.
- Build ideal model: map natural language into intermediate structures such as `ENTITY`, `FRAME/ROLE`, `STATE`, and `QUERY`.
- Mathematical expression: make reasoning depend on computable mechanisms: frame-role matching, current-state lookup, relation closure, and state overwrite.
- Experimental verification: every new capability needs tests showing that different surface forms map to the same structural model. Training-related changes also need tests for data loading, feedback writing, and schema validation.

### Core Structures

`FRAME/ROLE/STATE/QUERY/INTENT` is the core semantic layer of the symbolic baseline:

- `ENTITY`: recognized objects, such as people, items, containers, and places.
- `FRAME`: historical events preserved in source order, such as `put_in`, `move`, `give`, and `paint`.
- `ROLE`: event roles, such as `actor`, `theme`, `goal`, and `recipient`.
- `STATE`: the current world state projected from events, such as `in(芯片,盒子)` or `at(托盘,实验室)`.
- `QUERY`: a computable abstraction of the user question, such as `actor_for_event(put_in,item=芯片,holder=盒子)`.
- `INTENT`: a mental-state hypothesis inferred from observed behavior, with fields such as `subject`, `goal`, `belief`, `strategy`, `evidence`, and `confidence`.
- `RULE`: an inferred reasoning rule, such as `event_actor_matches` or `container_moves_contents`.

`REL` and `EVENT` are still displayed by the CLI, but they are compatibility and readability views. `REL` is materialized from current `STATE`; `EVENT` is materialized from historical `FRAME`. New reasoning capabilities should be built on `FRAME/ROLE/STATE/QUERY`, not on `REL/EVENT` as the primary model.

### Data Flow

```text
raw text
  -> perception/lexer.py: sentence splitting and query-candidate retention
  -> perception/normalizer.py: surface cleanup and slot normalization
  -> my_neural.py: load statement_neural_model.pt/json -> neural Entity + FRAME/ROLE
  -> world/state.py: FRAME -> current STATE
  -> my_neural.py: load query_neural_model.pt/json -> neural QUERY
  -> memory/long_term.py: load memory_model.json -> confirmed long-term STATE
  -> intent_analyzers: complete Structure + learned examples -> INTENT
  -> reasoning/pipeline.py facade -> reasoning/core.py: QUERY + FRAME/STATE -> RULE
  -> motor/dialogue.py: supplies learned replies
  -> kernel.py: orchestration and Prediction
```

Order matters. Statements are processed in source order. Later `put_in`, `move`, `give`, and `paint` events overwrite the old current state for the same object. Historical events remain available as `FRAME`, so "where is it now?" and "who previously put X into Y?" can both be answered.

### End-To-End Implementation Flow

The project works as "training data distills capability; runtime loads neural capability artifacts":

| Stage | Purpose | Method / Technology | Main Code / Artifact |
| --- | --- | --- | --- |
| 1. Collect examples | Save user inputs, failures, human feedback, and confirmed memory | Append-only JSONL datasets; training schemas; human-confirmed feedback | `data/query_examples.jsonl`, `data/statement_examples.jsonl`, `data/intent_examples.jsonl`, `data/dialog_answer_examples.jsonl`, `data/memory_direct_examples.jsonl`, `data/memory_chat_examples.jsonl` |
| 2. Validate examples | Ensure records are complete and structurally valid | Structured schema validation; `dataclass` sample objects; slot-field checks | `comprehension/query.py`, `comprehension/statement.py`, `comprehension/intent_dataset.py` |
| 3. Train neural capability artifacts | Train neural input models from examples while materializing verified reply and memory artifacts | Character-level bidirectional GRU; entity-role sequence tagging; Query/Statement structural labels; learned reply aggregation; memory-state merge; source-data `sha256`; atomic JSON/weight writes | `data/query_neural_model.pt`, `data/query_neural_model.json`, `data/statement_neural_model.pt`, `data/statement_neural_model.json`, `data/dialog_answer_model.json`, `data/memory_model.json` |
| 4. Load at runtime | Load neural input models, verified answers, and confirmed memory instead of scanning raw files | PyTorch weight loading; capability registration; replaceable learner interfaces; long-term state injection | `my_neural.make_model()`, `NeuralQueryParser`, `NeuralStatementParser`, `LearnedDialogActAnswerer`, `memory_states` |
| 5. Split text | Separate statements from query candidates | Punctuation sentence splitting; comma/semicolon candidate splitting; chat-fragment retention; tail retention | `perception/lexer.py` |
| 6. Normalize surface form | Remove wording differences that do not change meaning | Particle cleanup; question-wrapper cleanup; synonym action normalization; container-suffix normalization; `啥 -> 什么` | `perception/normalizer.py` |
| 7. Understand statements | Convert statements into entities and historical events | Sentence-template slot extraction; entity-role instantiation; `FRAME/ROLE` template instantiation | `comprehension/statement.py`, `ENTITY`, `FRAME`, `ROLE` |
| 8. Understand Query | Convert questions into computable queries | Character-level bidirectional GRU classification; role-slot instantiation; structural Query reconstruction; compound-query composition | `neural/query_classifier.py`, `comprehension/query.py`, `QUERY` |
| 9. Project state | Derive current world state from historical events | Event schemas; state projectors; state reducers; later events overwrite earlier state | `world/state.py`, `world/event_schema.py`, `STATE` |
| 10. Reason structurally | Infer rules and answers from structure | Frame-role matching; state lookup; relation closure; event constraints; counterfactual replay; answerers | `reasoning/core.py`, `reasoning/pipeline.py`, `reasoning/rules/`, `reasoning/answers/`, `RULE`, answerer |
| 11. Uncertainty decision | Choose direct answer, confirmation, or learning by confidence | Confidence bands; `>=0.90` answers directly; `0.50-0.90` asks for confirmation; `<0.50` starts guided learning | `metacognition/confidence.py`, `motor/feedback.py` |
| 12. Self-learning feedback | Confirm similar meanings or record examples for later labeling | Mid-confidence recall; low-confidence queueing; answers load from trusted sources; neural retraining; immediate retry | `motor/feedback.py`, `motor/learning_queue.py`, `motor/dialogue.py`, `struct-ask --learn-on-fail` |
| 13. Verify experimentally | Validate datasets, artifacts, and end-to-end behavior | Dataset evaluation; unittest regression; structure linearization assertions; answer assertions | `make check`, `uv run python -m unittest discover -q -b` |

The Query and Statement lanes now use neural-network weights: `data/query_neural_model.pt/json` and `data/statement_neural_model.pt/json` store PyTorch parameters, vocabulary, labels, and source-data fingerprints. The old Query/Statement compiled artifacts have been removed.

### Technologies Used

The implementation keeps structural reasoning lightweight, while Query understanding uses PyTorch:

- Python `dataclass`: defines `Entity`, `Frame`, `Role`, `State`, `Query`, `Intention`, training examples, and neural metadata structures.
- PyTorch: trains a character-level bidirectional GRU Query classifier that maps questions to structured labels.
- JSONL datasets: `data/query_examples.jsonl`, `data/statement_examples.jsonl`, `data/intent_examples.jsonl`, `data/dialog_answer_examples.jsonl`, `data/memory_direct_examples.jsonl`, and `data/memory_chat_examples.jsonl` store appendable training, feedback, and confirmed memory records.
- Neural input artifacts: `data/query_neural_model.pt/json` and `data/statement_neural_model.pt/json` store learned Query and Statement understanding; `data/dialog_answer_model.json` and `data/memory_model.json` store verified replies and long-term memory.
- Structural slots: `$item#1`, `$container#1`, `$person#1` encode entity roles and occurrence order.
- Surface normalization: `perception/normalizer.py` unifies particles, synonym actions, container suffixes, question wrappers, and surface variants such as `啥/什么`.
- Neural Query classifier: `my_neural.py` trains and loads a character-level bidirectional GRU classifier, then reconstructs a structured `QUERY`.
- Neural Query metadata: retains structural labels, representative abstract questions, and character features for feedback tooling and similar-meaning suggestions, without acting as an old compiled runtime fallback.
- Uncertainty policy: `metacognition/confidence.py` owns confidence thresholds so CLI and reasoning layers do not grow scattered decisions.
- Template instantiation: neural Statement metadata keeps structural templates for instantiating `ENTITY + FRAME/ROLE`.
- State projection: `world/state.py` turns historical `FRAME` values into current `STATE` values and handles later events overwriting earlier state.
- Structural inference: `reasoning/core.py` produces `RULE` and answers from frame-role matching, state lookup, relation closure, event constraints, and counterfactual replay; `reasoning/pipeline.py` is only the stable facade.
- Capability registration: `CognitiveCapabilities` composes statement learning, Query learning, state projection, state reduction, rule inference, and answer generation as replaceable capabilities.
- Feedback-learning service: `motor/feedback.py` and `motor/dialogue.py` encapsulate similar-meaning suggestions, new dialog capabilities, trusted answer loading, and neural retraining; the CLI only handles interaction.
- CLI and Makefile: `struct-ask`, `struct-train-neural`, `struct-eval-*`, and `struct-add-memory` expose commands; `make train-neural`, `make check`, `make ask`, and `make remember` are daily entry points.
- unittest regression: `tests/test_reasoner.py` covers loaders, feedback writes, neural training, runtime loading, structural reasoning, and end-to-end answers.

### Capability Composition

The cognitive kernel composes learned, state, reasoning, and answer capabilities through `CognitiveCapabilities`:

```python
capabilities = default_capabilities()
prediction = predict(text, capabilities)
```

There are seven pluggable capability types:

- `statement_parsers`: learn a statement mapping into `Entity + Frame`.
- `state_projectors`: project a `Frame` into `State`.
- `state_reducers`: decide how a new state updates the current world.
- `query_parsers`: learn a query mapping from a candidate into `Query`.
- `rule_inferers`: infer a rule name from a complete `Structure`.
- `answerers`: generate natural-language answers from a ruled `Structure`.
- `intent_analyzers`: learnable intent-analysis capabilities that take the raw text and complete `Structure`, then emit `Intention` hypotheses.

When adding a capability, add the corresponding JSONL training and evaluation examples first, then replace the relevant learned capability. Do not extend the runtime with wording-specific functions.

### Training-First Workflow

When a new failure appears, use this sequence:

1. Record the example: keep the raw observation, context, current structure output, and expected `INTENT/QUERY/STATE` or answer.
2. Write the dataset record: store it as JSONL training or feedback data before expanding wording-specific matching.
3. Evaluate the current learned capability to identify whether the gap is a label, a structural slot, or a model capability.
4. Train or replace the capability: prefer updating the dataset or relevant analyzer/learner/projector/inferer.
5. Verify the loop: add tests for loaders, schema validation, feedback writing, evaluation, and the end-to-end behavior.

### Neural Query Training

The Query flow is neural-first:

1. Put examples into `data/query_examples.jsonl`.
2. Run `make train-neural` or `uv run struct-train-neural` to produce `data/query_neural_model.pt` and `data/query_neural_model.json`.
3. `struct-ask --neural-provider "my_neural:make_model"` loads the neural Query model instead of scanning the raw dataset.

```json
{"question":"芯片在哪里","entities":[{"role":"item","name":"芯片"}],"query":{"intent":"location","target":"$item#1","qualifiers":[]},"source":"training","split":"train"}
```

`$item#1`, `$container#1`, and `$place#1` are structural slots, not regexes. Neural training learns the corresponding labels and reconstructs `QUERY` at runtime.

```bash
make train-neural
```

Use this to inspect the neural training result:

```bash
uv run struct-eval-query --query-data data/query_examples.jsonl
```

In short: add Query examples, run `make train-neural`, and use the neural model at runtime.

### Neural Statement Training

Statements now use the neural runtime path just like Query:

1. Put examples into `data/statement_examples.jsonl`.
2. Run `make train-neural` to train the character-level bidirectional GRU and produce `data/statement_neural_model.pt` plus `data/statement_neural_model.json`.
3. `struct-ask --neural-provider "my_neural:make_model"` loads the neural model, emits `ENTITY + FRAME/ROLE`, and then uses the existing state projection and structural reasoning.

```json
{"sentence":"小张认为芯片在托盘里","sentence_template":"$person#1认为芯片在托盘里","entities":[{"role":"person","name":"$person#1"}],"frames":[{"frame_type":"believe","roles":{"person":"$person#1","proposition":"芯片在托盘里"}}],"source":"human_feedback","split":"train"}
```

Active/passive voice, reordered phrases, and surface variants such as `里面/里边/里头` are normalized first, then collapsed into the same `FRAME/ROLE` model.

Train and inspect the neural statement model:

```bash
make train-neural
uv run struct-eval-statement --statement-data data/statement_examples.jsonl
```

Run the neural path day to day:

```bash
make ask TEXT="阿明递送芯片到库房。芯片在哪里？"
```

In short: add JSONL examples, run `make train-neural`, and let runtime use the neural Statement and Query models.

After changing training data, run:

```bash
make train-neural
make check
```

### Interactive Self-Learning

For daily testing, use:

```bash
make ask TEXT="你擅长什么"
```

Runtime behavior follows one confidence policy:

| Confidence | Behavior |
| --- | --- |
| `>= 0.90` | Treat the structure as certain enough and answer directly. |
| `0.50 - 0.90` | Do not guess the answer; ask the user to confirm the inferred meaning first. If confirmed, write the sentence back to JSONL, retrain the neural model, and retry. |
| `< 0.50` | Do not ask follow-up questions; write the input to `data/unrecognized_examples.jsonl` and tell the user it cannot be recognized yet. |

If the model cannot answer directly, it searches the neural Query metadata for a similar meaning, for example asking whether the sentence means "asking what I can do." Confirmed feedback is saved with that structure and the neural models are retrained. Low-confidence inputs, or rejected suggestions, are saved to `data/unrecognized_examples.jsonl` for later labeling.

If a queued example later becomes a new dialog capability, migrate it into `data/query_examples.jsonl` as a Query example, then run `make train-neural`. Answers are not generated from runtime interaction. Only trusted answer sources such as `training`, `teacher`, `self_model`, `knowledge`, `curated`, or `human_verified` enter `data/dialog_answer_model.json`. Without a trusted answer, the system says it understands the question but does not yet have a verified reply.

Long-term memory is separate and does not get written automatically. Use `make remember`, `make remember-state`, or an explicit `--remember-chat` flow with confirmation if you want to store something permanently.

### Feeding Intent Data

Intent analysis should not keep growing by matching more user phrasings. Feed "behavior observation -> mental-state hypothesis" data instead, so the system can learn `Goal + Belief + Strategy`:

```json
{"observation":"妈妈在找眼镜","intention":{"subject":"妈妈","goal":"找到眼镜","belief":"妈妈不知道眼镜在哪里","strategy":"在可能的位置寻找眼镜","evidence":"妈妈在找眼镜","confidence":0.75,"source":"human_feedback"}}
```

The full JSONL schema can also carry training context and evaluation targets:

```json
{"observation":"孩子伸手去拿杯子","context":["杯子在桌上"],"world_state":["at(杯子,桌上)"],"belief_state":["believes(孩子,visible(杯子))"],"answer":"孩子想拿到杯子。","source":"human_feedback","split":"train","intention":{"subject":"孩子","goal":"拿到杯子","belief":"孩子认为杯子在眼前","strategy":"伸手抓取杯子","evidence":"孩子伸手去拿杯子","confidence":0.85,"source":"human_feedback"}}
```

You can append feedback examples from the CLI:

```bash
uv run struct-add-intent-example "孩子伸手去拿杯子" \
  --subject "孩子" \
  --goal "拿到杯子" \
  --belief "孩子认为杯子在眼前" \
  --strategy "伸手抓取杯子" \
  --context "杯子在桌上" \
  --world-state "at(杯子,桌上)" \
  --answer "孩子想拿到杯子。"
```

Then run the smallest evaluation loop:

```bash
uv run struct-eval-intent --train-data data/intent_examples.jsonl
```

Save examples as JSONL, then inject them in code:

```python
from struct_llm.comprehension.intent import InMemoryIntentAnalyzer
from struct_llm.kernel import default_capabilities, predict

analyzer = InMemoryIntentAnalyzer.from_jsonl("data/intent_examples.jsonl")
capabilities = default_capabilities().with_intent_analyzers(analyzer)
prediction = predict("妈妈在找眼镜。你是谁？", capabilities)
print(prediction.structure.linearize())
```

Or feed the same file through the CLI:

```bash
uv run struct-ask --intent-data data/intent_examples.jsonl "妈妈在找眼镜。你是谁？"
```

`InMemoryIntentAnalyzer` is a cold-start prototype: without examples it does not guess intent; with examples it emits `INTENT` intermediate structures. Later this slot can be replaced by retrieval, a trained classifier/generator, online feedback persistence, or multi-agent reinforcement learning without expanding wording-specific rules.

## Project Structure And Configuration

```text
.
  AGENTS.md            # Agent constraints: structural-intelligence principles and module boundaries
  README.md            # Chinese project documentation
  README_EN.md         # English project documentation
  docs/
    symbolic_baseline_capabilities.md  # Chinese symbolic baseline capability catalog
    symbolic_baseline_capabilities_en.md  # English symbolic baseline capability catalog
  Makefile             # Common command entry points
  pyproject.toml       # Python package metadata, dependencies, and CLI entry points
  uv.toml              # uv configuration
  uv.lock              # uv lockfile
  data/
    query_examples.jsonl  # Query-parsing training examples
    statement_examples.jsonl  # Statement-parsing training examples
    query_neural_model.json  # Query neural metadata
    query_neural_model.pt    # Query neural weights
    statement_neural_model.json  # Statement neural metadata
    statement_neural_model.pt    # Statement neural weights
    intent_examples.jsonl # optional intent-analysis training/feedback examples
src/struct_llm/
  capabilities.py       # Cognitive capability registry and composition helpers
  kernel.py             # Cognitive loop orchestration: perception -> comprehension -> world -> reasoning -> output
  structure.py          # Structure types: entity, relation, event, frame, state, query, rule
  perception/
    lexer.py            # Sentence splitting and query candidate retention
    normalizer.py       # Particles, question frames, and slot-boundary normalization
    reference.py        # Reference resolution
  comprehension/
    statement.py        # statement schema, offline compilation, and evaluation
    query.py            # Query schema, offline compilation, and evaluation
    intent_dataset.py   # Intent training/feedback JSONL schema, validation, and append-only writing
    intent.py           # observation examples -> INTENT; replaceable by trained models
    structure_helpers.py # Pure structure construction and entity deduplication helpers
  world/
    event_schema.py     # Event schemas: role aliases, state effects, and event-query matching
    causal.py           # Conditional-rule expansion and causal state projection
    state.py            # FRAME -> current STATE
  reasoning/
    core.py             # QUERY + FRAME/STATE -> rules and answers
    pipeline.py         # Stable facade that forwards to core / rules / answers
    rules/              # Rule inference registration export
    answers/            # Answer generation registration export
  metacognition/
    confidence.py       # Confidence thresholds and uncertainty policy
  memory/
    working.py          # Working memory: focus entities, recent frames, current states
  motor/
    dialogue.py         # Dialog answer capability
    feedback.py         # Feedback learning service
    learning_queue.py   # Low-confidence sample queue
scripts/
  run_symbolic_demo.py  # Run the symbolic reasoning demo
tests/
  test_reasoner.py      # Standard-library tests
```

### Key File Responsibilities

`AGENTS.md` is the project constraint file. It states that new capabilities must follow "Observe phenomenon -> Remove secondary factors -> Build ideal model -> Mathematical expression -> Experimental verification", and it defines module ownership.

`pyproject.toml` defines package metadata, dependencies, and CLI commands:

```text
struct-demo = struct_llm.cli:run_symbolic_demo
struct-ask = struct_llm.cli:ask_symbolic
struct-add-intent-example = struct_llm.cli:add_intent_example
struct-eval-intent = struct_llm.cli:eval_intent_examples
struct-eval-query = struct_llm.cli:eval_query_examples
struct-eval-statement = struct_llm.cli:eval_statement_examples
struct-compile-dialog-answer = struct_llm.cli:compile_dialog_answer_model
struct-compile-memory = struct_llm.cli:compile_memory_model
struct-train-neural = my_neural:train
struct-add-memory = struct_llm.cli:add_memory_entry
```

`Makefile` is the daily command surface. `make ask` calls `uv run struct-ask --neural-provider "$(NEURAL_PROVIDER)" --learn-on-fail --memory-model data/memory_model.json "$(TEXT)"`; `make chat` opens interactive dialogue; `make remember` and `make remember-state` write long-term memory explicitly; `make test` runs standard-library unittest.

`structure.py` defines all intermediate structures. Prefer extending this structural model instead of encoding semantics in strings.

`comprehension/intent_dataset.py` owns the JSONL schema, validation, and append-only writing for intent training and feedback data.

`comprehension/intent.py` owns learnable intent analysis. It maps behavior observations into `Intention(subject, goal, belief, strategy, evidence, confidence)`. The default implementation consumes training/feedback examples and does not embed wording rules.

`world/event_schema.py` defines event role aliases and state effects. For example, `put_in` projects to `in(theme, goal)`, `move` projects to `at(theme, goal)`, `open/close` projects to `access(theme, result)`, and `create/destroy` projects to `exists(theme, result)`. Event-query matching and counterfactual event exclusion reuse the same role aliases.

`perception/normalizer.py` removes surface variation. For example, it currently normalizes `放到/放入/放进` into the same containment action and normalizes `盒子里面/盒子里` into the container slot `盒子`.

`comprehension/statement.py` owns the statement schema and sample evaluation. `neural/statement_classifier.py` trains and loads a character-level bidirectional GRU, including entity-role sequence tagging. `my_neural.py` loads `statement_neural_model.pt/json` at runtime and maps input to `ENTITY + FRAME/ROLE`.

`world/state.py` projects historical events into the current world state. If a new event changes the world, add a state projector or state reducer here.

`comprehension/query.py` owns the Query dataset and structural labels. `neural/query_classifier.py` trains, saves, and loads the character-level bidirectional GRU. `my_neural.py` loads `data/query_neural_model.pt` at runtime, so user input passes through neural Query parsing before structural reasoning.

`reasoning/core.py` owns rule inference and answer generation. `reasoning/pipeline.py` is the stable facade, while `reasoning/rules/` and `reasoning/answers/` currently provide default registration exports.

`kernel.py` only orchestrates: split text, call capabilities, assemble structures, and return answers. Business rules should not be placed here.

### Where To Add New Capabilities

When a new failing example appears, first identify the failing layer:

- Punctuation, tails, or conversational question fragments are not preserved: edit `perception/lexer.py`.
- Particles, synonym actions, or slot boundaries are polluted: edit `perception/normalizer.py`.
- A statement expression does not map to an existing `FRAME/ROLE`: add examples to `data/statement_examples.jsonl`, then run `make train-neural`; use `uv run struct-eval-statement --statement-data data/statement_examples.jsonl` to inspect the neural training result.
- A new event changes the current world, such as taking something out of a container: edit `world/state.py`.
- The user changed the wording but the meaning is the same: add examples to `data/query_examples.jsonl`, then run `make train-neural`; use `uv run struct-eval-query --query-data data/query_examples.jsonl` to inspect the neural training result.
- `QUERY` and `FRAME/STATE` already exist, but no rule is inferred: edit the rule inferers in `reasoning/core.py`.
- A rule is inferred, but the final answer wording is wrong: edit the answerers in `reasoning/core.py`.

Every change should add an end-to-end test in `tests/test_reasoner.py`. For surface variation, the test should prove that multiple phrasings map to the same `FRAME/ROLE/STATE/QUERY`, not merely that one sentence can be answered.

### Currently Supported Capabilities

The categorized capability catalog lives in [docs/symbolic_baseline_capabilities_en.md](docs/symbolic_baseline_capabilities_en.md).
In short, the symbolic baseline covers the common structural families: state, events, time, quantity, causality, belief, dialogue, and learning inputs.

## Setup

If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync the environment:

```bash
uv sync
```

## Common Commands

Most common commands:

```bash
make demo
make ask
make remember
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

`make chat` does not auto-write long-term memory. It keeps only the current dialogue context for reasoning.

User input analysis goes through the neural boundary first, then feeds the symbolic kernel.

Explicit memory writes:

```bash
make remember TEXT="I am Xiao Wang"
make remember-state NAME=name LEFT=我 RIGHT=小王
```

Long-term knowledge is separate from long-term state memory:

- `data/memory_knowledge_examples.jsonl` stores knowledge examples (QA + query structure).
- `data/memory_knowledge_model.json` stores the compiled knowledge model.

For batch import, pass a JSONL file via `--file`, or write a single entry with `QUESTION=... ANSWER=...`:

```bash
make knowledge FILE=path/to/your_qa.jsonl
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
