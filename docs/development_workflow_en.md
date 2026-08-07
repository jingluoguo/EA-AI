# Training-First Development Workflow

This document holds the details moved out of the README. [AGENTS.md](../AGENTS.md) remains the source of truth: new capabilities must be centered on training data, intermediate structures, and replaceable learned capabilities, without bypassing the structural learning path.

## Locate The Failing Layer

For each failing example, record:

- The raw input and context.
- The current `Structure.linearize()` output.
- The expected `INTENT/QUERY/STATE` or answer.
- The failing layer: lexing, normalization, statement parsing, Query parsing, state update, rule inference, or answer generation.

Recommended landing points:

- Punctuation, tails, or conversational question fragments are not preserved: edit `perception/lexer.py`.
- Particles, synonym actions, or slot boundaries are polluted: edit `perception/normalizer.py`.
- A statement does not map to an existing `FRAME/ROLE`: add `data/statement_examples.jsonl` samples, then run `make train`.
- A question uses different wording with the same meaning: add `data/query_examples.jsonl` samples, then run `make train`.
- A new event changes the current world: edit `world/state.py` or `world/event_schema.py`.
- `QUERY` and `FRAME/STATE` exist but no rule is inferred: edit `reasoning/rules/`.
- A rule is inferred but the answer wording is wrong: edit `reasoning/answers/`.

## End-To-End Flow

| Stage | Work | Main Code / Artifacts |
| --- | --- | --- |
| Collect samples | Save failures, feedback, verified memory | `data/*.jsonl` |
| Validate samples | Check fields, structure, slots | `comprehension/query.py`, `comprehension/statement.py`, `comprehension/intent_dataset.py` |
| Train neural capability | Train Query/Statement input models | `data/query_neural_model.*`, `data/statement_neural_model.*` |
| Runtime loading | Load weights, trusted answers, long-term memory | `my_neural.py`, `kernel.py`, `capabilities.py` |
| State projection | Project `FRAME` into current `STATE` | `world/state.py`, `world/event_schema.py` |
| Structural reasoning | Infer rules and answers from structures | `reasoning/selection/`, `reasoning/rules/`, `reasoning/answers/` |
| Uncertainty decision | Answer, ask for confirmation, or queue | `metacognition/confidence.py`, `motor/feedback.py` |
| Regression | Test schema, loaders, evaluation, and end-to-end answers | `make check`, `tests/test_*.py` |

## Query Neural Training

Query samples live in `data/query_examples.jsonl`:

```json
{"question":"芯片在哪里","entities":[{"role":"item","name":"芯片"}],"query":{"intent":"location","target":"$item#1","qualifiers":[]},"source":"training","split":"train"}
```

`$item#1`, `$container#1`, and `$place#1` are structural slots, not regexes. Training maps inputs to structural labels, then runtime materializes those labels into `QUERY`.

```bash
make train
uv run struct eval-query --query-data data/query_examples.jsonl
```

## Statement Neural Training

Statement samples live in `data/statement_examples.jsonl`:

```json
{"sentence":"小张认为芯片在托盘里","sentence_template":"$person#1认为芯片在托盘里","entities":[{"role":"person","name":"$person#1"}],"frames":[{"frame_type":"believe","roles":{"person":"$person#1","proposition":"芯片在托盘里"}}],"source":"human_feedback","split":"train"}
```

Active voice, passive voice, reordered forms, and surface variants such as container suffixes should normalize into the same `FRAME/ROLE` model.

```bash
make train
uv run struct eval-statement --statement-data data/statement_examples.jsonl
```

## Interactive Self-Learning

`struct ask --learn-on-fail` uses one uncertainty policy:

| Confidence | Behavior |
| --- | --- |
| `>= 0.90` | The structure is confident enough; answer directly. |
| `0.50 - 0.90` | Ask whether a similar structure is intended; if confirmed, write JSONL and retrain. |
| `< 0.50` | Write to `data/unrecognized_examples.jsonl` for offline labeling. |

Answers are not created directly from runtime interaction. Only trusted answer samples enter `data/dialog_answer_model.json`.

## Memory And Knowledge

Long-term memory has two sources:

- `data/memory_direct_examples.jsonl`: explicit state entries.
- `data/memory_chat_examples.jsonl`: manually confirmed chat sediment.

```bash
uv run struct compile-memory
uv run struct add-memory --state name 我 小王
uv run struct add-memory "我叫小王"
```

Long-term knowledge is separate:

- `data/memory_knowledge_examples.jsonl`
- `data/memory_knowledge_model.json`

```bash
make knowledge QUESTION="为什么天是蓝的？" ANSWER="因为短波长蓝光更容易被大气散射。"
make knowledge FILE=path/to/qa.jsonl
```

## Intent Data

Intent analysis should learn "observed behavior -> mental-state hypothesis":

```json
{"observation":"孩子伸手去拿杯子","context":["杯子在桌上"],"world_state":["at(杯子,桌上)"],"belief_state":["believes(孩子,visible(杯子))"],"answer":"孩子想拿到杯子。","source":"human_feedback","split":"train","intention":{"subject":"孩子","goal":"拿到杯子","belief":"孩子认为杯子在眼前","strategy":"伸手抓取杯子","evidence":"孩子伸手去拿杯子","confidence":0.85,"source":"human_feedback"}}
```

```bash
uv run struct add-intent "孩子伸手去拿杯子" \
  --subject "孩子" \
  --goal "拿到杯子" \
  --belief "孩子认为杯子在眼前" \
  --strategy "伸手抓取杯子"

uv run struct eval-intent --train-data data/intent_examples.jsonl
```

## Testing Requirements

Every new capability needs tests proving that active/passive, reordered, or synonymous expressions map to the same intermediate structure. Training-related changes must also cover data loading, feedback writes, schema validation, or evaluation paths.
