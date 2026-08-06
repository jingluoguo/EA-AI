# Symbolic Baseline Capability Catalog

This file is the source of truth for the shipped symbolic baseline capability set.
It groups supported types by task family instead of keeping the catalog inside the README.

## 1. State and fact tracking

- Current state relations: `in`, `at`, `owner`, `color`, `access`, `exists`, `name`, `likes`, `dislikes`
- State projectors and overwrites: `put_in`, `take_out`, `move`, `give`, `paint`, `open`, `close`, `create`, `destroy`, `be_in`, `not_in`, `profile_name`, `profile_like`, `profile_dislike`
- Normal state behavior: current-state overwrite, negation removal, correction overwrite, destruction cleanup

## 2. Event and role queries

- Compound query composition: `compound`
- Event actor queries: `actor_for_event`, `latest_actor_for_event`, `earliest_actor_for_event`
- Item handler queries: `actor_for_item`, `latest_actor_for_item`
- History by actor: `actions_by_actors`, `inventories`
- Role aliases used by event matching: `item`, `holder`, `source`, `actor`, `theme`, `goal`, `recipient`
- Item usage history frame: `handle`

## 3. Location, containment, and quantity

- Location queries: `location`, `initial_location`, `location_before_actor_action`, `location_before_event`, `location_after_event`
- Containment queries: `contents`, `contents_except`, `contents_before_event`, `contents_after_event`, `places_visited`
- Comparison queries: `count`, `compare_count`, `same_location`
- Polar variants: `polar_location`, `polar_contents`, `polar_existence`
- Nested closure: multi-level container and place propagation

## 4. Existence and attribute tracking

- Existence queries: `existence`
- Attribute queries: `owner`, `color`, `object_state`
- Polar existence and existence cleanup after destruction

## 5. Causal, belief, and contradiction reasoning

- Causal queries: `why`, `counterfactual_location`
- Causal frames: `if_then`, `because`
- Reported speech and belief frames: `say`, `believe`
- Source queries: `claim_source`, `belief_source`
- Personal-view query: `belief_location`
- Consistency checks: `contradictions`

## 6. Dialog and profile

- Dialog intent: `dialog_act`
- Default dialog targets: `greeting`, `thanks`, `farewell`, `identity`, `capabilities`, `summary`
- Profile lookup intent: `profile`
- Reference resolution for follow-up fragments: `this chip`, `here`, and similar context-based pointers

## 7. Learning inputs

- Statement training set: `data/statement_examples.jsonl`
- Query training set: `data/query_examples.jsonl`
- Intent training / feedback set: `data/intent_examples.jsonl`
- Runtime artifacts: `data/statement_model.json`, `data/query_model.json`, `data/dialog_answer_model.json`
