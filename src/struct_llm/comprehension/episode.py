from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..dataset_io import append_jsonl_object, file_sha256, load_jsonl_objects
from ..memory.long_term import memory_entities_from_states
from ..perception.normalizer import bare_topic_followup, normalize_question
from ..structure import Entity, Frame, PragmaticAct, Query, Role, State, Structure
from .query import query_from_dict, query_to_dict


EPISODE_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "episode_examples.jsonl"
EPISODE_MODEL_SCHEMA = "struct_llm.episode_model.v1"
EPISODE_RECORD_SCHEMA = "struct_llm.episode_example.v1"
RESOLVED_QUERY_SUPPRESSED_ACTS = frozenset(
    {
        "ambiguous_reference",
        "underspecified_action_request",
        "incomplete_utterance",
    }
)
RESPONSE_POLICIES = frozenset(
    {
        "answer",
        "acknowledge",
        "ask_clarification",
        "refuse",
        "remember",
        "wait_for_completion",
        "repair",
        "confirm",
    }
)


@dataclass(frozen=True)
class DialogueTurn:
    speaker: str
    text: str
    turn: int = 0


@dataclass(frozen=True)
class FeedbackDiagnosis:
    error_type: str
    user_feedback: str = ""
    previous_answer: str = ""
    original_understanding: str = ""
    correct_structure: str = ""
    generalization_target: str = ""


@dataclass(frozen=True)
class ActionResult:
    action: str
    status: str
    observation: str = ""
    user_satisfaction: str = ""


@dataclass(frozen=True)
class EpisodeTrainingExample:
    text: str
    expected_response_policy: str
    expected_pragmatic_acts: tuple[PragmaticAct, ...]
    episode_id: str = ""
    dialogue_turn: int = 0
    speaker: str = "user"
    scene: str = ""
    previous_turns: tuple[DialogueTurn, ...] = ()
    known_world_state: tuple[State, ...] = ()
    belief_state: tuple[State, ...] = ()
    relationship_state: tuple[State, ...] = ()
    focus: tuple[str, ...] = ()
    expected_entities: tuple[Entity, ...] = ()
    expected_query: Query | None = None
    expected_frames: tuple[Frame, ...] = ()
    expected_state_delta: tuple[State, ...] = ()
    expected_answer: str = ""
    action_result: ActionResult | None = None
    feedback_diagnosis: FeedbackDiagnosis | None = None
    source: str = "training"
    split: str = "train"


@dataclass(frozen=True)
class CompiledEpisodePattern:
    text: str
    response_policy: str
    pragmatic_acts: tuple[PragmaticAct, ...]
    feature_units: tuple[str, ...]
    support: int = 1


@dataclass(frozen=True)
class CompiledEpisodeModel:
    schema: str
    source_sha256: str
    example_count: int
    patterns: tuple[CompiledEpisodePattern, ...]


@dataclass(frozen=True)
class EpisodeEvaluationResult:
    total: int
    matched: int

    @property
    def accuracy(self) -> float:
        return self.matched / self.total if self.total else 0.0


@dataclass(frozen=True)
class InMemoryPragmaticAnalyzer:
    examples: tuple[EpisodeTrainingExample, ...] = ()
    min_score: float = 0.62

    @classmethod
    def from_jsonl(cls, path: str | Path, min_score: float = 0.62) -> InMemoryPragmaticAnalyzer:
        return cls(load_episode_jsonl(path), min_score=min_score)

    def __call__(self, text: str, structure) -> tuple[PragmaticAct, ...]:
        if not self.examples:
            return ()
        if structure is not None and getattr(structure, "query", None) is not None:
            query = structure.query
            if query is not None and query.intent != "dialog_act":
                examples = tuple(
                    example
                    for example in self.examples
                    if not resolved_query_suppresses_pragmatic_example(example)
                )
            else:
                examples = self.examples
        else:
            examples = self.examples
        normalized = normalize_episode_text(text)
        scored = [
            (episode_score(normalize_episode_text(example.text), normalized), example)
            for example in examples
        ]
        matches = [
            acts_with_runtime_source(example.expected_pragmatic_acts, score)
            for score, example in sorted(scored, key=lambda item: item[0], reverse=True)
            if score >= self.min_score
        ]
        acts: list[PragmaticAct] = []
        seen: set[tuple[str, str, tuple[str, ...]]] = set()
        structural_acts = (
            bare_topic_followup_act(text, structure),
            ambiguous_reference_act(structure),
        )
        structural_targets: set[tuple[str, str]] = set()
        for structural_act in structural_acts:
            if structural_act is None:
                continue
            signature = (structural_act.act, structural_act.target, structural_act.qualifiers)
            seen.add(signature)
            structural_targets.add((structural_act.act, structural_act.target))
            acts.append(structural_act)
        for candidate_acts in matches:
            for act in candidate_acts:
                signature = (act.act, act.target, act.qualifiers)
                if signature in seen or (act.act, act.target) in structural_targets:
                    continue
                seen.add(signature)
                acts.append(act)
        return tuple(acts[:3])


def resolved_query_suppresses_pragmatic_example(example: EpisodeTrainingExample) -> bool:
    return any(act.act in RESOLVED_QUERY_SUPPRESSED_ACTS for act in example.expected_pragmatic_acts)


def bare_topic_followup_act(text: str, structure) -> PragmaticAct | None:
    if structure is not None and getattr(structure, "query", None) is not None:
        return None
    topic = bare_topic_followup(text)
    if topic is None:
        return None
    return PragmaticAct(
        "underspecified_reference_query",
        topic,
        ("missing=query_intent", "response_policy=ask_clarification"),
        confidence=1.0,
        source="structural",
    )


def ambiguous_reference_act(structure) -> PragmaticAct | None:
    if structure is None or getattr(structure, "query", None) is not None:
        return None
    if getattr(structure, "frames", ()):
        return None
    reference = unresolved_reference_entity(structure)
    if reference is None:
        return None
    candidates = discourse_reference_candidates(structure)
    if len(candidates) < 2:
        return None
    return PragmaticAct(
        "ambiguous_reference",
        reference.name,
        (
            "missing=referent",
            f"candidates={'|'.join(candidates)}",
            "depends_on=focus",
            "response_policy=ask_clarification",
        ),
        confidence=1.0,
        source="structural",
    )


def unresolved_reference_entity(structure):
    return next(
        (entity for entity in getattr(structure, "entities", ()) if entity.role == "unresolved_reference"),
        None,
    )


def discourse_reference_candidates(structure) -> tuple[str, ...]:
    entities = tuple(getattr(structure, "entities", ()))
    states = tuple(getattr(structure, "states", ()))
    candidate_roles = {"profile_value", "item", "container", "thing", "person", "giver", "receiver", "place"}
    candidate_states = {"name", "likes", "dislikes", "at", "in", "owner"}
    candidates = [entity.name for entity in entities if entity.role in candidate_roles and entity.name]
    for state in states:
        if state.name not in candidate_states:
            continue
        if state.name in {"name", "likes", "dislikes", "at"} and state.right:
            candidates.append(state.right)
        elif state.name in {"in", "owner"}:
            candidates.extend(value for value in (state.left, state.right) if value)
    return tuple(dict.fromkeys(candidates))


def compile_episode_examples(
    examples: tuple[EpisodeTrainingExample, ...],
    source_sha256: str = "",
) -> CompiledEpisodeModel:
    grouped: dict[tuple[Any, ...], CompiledEpisodePattern] = {}
    for example in examples:
        normalized = normalize_episode_text(example.text)
        key = (
            normalized,
            example.expected_response_policy,
            tuple(pragmatic_act_signature(act) for act in example.expected_pragmatic_acts),
        )
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = CompiledEpisodePattern(
                text=normalized,
                response_policy=example.expected_response_policy,
                pragmatic_acts=example.expected_pragmatic_acts,
                feature_units=tuple(sorted(character_bigrams(normalized))),
                support=1,
            )
            continue
        grouped[key] = CompiledEpisodePattern(
            text=previous.text,
            response_policy=previous.response_policy,
            pragmatic_acts=previous.pragmatic_acts,
            feature_units=previous.feature_units,
            support=previous.support + 1,
        )
    return CompiledEpisodeModel(
        schema=EPISODE_MODEL_SCHEMA,
        source_sha256=source_sha256,
        example_count=len(examples),
        patterns=tuple(sorted(grouped.values(), key=lambda pattern: (-pattern.support, pattern.text))),
    )


def compile_episode_model_from_jsonl(path: str | Path = EPISODE_DATA_PATH) -> CompiledEpisodeModel:
    return compile_episode_examples(load_episode_jsonl(path), source_sha256=file_sha256(path))


def evaluate_pragmatic_analyzer(
    analyzer: InMemoryPragmaticAnalyzer,
    examples: tuple[EpisodeTrainingExample, ...],
) -> EpisodeEvaluationResult:
    matched = 0
    for example in examples:
        predictions = analyzer(example.text, episode_evaluation_structure(example))
        if all(any(pragmatic_act_matches(prediction, expected) for prediction in predictions) for expected in example.expected_pragmatic_acts):
            matched += 1
    return EpisodeEvaluationResult(total=len(examples), matched=matched)


def episode_evaluation_structure(example: EpisodeTrainingExample) -> Structure:
    states = (
        *example.known_world_state,
        *example.belief_state,
        *example.relationship_state,
    )
    return Structure(
        entities=(*memory_entities_from_states(states), *example.expected_entities),
        rules=(),
        query=example.expected_query,
        frames=example.expected_frames,
        states=states,
    )


def load_episode_jsonl(path: str | Path) -> tuple[EpisodeTrainingExample, ...]:
    return tuple(
        episode_example_from_dict(raw_record, line_number=line_number)
        for line_number, raw_record in enumerate(load_jsonl_objects(path, "episode"), start=1)
    )


def append_episode_record(path: str | Path, record: dict[str, Any]) -> EpisodeTrainingExample:
    example = episode_example_from_dict(record)
    append_jsonl_object(path, episode_example_to_record(example), sort_keys=True)
    return example


def build_episode_record(
    text: str,
    response_policy: str,
    *,
    pragmatic_acts: tuple[PragmaticAct, ...],
    episode_id: str = "",
    dialogue_turn: int = 0,
    speaker: str = "user",
    scene: str = "",
    previous_turns: tuple[DialogueTurn, ...] = (),
    known_world_state: tuple[State, ...] = (),
    belief_state: tuple[State, ...] = (),
    relationship_state: tuple[State, ...] = (),
    focus: Iterable[str] = (),
    expected_entities: tuple[Entity, ...] = (),
    expected_query: Query | None = None,
    expected_frames: tuple[Frame, ...] = (),
    expected_state_delta: tuple[State, ...] = (),
    expected_answer: str = "",
    action_result: ActionResult | None = None,
    feedback_diagnosis: FeedbackDiagnosis | None = None,
    source: str = "human_feedback",
    split: str = "train",
) -> dict[str, Any]:
    return episode_example_to_record(
        EpisodeTrainingExample(
            text=text.strip(),
            expected_response_policy=response_policy.strip(),
            expected_pragmatic_acts=pragmatic_acts,
            episode_id=episode_id.strip(),
            dialogue_turn=int(dialogue_turn),
            speaker=speaker.strip() or "user",
            scene=scene.strip(),
            previous_turns=previous_turns,
            known_world_state=known_world_state,
            belief_state=belief_state,
            relationship_state=relationship_state,
            focus=clean_string_tuple(focus),
            expected_entities=expected_entities,
            expected_query=expected_query,
            expected_frames=expected_frames,
            expected_state_delta=expected_state_delta,
            expected_answer=expected_answer.strip(),
            action_result=action_result,
            feedback_diagnosis=feedback_diagnosis,
            source=source.strip() or "human_feedback",
            split=split.strip() or "train",
        )
    )


def episode_example_to_record(example: EpisodeTrainingExample) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": EPISODE_RECORD_SCHEMA,
        "text": example.text,
        "dialogue_turn": example.dialogue_turn,
        "speaker": example.speaker,
        "expected_response_policy": example.expected_response_policy,
        "expected_pragmatic_acts": [pragmatic_act_to_dict(act) for act in example.expected_pragmatic_acts],
        "source": example.source,
        "split": example.split,
    }
    if example.episode_id:
        record["episode_id"] = example.episode_id
    if example.scene:
        record["scene"] = example.scene
    if example.previous_turns:
        record["previous_turns"] = [dialogue_turn_to_dict(turn) for turn in example.previous_turns]
    if example.known_world_state:
        record["known_world_state"] = [state_to_dict(state) for state in example.known_world_state]
    if example.belief_state:
        record["belief_state"] = [state_to_dict(state) for state in example.belief_state]
    if example.relationship_state:
        record["relationship_state"] = [state_to_dict(state) for state in example.relationship_state]
    if example.focus:
        record["focus"] = list(example.focus)
    if example.expected_entities:
        record["expected_entities"] = [entity_to_dict(entity) for entity in example.expected_entities]
    if example.expected_query is not None:
        record["expected_query"] = query_to_dict(example.expected_query)
    if example.expected_frames:
        record["expected_frames"] = [frame_to_dict(frame) for frame in example.expected_frames]
    if example.expected_state_delta:
        record["expected_state_delta"] = [state_to_dict(state) for state in example.expected_state_delta]
    if example.expected_answer:
        record["expected_answer"] = example.expected_answer
    if example.action_result is not None:
        record["action_result"] = action_result_to_dict(example.action_result)
    if example.feedback_diagnosis is not None:
        record["feedback_diagnosis"] = feedback_diagnosis_to_dict(example.feedback_diagnosis)
    return record


def episode_example_from_dict(record: dict[str, Any], *, line_number: int | None = None) -> EpisodeTrainingExample:
    prefix = f"Episode example at line {line_number}" if line_number is not None else "Episode example"
    schema = str(record.get("schema") or EPISODE_RECORD_SCHEMA).strip()
    if schema != EPISODE_RECORD_SCHEMA:
        raise ValueError(f"{prefix} has unsupported schema: {schema}")
    text = str(record.get("text") or "").strip()
    if not text:
        raise ValueError(f"{prefix} requires text.")
    response_policy = str(record.get("expected_response_policy") or "").strip()
    if response_policy not in RESPONSE_POLICIES:
        raise ValueError(f"{prefix} expected_response_policy must be one of {sorted(RESPONSE_POLICIES)}.")
    raw_acts = record.get("expected_pragmatic_acts", ())
    if not isinstance(raw_acts, list) or not raw_acts:
        raise ValueError(f"{prefix} requires non-empty expected_pragmatic_acts list.")

    raw_query = record.get("expected_query")
    raw_action = record.get("action_result")
    raw_feedback = record.get("feedback_diagnosis")

    return EpisodeTrainingExample(
        text=text,
        expected_response_policy=response_policy,
        expected_pragmatic_acts=tuple(pragmatic_act_from_dict(value, prefix) for value in raw_acts),
        episode_id=str(record.get("episode_id") or "").strip(),
        dialogue_turn=int(record.get("dialogue_turn") or 0),
        speaker=str(record.get("speaker") or "user").strip() or "user",
        scene=str(record.get("scene") or "").strip(),
        previous_turns=tuple(dialogue_turn_from_dict(value, prefix) for value in list_field(record, "previous_turns", prefix)),
        known_world_state=tuple(state_from_dict(value, prefix) for value in list_field(record, "known_world_state", prefix)),
        belief_state=tuple(state_from_dict(value, prefix) for value in list_field(record, "belief_state", prefix)),
        relationship_state=tuple(state_from_dict(value, prefix) for value in list_field(record, "relationship_state", prefix)),
        focus=field_to_tuple(record.get("focus"), "focus", prefix),
        expected_entities=tuple(entity_from_dict(value, prefix) for value in list_field(record, "expected_entities", prefix)),
        expected_query=query_from_dict(raw_query, prefix) if isinstance(raw_query, dict) else None,
        expected_frames=tuple(frame_from_dict(value, prefix) for value in list_field(record, "expected_frames", prefix)),
        expected_state_delta=tuple(state_from_dict(value, prefix) for value in list_field(record, "expected_state_delta", prefix)),
        expected_answer=str(record.get("expected_answer") or "").strip(),
        action_result=action_result_from_dict(raw_action, prefix) if isinstance(raw_action, dict) else None,
        feedback_diagnosis=feedback_diagnosis_from_dict(raw_feedback, prefix) if isinstance(raw_feedback, dict) else None,
        source=str(record.get("source") or "training").strip() or "training",
        split=str(record.get("split") or "train").strip() or "train",
    )


def acts_with_runtime_source(acts: tuple[PragmaticAct, ...], score: float) -> tuple[PragmaticAct, ...]:
    return tuple(
        PragmaticAct(
            act=act.act,
            target=act.target,
            qualifiers=act.qualifiers,
            confidence=min(1.0, max(act.confidence, score)),
            source=act.source,
        )
        for act in acts
    )


def pragmatic_act_matches(predicted: PragmaticAct, expected: PragmaticAct) -> bool:
    return (
        predicted.act == expected.act
        and predicted.target == expected.target
        and set(predicted.qualifiers) >= set(expected.qualifiers)
    )


def pragmatic_act_signature(act: PragmaticAct) -> tuple[Any, ...]:
    return (act.act, act.target, act.qualifiers)


def pragmatic_act_from_dict(record: Any, prefix: str) -> PragmaticAct:
    if not isinstance(record, dict):
        raise ValueError(f"{prefix} pragmatic act entries must be objects.")
    act = str(record.get("act") or "").strip()
    if not act:
        raise ValueError(f"{prefix} pragmatic act requires act.")
    raw_qualifiers = record.get("qualifiers", ())
    if raw_qualifiers is None:
        raw_qualifiers = ()
    if not isinstance(raw_qualifiers, list):
        raise ValueError(f"{prefix} pragmatic act qualifiers must be a list.")
    confidence = float(record.get("confidence") or 1.0)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{prefix} pragmatic act confidence must be between 0 and 1.")
    return PragmaticAct(
        act=act,
        target=str(record.get("target") or "").strip(),
        qualifiers=tuple(str(value).strip() for value in raw_qualifiers if str(value).strip()),
        confidence=confidence,
        source=str(record.get("source") or "episode").strip() or "episode",
    )


def pragmatic_act_to_dict(act: PragmaticAct) -> dict[str, Any]:
    return {
        "act": act.act,
        "target": act.target,
        "qualifiers": list(act.qualifiers),
        "confidence": act.confidence,
        "source": act.source,
    }


def state_from_dict(record: Any, prefix: str) -> State:
    if not isinstance(record, dict):
        raise ValueError(f"{prefix} state entries must be objects.")
    name = str(record.get("name") or "").strip()
    left = str(record.get("left") or "").strip()
    right = str(record.get("right") or "").strip()
    if not name or not left:
        raise ValueError(f"{prefix} states require name and left.")
    return State(name, left, right, str(record.get("source") or "").strip() or None)


def state_to_dict(state: State) -> dict[str, str]:
    record = {"name": state.name, "left": state.left, "right": state.right}
    if state.source is not None:
        record["source"] = state.source
    return record


def entity_from_dict(record: Any, prefix: str) -> Entity:
    if not isinstance(record, dict):
        raise ValueError(f"{prefix} entity entries must be objects.")
    role = str(record.get("role") or "").strip()
    name = str(record.get("name") or "").strip()
    if not role or not name:
        raise ValueError(f"{prefix} entities require role and name.")
    return Entity(role, name)


def entity_to_dict(entity: Entity) -> dict[str, str]:
    return {"role": entity.role, "name": entity.name}


def frame_from_dict(record: Any, prefix: str) -> Frame:
    if not isinstance(record, dict):
        raise ValueError(f"{prefix} frame entries must be objects.")
    frame_type = str(record.get("frame_type") or "").strip()
    raw_roles = record.get("roles")
    if not frame_type or not isinstance(raw_roles, dict):
        raise ValueError(f"{prefix} frames require frame_type and roles object.")
    frame_id = str(record.get("frame_id") or "expected").strip() or "expected"
    return Frame(
        frame_id=frame_id,
        frame_type=frame_type,
        time=int(record.get("time") or 0),
        roles=tuple(Role(frame_id, str(name), str(value)) for name, value in raw_roles.items()),
    )


def frame_to_dict(frame: Frame) -> dict[str, Any]:
    return {
        "frame_id": frame.frame_id,
        "frame_type": frame.frame_type,
        "time": frame.time,
        "roles": {role.name: role.value for role in frame.roles},
    }


def dialogue_turn_from_dict(record: Any, prefix: str) -> DialogueTurn:
    if not isinstance(record, dict):
        raise ValueError(f"{prefix} previous_turns entries must be objects.")
    speaker = str(record.get("speaker") or "").strip()
    text = str(record.get("text") or "").strip()
    if not speaker or not text:
        raise ValueError(f"{prefix} previous_turns entries require speaker and text.")
    return DialogueTurn(speaker=speaker, text=text, turn=int(record.get("turn") or 0))


def dialogue_turn_to_dict(turn: DialogueTurn) -> dict[str, Any]:
    return {"speaker": turn.speaker, "text": turn.text, "turn": turn.turn}


def feedback_diagnosis_from_dict(record: Any, prefix: str) -> FeedbackDiagnosis:
    if not isinstance(record, dict):
        raise ValueError(f"{prefix} feedback_diagnosis must be an object.")
    error_type = str(record.get("error_type") or "").strip()
    if not error_type:
        raise ValueError(f"{prefix} feedback_diagnosis requires error_type.")
    return FeedbackDiagnosis(
        error_type=error_type,
        user_feedback=str(record.get("user_feedback") or "").strip(),
        previous_answer=str(record.get("previous_answer") or "").strip(),
        original_understanding=str(record.get("original_understanding") or "").strip(),
        correct_structure=str(record.get("correct_structure") or "").strip(),
        generalization_target=str(record.get("generalization_target") or "").strip(),
    )


def feedback_diagnosis_to_dict(diagnosis: FeedbackDiagnosis) -> dict[str, str]:
    return {
        "error_type": diagnosis.error_type,
        "user_feedback": diagnosis.user_feedback,
        "previous_answer": diagnosis.previous_answer,
        "original_understanding": diagnosis.original_understanding,
        "correct_structure": diagnosis.correct_structure,
        "generalization_target": diagnosis.generalization_target,
    }


def action_result_from_dict(record: Any, prefix: str) -> ActionResult:
    if not isinstance(record, dict):
        raise ValueError(f"{prefix} action_result must be an object.")
    action = str(record.get("action") or "").strip()
    status = str(record.get("status") or "").strip()
    if not action or not status:
        raise ValueError(f"{prefix} action_result requires action and status.")
    return ActionResult(
        action=action,
        status=status,
        observation=str(record.get("observation") or "").strip(),
        user_satisfaction=str(record.get("user_satisfaction") or "").strip(),
    )


def action_result_to_dict(result: ActionResult) -> dict[str, str]:
    return {
        "action": result.action,
        "status": result.status,
        "observation": result.observation,
        "user_satisfaction": result.user_satisfaction,
    }


def list_field(record: dict[str, Any], field_name: str, prefix: str) -> tuple[Any, ...]:
    raw = record.get(field_name)
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{prefix} {field_name} must be a list.")
    return tuple(raw)


def field_to_tuple(value: Any, field_name: str, prefix: str) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, list):
        raise ValueError(f"{prefix} {field_name} must be a string or list of strings.")
    return clean_string_tuple(value)


def clean_string_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values if str(value).strip())


def normalize_episode_text(text: str) -> str:
    return normalize_question(text).replace("。", "").replace("，", "").replace(",", "").strip()


def episode_score(example: str, text: str) -> float:
    if not example or not text:
        return 0.0
    if example == text:
        return 1.0
    if len(text) >= 3 and text in example:
        return 0.95
    if len(example) <= 4:
        return 0.0
    if example in text:
        return 0.95
    example_units = character_bigrams(example)
    text_units = character_bigrams(text)
    if not example_units or not text_units:
        return 0.0
    return len(example_units & text_units) / len(example_units | text_units)


def character_bigrams(text: str) -> set[str]:
    if len(text) <= 1:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}
