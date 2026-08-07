from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..dataset_io import append_jsonl_object, file_sha256, load_jsonl_objects
from ..errors import ParseError
from ..structure import Entity, Query
from ..capabilities import QueryParser
from ..perception.lexer import split_query_candidate
from ..perception.normalizer import is_question_noise, normalize_question


QUERY_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "query_examples.jsonl"
QUERY_MODEL_SCHEMA = "struct_llm.query_model.v1"
UNRESOLVED_REFERENCE_WORDS = ("前者", "后者", "前一个", "后一个", "它", "他", "她")


@dataclass(frozen=True)
class EntityExample:
    role: str
    name: str


@dataclass(frozen=True)
class QueryTrainingExample:
    question: str
    entities: tuple[EntityExample, ...]
    query: Query
    abstract_question: str
    source: str = "training"
    split: str = "train"


@dataclass(frozen=True)
class CompiledQueryPattern:
    abstract_question: str
    entities: tuple[EntityExample, ...]
    query: Query
    feature_units: tuple[str, ...]
    support: int = 1


@dataclass(frozen=True)
class CompiledQueryModel:
    schema: str
    source_sha256: str
    example_count: int
    patterns: tuple[CompiledQueryPattern, ...]


@dataclass(frozen=True)
class QueryEvaluationResult:
    total: int
    matched: int

    @property
    def accuracy(self) -> float:
        return self.matched / self.total if self.total else 0.0


def compile_query_examples(
    examples: tuple[QueryTrainingExample, ...],
    source_sha256: str = "",
) -> CompiledQueryModel:
    grouped: dict[tuple[Any, ...], CompiledQueryPattern] = {}
    for example in examples:
        abstract = example.abstract_question.strip() or abstract_question(example.question, example.entities)
        feature_units = tuple(sorted(character_bigrams(abstract)))
        key = (
            abstract,
            tuple((entity.role, entity.name) for entity in example.entities),
            query_signature(example.query),
        )
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = CompiledQueryPattern(
                abstract_question=abstract,
                entities=example.entities,
                query=example.query,
                feature_units=feature_units,
                support=1,
            )
            continue
        grouped[key] = CompiledQueryPattern(
            abstract_question=previous.abstract_question,
            entities=previous.entities,
            query=previous.query,
            feature_units=previous.feature_units,
            support=previous.support + 1,
        )
    return CompiledQueryModel(
        schema=QUERY_MODEL_SCHEMA,
        source_sha256=source_sha256,
        example_count=len(examples),
        patterns=tuple(sorted(grouped.values(), key=lambda pattern: (-pattern.support, pattern.abstract_question))),
    )


def query_pattern_from_dict(record: Any) -> CompiledQueryPattern:
    if not isinstance(record, dict):
        raise ValueError("Query model pattern entries must be objects.")
    abstract = str(record.get("abstract_question") or "").strip()
    if not abstract:
        raise ValueError("Query model pattern requires abstract_question.")
    raw_entities = record.get("entities", ())
    if not isinstance(raw_entities, list):
        raise ValueError("Query model pattern entities must be a list.")
    raw_query = record.get("query")
    if not isinstance(raw_query, dict):
        raise ValueError("Query model pattern query must be an object.")
    raw_units = record.get("feature_units", ())
    if not isinstance(raw_units, list):
        raise ValueError("Query model pattern feature_units must be a list.")
    return CompiledQueryPattern(
        abstract_question=abstract,
        entities=tuple(entity_example_from_dict(value, "Query model pattern") for value in raw_entities),
        query=query_from_dict(raw_query, "Query model pattern"),
        feature_units=tuple(str(value) for value in raw_units if str(value)),
        support=int(record.get("support") or 1),
    )


def query_pattern_to_dict(pattern: CompiledQueryPattern) -> dict[str, Any]:
    return {
        "abstract_question": pattern.abstract_question,
        "entities": [entity_example_to_dict(entity) for entity in pattern.entities],
        "query": query_to_dict(pattern.query),
        "feature_units": list(pattern.feature_units),
        "support": pattern.support,
    }


def resolve_query_candidates(
    candidates: list[str],
    entities: tuple[Entity, ...],
    parsers: tuple[QueryParser, ...],
) -> Query | None:
    parsed_queries: list[Query] = []
    errors: list[ParseError] = []
    meaningful_candidates: list[str] = []

    for candidate in candidates:
        full_query = resolve_query_candidate(candidate, entities, parsers)
        if is_question_noise(candidate) and not noise_candidate_has_meaningful_query(full_query):
            continue
        meaningful_candidates.append(candidate)

        fragments = split_query_candidate(candidate)
        if full_query is not None and (len(fragments) == 1 or query_candidate_is_learned_unit(candidate, entities, parsers)):
            parsed_queries.append(full_query)
            continue

        for fragment in fragments:
            if is_question_noise(fragment):
                continue
            try:
                query = resolve_query_candidate_or_raise(fragment, entities, parsers)
            except ParseError as error:
                errors.append(error)
                continue
            parsed_queries.append(query)

    meaningful_queries = [
        query
        for query in parsed_queries
        if not (query.intent == "dialog_act" and query.target in {"greeting", "thanks", "farewell", "clarification"})
    ]
    if meaningful_queries:
        parsed_queries = meaningful_queries
    parsed_queries = dedupe_queries(parsed_queries)

    if len(parsed_queries) > 1:
        return Query("compound", "multi", subqueries=tuple(parsed_queries))
    if len(parsed_queries) == 1:
        return parsed_queries[0]

    combined = "，".join(meaningful_candidates)
    if combined:
        try:
            return resolve_query_candidate_or_raise(combined, entities, parsers)
        except ParseError:
            pass
    if errors:
        raise errors[-1]
    return None


def noise_candidate_has_meaningful_query(query: Query | None) -> bool:
    if query is None:
        return False
    if query.intent != "dialog_act":
        return True
    return query.target not in {"greeting", "capabilities"}


def resolve_query_candidate(
    sentence: str,
    entities: tuple[Entity, ...],
    parsers: tuple[QueryParser, ...],
) -> Query | None:
    normalized = normalize_question(sentence)
    if contains_unresolved_reference(normalized, entities):
        return None
    for parser in parsers:
        for candidate in dict.fromkeys((sentence, normalized)):
            query = parser(candidate, entities)
            if query is not None:
                return query
    return None


def contains_unresolved_reference(sentence: str, entities: tuple[Entity, ...]) -> bool:
    return any(word in sentence for word in UNRESOLVED_REFERENCE_WORDS)


def resolve_query_candidate_or_raise(
    sentence: str,
    entities: tuple[Entity, ...],
    parsers: tuple[QueryParser, ...],
) -> Query:
    query = resolve_query_candidate(sentence, entities, parsers)
    if query is None:
        raise ParseError(f"Cannot learn query structure from question: {sentence}")
    return query


def query_candidate_is_learned_unit(
    sentence: str,
    entities: tuple[Entity, ...],
    parsers: tuple[QueryParser, ...],
) -> bool:
    for parser in parsers:
        structural_unit = getattr(parser, "candidate_is_structural_unit", None)
        if structural_unit is not None and structural_unit(sentence, entities):
            return True

        best_match = getattr(parser, "best_match", None)
        if best_match is None:
            continue
        best = best_match(sentence, entities)
        if best is None or best[0] < 0.96:
            continue
        # A high-scoring template should not swallow a comma-separated
        # compound query unless the model is nearly exact on the whole input.
        if len(split_query_candidate(sentence)) > 1 and best[0] < 1.0:
            continue
        return True
    return False


def suggest_query_pattern(
    sentence: str,
    entities: tuple[Entity, ...],
    parsers: tuple[QueryParser, ...],
    min_score: float = 0.12,
) -> tuple[float, CompiledQueryPattern] | None:
    suggestions: list[tuple[float, CompiledQueryPattern]] = []
    for parser in parsers:
        best_match = getattr(parser, "best_match", None)
        if best_match is not None:
            best = best_match(sentence, entities)
            if best is not None:
                score, pattern = best
                if score >= min_score:
                    suggestions.append((score, pattern))
    if not suggestions:
        return None
    return max(suggestions, key=lambda item: item[0])


def dedupe_queries(queries: list[Query]) -> list[Query]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[Query] = []
    for query in queries:
        signature = query_signature(query)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(query)
    return deduped


def evaluate_query_parser(
    parser: QueryParser,
    examples: tuple[QueryTrainingExample, ...],
) -> QueryEvaluationResult:
    matched = 0
    for example in examples:
        runtime_entities = tuple(Entity(entity.role, entity.name) for entity in example.entities)
        query = parser(example.question, runtime_entities)
        expected = instantiate_query(example.query, example.entities, example.question, runtime_entities)
        if query is not None and query.linearize() == expected.linearize():
            matched += 1
    return QueryEvaluationResult(total=len(examples), matched=matched)


def load_query_jsonl(path: str | Path) -> tuple[QueryTrainingExample, ...]:
    return tuple(
        query_example_from_dict(raw_record, line_number=line_number)
        for line_number, raw_record in enumerate(load_jsonl_objects(path, "query"), start=1)
    )


def append_query_record(path: str | Path, record: dict[str, Any]) -> QueryTrainingExample:
    example = query_example_from_dict(record)
    append_jsonl_object(path, query_example_to_record(example))
    return example


def build_query_record(
    question: str,
    intent: str,
    target: str,
    *,
    entities: tuple[EntityExample, ...] = (),
    qualifiers: tuple[str, ...] = (),
    source: str = "human_feedback",
    split: str = "train",
) -> dict[str, Any]:
    return {
        "question": question.strip(),
        "entities": [entity_example_to_dict(entity) for entity in entities],
        "query": {
            "intent": intent.strip(),
            "target": target.strip(),
            "qualifiers": [value.strip() for value in qualifiers if value.strip()],
        },
        "source": source.strip() or "human_feedback",
        "split": split.strip() or "train",
    }


def query_example_to_record(example: QueryTrainingExample) -> dict[str, Any]:
    return {
        "question": example.question,
        "entities": [entity_example_to_dict(entity) for entity in example.entities],
        "query": query_to_dict(example.query),
        "source": example.source,
        "split": example.split,
    }


def query_example_from_dict(record: dict[str, Any], *, line_number: int | None = None) -> QueryTrainingExample:
    prefix = f"Query example at line {line_number}" if line_number is not None else "Query example"
    question = str(record.get("question") or record.get("text") or "").strip()
    if not question:
        raise ValueError(f"{prefix} requires a question or text field.")

    raw_entities = record.get("entities", ())
    if not isinstance(raw_entities, list):
        raise ValueError(f"{prefix} entities must be a list.")
    entities = tuple(entity_example_from_dict(value, prefix) for value in raw_entities)

    raw_query = record.get("query")
    if not isinstance(raw_query, dict):
        raise ValueError(f"{prefix} query must be an object.")
    query = query_from_dict(raw_query, prefix)
    abstract = str(record.get("abstract_question") or "").strip() or abstract_question(question, entities)
    return QueryTrainingExample(
        question=question,
        entities=entities,
        query=query,
        abstract_question=abstract,
        source=str(record.get("source") or "training").strip() or "training",
        split=str(record.get("split") or "train").strip() or "train",
    )


def entity_example_from_dict(record: Any, prefix: str) -> EntityExample:
    if not isinstance(record, dict):
        raise ValueError(f"{prefix} entity entries must be objects.")
    role = str(record.get("role") or "").strip()
    name = str(record.get("name") or "").strip()
    if not role or not name:
        raise ValueError(f"{prefix} entity entries require role and name.")
    return EntityExample(role, name)


def query_from_dict(record: dict[str, Any], prefix: str) -> Query:
    intent = str(record.get("intent") or "").strip()
    target = str(record.get("target") or "").strip()
    if not intent or not target:
        raise ValueError(f"{prefix} query requires intent and target.")
    raw_qualifiers = record.get("qualifiers", ())
    if raw_qualifiers is None:
        raw_qualifiers = ()
    if not isinstance(raw_qualifiers, list):
        raise ValueError(f"{prefix} query.qualifiers must be a list.")
    raw_subqueries = record.get("subqueries", [])
    if raw_subqueries is None:
        raw_subqueries = ()
    if not isinstance(raw_subqueries, list):
        raise ValueError(f"{prefix} query.subqueries must be a list.")
    return Query(
        intent,
        target,
        tuple(str(value).strip() for value in raw_qualifiers if str(value).strip()),
        tuple(query_from_dict(value, prefix) for value in raw_subqueries if isinstance(value, dict)),
    )


def entity_example_to_dict(entity: EntityExample) -> dict[str, str]:
    return {"role": entity.role, "name": entity.name}


def query_to_dict(query: Query) -> dict[str, Any]:
    return {
        "intent": query.intent,
        "target": query.target,
        "qualifiers": list(query.qualifiers),
        "subqueries": [query_to_dict(subquery) for subquery in query.subqueries],
    }


def query_signature(query: Query) -> tuple[Any, ...]:
    return (
        query.intent,
        query.target,
        query.qualifiers,
        tuple(query_signature(subquery) for subquery in query.subqueries),
    )


def instantiate_query(
    template: Query,
    example_entities: tuple[EntityExample, ...],
    sentence: str,
    runtime_entities: tuple[Entity, ...],
    *,
    allow_example_slot_values: bool = True,
) -> Query:
    return Query(
        template.intent,
        instantiate_value(template.target, example_entities, sentence, runtime_entities, allow_example_slot_values=allow_example_slot_values),
        tuple(
            instantiate_qualifier(value, example_entities, sentence, runtime_entities, allow_example_slot_values=allow_example_slot_values)
            for value in template.qualifiers
        ),
        tuple(
            instantiate_query(
                subquery,
                example_entities,
                sentence,
                runtime_entities,
                allow_example_slot_values=allow_example_slot_values,
            )
            for subquery in template.subqueries
        ),
    )


def instantiate_qualifier(
    qualifier: str,
    example_entities: tuple[EntityExample, ...],
    sentence: str,
    runtime_entities: tuple[Entity, ...],
    *,
    allow_example_slot_values: bool = True,
) -> str:
    if "=" not in qualifier:
        return instantiate_value(qualifier, example_entities, sentence, runtime_entities, allow_example_slot_values=allow_example_slot_values)
    key, value = qualifier.split("=", 1)
    return f"{key}={instantiate_value(value, example_entities, sentence, runtime_entities, allow_example_slot_values=allow_example_slot_values)}"


def instantiate_value(
    value: str,
    example_entities: tuple[EntityExample, ...],
    sentence: str,
    runtime_entities: tuple[Entity, ...],
    *,
    allow_example_slot_values: bool = True,
) -> str:
    if "$" not in value:
        return value
    runtime_matches = entities_ordered_in_text(sentence, entity_examples_from_runtime(runtime_entities))
    placeholders = placeholder_entities(example_entities)
    example_matches = entities_ordered_in_text(value, placeholders)
    if not example_matches:
        return value

    resolved = value
    for example_entity in example_matches:
        runtime_entity = runtime_entity_for(example_entity, runtime_matches)
        if runtime_entity is not None:
            resolved = resolved.replace(example_entity.name, runtime_entity.name, 1)
            continue
        if not allow_example_slot_values:
            unique_runtime_entity = unique_runtime_entity_for(example_entity, entity_examples_from_runtime(runtime_entities))
            if unique_runtime_entity is not None:
                resolved = resolved.replace(example_entity.name, unique_runtime_entity.name, 1)
            continue
        original = original_entity_for_placeholder(example_entity, placeholders, example_entities)
        if original is not None:
            resolved = resolved.replace(example_entity.name, original.name, 1)
    return resolved.replace("$", "")


def runtime_entity_for(example_entity: EntityExample, runtime_matches: tuple[EntityExample, ...]) -> EntityExample | None:
    role_matches = [entity for entity in runtime_matches if roles_compatible(example_entity.role, entity.role)]
    if not role_matches:
        return None
    occurrence = placeholder_occurrence(example_entity)
    if occurrence < len(role_matches):
        return role_matches[occurrence]
    return role_matches[-1]


def unique_runtime_entity_for(
    example_entity: EntityExample,
    runtime_entities: tuple[EntityExample, ...],
) -> EntityExample | None:
    role_matches = [entity for entity in runtime_entities if roles_compatible(example_entity.role, entity.role)]
    if len(role_matches) != 1:
        return None
    return role_matches[0]


def placeholder_occurrence(entity: EntityExample) -> int:
    if "#" not in entity.name:
        return 0
    try:
        return max(0, int(entity.name.rsplit("#", 1)[1]) - 1)
    except ValueError:
        return 0


def original_entity_for_placeholder(
    placeholder: EntityExample,
    placeholders: tuple[EntityExample, ...],
    originals: tuple[EntityExample, ...],
) -> EntityExample | None:
    for index, candidate in enumerate(placeholders):
        if candidate == placeholder and index < len(originals):
            return originals[index]
    return None


def abstract_question(question: str, entities: tuple[EntityExample, ...]) -> str:
    normalized = normalize_question(question).strip()
    for entity in sorted(entities, key=lambda value: len(value.name), reverse=True):
        normalized = normalized.replace(entity.name, role_token(entity.role))
    return normalized


def query_example_score(example: str, sentence: str) -> float:
    if example == sentence:
        return 1.0
    if not example or not sentence:
        return 0.0
    return max(character_bigram_similarity(example, sentence), character_set_similarity(example, sentence))


def query_pattern_score(pattern: CompiledQueryPattern, sentence: str) -> float:
    if pattern.abstract_question == sentence:
        return 1.0
    if not pattern.abstract_question or not sentence:
        return 0.0
    extracted_slots = extract_role_token_slots(pattern.abstract_question, sentence)
    if extracted_slots is not None:
        # Prefer the pattern that explains more structural slots when several
        # templates share the same question tail, such as belief vs. location.
        return min(0.99, 0.95 + 0.01 * len(extracted_slots))
    sentence_units = character_bigrams(sentence)
    feature_units = set(pattern.feature_units)
    bigram_score = 0.0 if not sentence_units or not feature_units else len(feature_units & sentence_units) / len(feature_units | sentence_units)
    return max(bigram_score, character_set_similarity(pattern.abstract_question, sentence))


def character_bigram_similarity(left: str, right: str) -> float:
    left_units = character_bigrams(left)
    right_units = character_bigrams(right)
    if not left_units or not right_units:
        return 0.0
    return len(left_units & right_units) / len(left_units | right_units)


def character_set_similarity(left: str, right: str) -> float:
    left_units = set(left)
    right_units = set(right)
    if not left_units or not right_units:
        return 0.0
    return len(left_units & right_units) / len(left_units | right_units)


def character_bigrams(text: str) -> set[str]:
    if len(text) <= 1:
        return {text} if text else set()
    return {text[index : index + 2] for index in range(len(text) - 1)}


def infer_entities_from_abstract_pattern(
    template: str,
    abstract_sentence: str,
    runtime_entities: tuple[Entity, ...],
) -> tuple[Entity, ...]:
    extracted = extract_role_token_slots(template, abstract_sentence)
    if extracted is None:
        return ()
    known_names = {entity.name for entity in runtime_entities}
    inferred: list[Entity] = []
    for role, value in extracted:
        if not value or (value.startswith("<") and value.endswith(">")) or value in known_names:
            continue
        inferred.append(Entity(role, value))
        known_names.add(value)
    return tuple(inferred)


def extract_role_token_slots(template: str, sentence: str) -> tuple[tuple[str, str], ...] | None:
    parts = split_role_token_template(template)
    if not template_has_literal_anchor(parts):
        return extract_unanchored_role_token_slots(parts, sentence)
    slots: list[tuple[str, str]] = []
    position = 0
    for index, part in enumerate(parts):
        role = role_from_token(part)
        if role is not None:
            next_literal = next((value for value in parts[index + 1 :] if role_from_token(value) is None and value), "")
            if next_literal:
                next_position = sentence.find(next_literal, position)
                if next_position < position:
                    return None
                value = sentence[position:next_position]
                position = next_position
            else:
                value = sentence[position:]
                position = len(sentence)
            if not value:
                return None
            value_role = role_from_token(value)
            if value_role is not None:
                if not roles_compatible(role, value_role):
                    return None
            elif slot_value_is_unresolved_reference(value):
                return None
            slots.append((role, value))
            continue
        if not part:
            continue
        found = sentence.find(part, position)
        if found < position:
            return None
        position = found + len(part)
    if position > len(sentence):
        return None
    return tuple(slots)


def template_has_literal_anchor(parts: tuple[str, ...]) -> bool:
    return any(role_from_token(part) is None and part.strip() for part in parts)


def extract_unanchored_role_token_slots(parts: tuple[str, ...], sentence: str) -> tuple[tuple[str, str], ...] | None:
    if len(parts) != 1:
        return None
    role = role_from_token(parts[0])
    if role is None:
        return None
    sentence_role = role_from_token(sentence)
    if sentence_role is None or not roles_compatible(role, sentence_role):
        return None
    return ((role, sentence),)


def split_role_token_template(template: str) -> tuple[str, ...]:
    parts: list[str] = []
    index = 0
    while index < len(template):
        if template[index] != "<":
            next_token = template.find("<", index)
            if next_token < 0:
                parts.append(template[index:])
                break
            parts.append(template[index:next_token])
            index = next_token
            continue
        end = template.find(">", index + 1)
        if end < 0:
            parts.append(template[index:])
            break
        parts.append(template[index : end + 1])
        index = end + 1
    return tuple(part for part in parts if part != "")


def role_from_token(value: str) -> str | None:
    if not (value.startswith("<") and value.endswith(">")):
        return None
    role = value[1:-1].strip()
    if not role:
        return None
    return role


def slot_value_is_unresolved_reference(value: str) -> bool:
    normalized = normalize_question(value).strip()
    return normalized in UNRESOLVED_REFERENCE_WORDS


def entity_examples_from_runtime(entities: tuple[Entity, ...]) -> tuple[EntityExample, ...]:
    return tuple(EntityExample(role=entity.role, name=entity.name) for entity in entities)


def placeholder_entities(entities: tuple[EntityExample, ...]) -> tuple[EntityExample, ...]:
    counts: dict[str, int] = {}
    placeholders: list[EntityExample] = []
    for entity in entities:
        counts[entity.role] = counts.get(entity.role, 0) + 1
        placeholders.append(EntityExample(entity.role, f"${entity.role}#{counts[entity.role]}"))
    return tuple(placeholders)


def entities_ordered_in_text(text: str, entities: tuple[EntityExample, ...]) -> tuple[EntityExample, ...]:
    matches = [entity for entity in entities if entity.name in text]
    return tuple(sorted(matches, key=lambda entity: text.index(entity.name)))


def role_token(role: str) -> str:
    if role in {"giver", "receiver"}:
        return "<person>"
    if role == "thing":
        return "<item>"
    return f"<{role}>"


def roles_compatible(example_role: str, runtime_role: str) -> bool:
    if example_role == runtime_role:
        return True
    return {example_role, runtime_role} <= {"item", "thing"}
