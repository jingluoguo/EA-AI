from __future__ import annotations

from ..capabilities import Answerer, RuleInferer
from ..errors import ParseError
from ..structure import Query, Structure
from .selectors import require_query, structure_with_query


def infer_rules(
    structure: Structure,
    inferers: tuple[RuleInferer, ...] | None = None,
) -> tuple[str, ...]:
    if structure.query is None:
        return ()

    if inferers is None:
        from .rules import DEFAULT_RULE_INFERERS

        inferers = DEFAULT_RULE_INFERERS

    rules: list[str] = []
    for inferer in inferers:
        rule = inferer(structure)
        if rule is not None:
            rules.append(rule)
    return tuple(rules)


def answer_from_structure(
    structure: Structure,
    answerers: tuple[Answerer, ...] | None = None,
) -> str:
    if answerers is None:
        from .answers import DEFAULT_ANSWERERS

        answerers = DEFAULT_ANSWERERS

    for answerer in answerers:
        answer = answerer(structure)
        if answer is not None:
            return answer

    raise ParseError(f"No rule can answer structure: {structure.linearize()}")


def answer_compound_query(structure: Structure) -> str | None:
    if "compound_query" not in set(structure.rules):
        return None
    query = require_query(structure)
    answers = [answer_subquery(structure, subquery).rstrip("。！？!?") for subquery in query.subqueries]
    return "；".join(answers) + "。"


def answer_subquery(structure: Structure, query: Query) -> str:
    substructure = structure_with_query(structure, query)
    substructure = Structure(
        entities=substructure.entities,
        relations=substructure.relations,
        events=substructure.events,
        rules=infer_rules(substructure),
        query=substructure.query,
        frames=substructure.frames,
        states=substructure.states,
        intentions=substructure.intentions,
    )
    return answer_from_structure(substructure)
