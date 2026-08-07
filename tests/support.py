from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from struct_llm.errors import ParseError
from struct_llm.capabilities import CognitiveCapabilities
from struct_llm.comprehension.structure_helpers import frame_from_roles, with_time
from struct_llm.comprehension.intent_dataset import (
    append_intent_record,
    build_intent_record,
    intent_record_from_dict,
    load_intent_jsonl,
)
from struct_llm.kernel import default_capabilities, parse_text, predict as _predict
from struct_llm.neural import (
    InMemoryNeuralBoundaryModel,
    NeuralQueryParser,
    NeuralStatementParser,
    load_neural_boundary_model,
    with_neural_boundary,
)
from my_neural import make_model, train_summary
from struct_llm.motor.feedback import (
    LearningPaths,
    accept_query_suggestion,
    assess_query_uncertainty,
    confidence_band,
    save_chat_memory_feedback,
    save_direct_memory_feedback,
    save_manual_query_feedback,
    save_manual_statement_feedback,
    save_new_dialog_capability_feedback,
    save_unrecognized_feedback,
    suggest_query_feedback,
)
from struct_llm.motor.dialogue import (
    LearnedDialogActAnswerer,
    default_learned_dialog_answerer,
    save_manual_dialog_answer_feedback,
)
from struct_llm.motor.feedback import save_memory_knowledge_feedback
from struct_llm.comprehension.intent import InMemoryIntentAnalyzer, evaluate_intent_analyzer
from struct_llm.motor.learning_queue import load_unrecognized_jsonl
from struct_llm.memory.long_term import load_memory_model
from struct_llm.memory.knowledge import (
    LearnedMemoryKnowledgeAnswerer,
    default_learned_memory_knowledge_answerer,
    load_memory_knowledge_model,
)
from struct_llm.comprehension.query import (
    evaluate_query_parser,
    load_query_jsonl,
)
from struct_llm.comprehension.statement import (
    EntitySlot,
    FrameTemplate,
    evaluate_statement_parser,
    linearize_statement_result,
    load_statement_jsonl,
    normalize_statement_text,
    statement_example_from_dict,
)
from struct_llm.world.event_schema import EVENT_SCHEMAS, frame_matches_qualifiers, states_for_frame_schema
from struct_llm.structure import Entity, Intention, Query, Structure
from struct_llm.structure import State
from struct_llm.neural.query_classifier import (
    LoadedNeuralQueryParser,
    default_neural_query_parser,
    train_query_neural_model,
)
from struct_llm.neural.statement_classifier import (
    LoadedNeuralStatementParser,
    default_neural_statement_parser,
    train_statement_neural_model,
)


def predict(text: str, capabilities: CognitiveCapabilities | None = None):
    try:
        prediction = _predict(text, capabilities)
    except Exception as error:
        print(f"{text} -> ERROR: {error}", flush=True)
        raise
    print(f"{text} -> {prediction.answer}", flush=True)
    return prediction


