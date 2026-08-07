.PHONY: ask chat remember knowledge train test check

# Daily entrypoints stay intentionally small; lower-level CLI commands live
# under `uv run struct <command>` when an experiment needs them.
NEURAL_PROVIDER ?= my_neural:make_model
ASK_FLAGS = --neural-provider "$(NEURAL_PROVIDER)" --learn-on-fail --memory-model data/memory_model.json --memory-knowledge-model data/memory_knowledge_model.json
MEMORY_FLAGS = --memory-direct-data data/memory_direct_examples.jsonl --memory-chat-data data/memory_chat_examples.jsonl --output data/memory_model.json
KNOWLEDGE_FLAGS = --memory-knowledge-data data/memory_knowledge_examples.jsonl --output data/memory_knowledge_model.json

# Ask one question. Pass TEXT="..." for a single turn.
ask:
	uv run struct ask $(ASK_FLAGS) "$(TEXT)"

# Start the same reasoner in interactive mode.
chat:
	uv run struct ask $(ASK_FLAGS)

# Persist stable user/world facts into the long-term memory model.
remember:
	uv run struct add-memory --neural-provider "$(NEURAL_PROVIDER)" --memory-model data/memory_model.json "$(TEXT)"

# Add verified question-answer knowledge. Use FILE=... for batch JSONL import,
# otherwise pass QUESTION="..." ANSWER="...".
knowledge:
	@if [ -n "$(FILE)" ]; then \
		uv run struct add-knowledge --neural-provider "$(NEURAL_PROVIDER)" --memory-knowledge-data data/memory_knowledge_examples.jsonl --memory-knowledge-model data/memory_knowledge_model.json --file "$(FILE)"; \
	else \
		uv run struct add-knowledge --neural-provider "$(NEURAL_PROVIDER)" --memory-knowledge-data data/memory_knowledge_examples.jsonl --memory-knowledge-model data/memory_knowledge_model.json "$(QUESTION)" --answer "$(ANSWER)" --source curated; \
	fi

# Rebuild all runtime artifacts after JSONL training data changes.
train:
	uv run struct train
	uv run struct compile-memory $(MEMORY_FLAGS)
	uv run struct compile-memory-knowledge $(KNOWLEDGE_FLAGS)
	uv run struct compile-dialog-answer --dialog-answer-data data/dialog_answer_examples.jsonl --output data/dialog_answer_model.json

# Run the regression suite without rebuilding artifacts.
test:
	uv run python -m unittest discover -q

# Full local verification: refresh learned artifacts, then test.
check: train test
