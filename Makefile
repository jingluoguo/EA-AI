.PHONY: sync demo ask chat remember remember-state remember-knowledge train-neural compile model check intent-example intent-eval query-eval statement-eval test

NEURAL_PROVIDER ?= my_neural:make_model

sync:
	uv sync

demo:
	uv run struct-demo

ask:
	uv run struct-ask --neural-provider "$(NEURAL_PROVIDER)" --learn-on-fail --memory-model data/memory_model.json --memory-knowledge-model data/memory_knowledge_model.json "$(TEXT)"

chat:
	uv run struct-ask --neural-provider "$(NEURAL_PROVIDER)" --learn-on-fail --memory-model data/memory_model.json --memory-knowledge-model data/memory_knowledge_model.json

remember:
	uv run struct-add-memory --neural-provider "$(NEURAL_PROVIDER)" --memory-model data/memory_model.json "$(TEXT)"

remember-state:
	uv run struct-add-memory --state "$(NAME)" "$(LEFT)" "$(RIGHT)" --memory-model data/memory_model.json

remember-knowledge:
	uv run struct-add-knowledge --neural-provider "$(NEURAL_PROVIDER)" --query-model data/query_model.json --memory-knowledge-data data/memory_knowledge_examples.jsonl --memory-knowledge-model data/memory_knowledge_model.json "$(QUESTION)" --answer "$(ANSWER)" --source curated

remember-knowledge-file:
	uv run struct-add-knowledge --neural-provider "$(NEURAL_PROVIDER)" --query-model data/query_model.json --memory-knowledge-data data/memory_knowledge_examples.jsonl --memory-knowledge-model data/memory_knowledge_model.json --file "$(FILE)"

train-neural:
	uv run struct-train-neural

compile: model

model:
	uv run struct-compile-query
	uv run struct-compile-statement
	uv run struct-compile-dialog-answer
	uv run struct-compile-memory
	uv run struct-compile-memory-knowledge

check: model
	uv run struct-eval-query --query-data data/query_examples.jsonl --query-model data/query_model.json
	uv run struct-eval-statement --statement-data data/statement_examples.jsonl --statement-model data/statement_model.json

intent-example:
	uv run struct-add-intent-example "$(OBSERVATION)" --subject "$(SUBJECT)" --goal "$(GOAL)"

intent-eval:
	uv run struct-eval-intent

query-eval:
	uv run struct-eval-query

statement-eval:
	uv run struct-eval-statement

test:
	uv run python -m unittest discover -q
