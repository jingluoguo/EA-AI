.PHONY: sync demo ask chat train-neural compile model check intent-example intent-eval query-eval statement-eval test

NEURAL_PROVIDER ?= my_neural:make_model

sync:
	uv sync

demo:
	uv run struct-demo

ask:
	uv run struct-ask --neural-provider "$(NEURAL_PROVIDER)" --learn-on-fail "$(TEXT)"

chat:
	uv run struct-ask --neural-provider "$(NEURAL_PROVIDER)" --learn-on-fail

train-neural:
	uv run struct-train-neural

compile: model

model:
	uv run struct-compile-query
	uv run struct-compile-statement
	uv run struct-compile-dialog-answer

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
