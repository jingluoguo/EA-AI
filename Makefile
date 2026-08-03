.PHONY: sync demo ask ask-neural chat chat-neural data test train

TEXT ?= 小明把钥匙放进盒子。盒子被带到厨房。钥匙在哪里？

sync:
	uv sync

demo:
	uv run struct-demo

ask:
	uv run struct-ask "$(TEXT)"

ask-neural:
	uv run --extra neural struct-ask-neural "$(TEXT)"

chat:
	uv run struct-ask

chat-neural:
	uv run --extra neural struct-ask-neural

data:
	uv run struct-make-dataset

test:
	uv run python -m unittest discover

train:
	uv run --extra neural struct-train-tiny
