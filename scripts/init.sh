#!/bin/bash

uv sync --all-groups
uv run --group pre-commit pre-commit install
