#!/bin/bash

uv run --group lint ruff check --fix "$@"
