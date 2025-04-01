#!/bin/bash

uv run --group vulture \
	vulture \
	--ignore-decorators @*.dispatch*,@*.instance* \
	--ignore-names __*[!_][!_] \
	"$@"