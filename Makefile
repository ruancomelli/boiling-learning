.PROJECT = boiling_learning
.TESTS_FOLDER = tests

.PHONY: coverage
coverage:
	@uv run coverage run --source=$(.PROJECT)/ -m pytest $(.TESTS_FOLDER)
	@uv run coverage report -m

.PHONY: test
test:
	@uv run pytest --doctest-modules $(.PROJECT) $(.TESTS_FOLDER) -vv

.PHONY: format
format:
	@uv run --group format ruff format $(.PROJECT) $(.TESTS_FOLDER)

.PHONY: lint
lint:
	@uv run --group lint ruff check --fix $(.PROJECT) $(.TESTS_FOLDER)

.PHONY: typecheck
typecheck:
	@uv run mypy $(.PROJECT)

.PHONY: vulture
vulture:
	@uv run vulture --ignore-decorators @*.dispatch*,@*.instance* --ignore-names __*[!_][!_] $(.PROJECT) main.py

.PHONY: release
release:
# for v1.0.0 and after, the following line should be used to bump the project version:
# 	cz bump
# before v1, use the following command, which maps the following bumps:
# 	MAJOR -> MINOR (v0.2.3 -> v0.3.0)
# 	MINOR or PATCH -> PATCH (v0.2.3 -> v0.2.4)
# effectively avoiding incrementing the MAJOR version number while the first
# stable version (v1.0.0) is not released
	uv run cz bump --increment $(shell uv run cz bump --dry-run | grep -q "MAJOR" && echo "MINOR" || echo "PATCH")
	git push
	git push --tags
