.PROJECT = boiling_learning
.TESTS_FOLDER = tests
.STUBS_FOLDER = typings

.AUTOFLAKE = $(shell poetry run autoflake --in-place --recursive --expand-star-imports --remove-duplicate-keys --remove-unused-variables --remove-all-unused-imports --ignore-init-module-imports $(.PROJECT) $(.STUBS_FOLDER) $(.TESTS_FOLDER))
.UNIMPORT = $(shell poetry run unimport --remove --gitignore --ignore-init --include-star-import $(.PROJECT) $(.STUBS_FOLDER) $(.TESTS_FOLDER))
.BLACK = $(shell poetry run black $(.PROJECT) $(.STUBS_FOLDER) $(.TESTS_FOLDER))
.ISORT = $(shell poetry run isort $(.PROJECT) $(.STUBS_FOLDER) $(.TESTS_FOLDER))
.FORMAT = $(foreach command,.AUTOFLAKE .UNIMPORT .BLACK .ISORT,$(call $(command)))

.READD = $(shell git update-index --again)
.CHECK = $(shell pre-commit run)

.PHONY: coverage
coverage:
	@poetry run coverage run --source=$(.PROJECT)/ -m pytest $(.TESTS_FOLDER)
	@poetry run coverage report -m

.PHONY: test
test:
	@poetry run pytest --doctest-modules $(.PROJECT) $(.TESTS_FOLDER) -vv

.PHONY: tox
tox:
	@poetry run tox

.PHONY: check
check:
	@$(call $(.CHECK))

.PHONY: typecheck
typecheck:
	@poetry run mypy $(.PROJECT)

.PHONY: check_valid_python
check_valid_python:
	@poetry run flake8 boiling_learning/* --count --select=E9,F63,F7,F82 --show-source --statistics

.PHONY: format
format:
	@$(call $(.FORMAT))

.PHONY: autofix
autofix:
	@-$(call $(.FORMAT))
	@$(call $(.READD))
	@-$(call $(.CHECK))
	@$(call $(.READD))

.PHONY: commit
commit: autofix
	@poetry run cz commit

.PHONY: release
release:
# for v1.0.0 and after, the following line should be used to bump the project version:
# 	cz bump
# before v1, use the following command, which maps the following bumps:
# 	MAJOR -> MINOR (v0.2.3 -> v0.3.0)
# 	MINOR or PATCH -> PATCH (v0.2.3 -> v0.2.4)
# effectively avoiding incrementing the MAJOR version number while the first
# stable version (v1.0.0) is not released
	poetry run cz bump --increment $(shell cz bump --dry-run | grep -q "MAJOR" && echo "MINOR" || echo "PATCH")
	git push
	git push --tags

.PHONY: run
run:
	poetry run python main.py
