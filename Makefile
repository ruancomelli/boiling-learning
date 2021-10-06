.PROJECT = boiling_learning
.AUTOFLAKE = $(shell autoflake --in-place --recursive --expand-star-imports --remove-duplicate-keys --remove-unused-variables --remove-all-unused-imports --ignore-init-module-imports $(.PROJECT) tests)
.UNIMPORT = $(shell unimport --remove --gitignore --ignore-init --include-star-import $(.PROJECT) tests)
.BLACK = $(shell black $(.PROJECT) tests)
.ISORT = $(shell isort $(.PROJECT) tests)
.REFACTOR = $(foreach command,.AUTOFLAKE .UNIMPORT .BLACK .ISORT,$(call $(command)))

.READD = $(shell git update-index --again)
.CHECK = $(shell pre-commit run)

.PHONY: init
init:
	pip install -U pip wheel
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

.PHONY: coverage
coverage:
	@coverage run --source=$(.PROJECT)/ -m pytest tests/
	@coverage report -m

.PHONY: test
test:
	@pytest --doctest-modules $(.PROJECT) tests

.PHONY: tox
tox:
	@tox

.PHONY: check
check:
	@$(call $(.CHECK))

.PHONY: typecheck
typecheck:
	@mypy $(.PROJECT)

.PHONY: refactor
refactor:
	@$(call $(.REFACTOR))

.PHONY: autofix
autofix:
	@-$(call $(.REFACTOR))
	@$(call $(.READD))
	@-$(call $(.CHECK))
	@$(call $(.READD))

.PHONY: commit
commit: autofix
	@cz commit

.PHONY: release
release:
# for v1.0.0 and after, the following line should be used to bump the project version:
# 	cz bump
# before v1, use the following command, which maps the following bumps:
# 	MAJOR -> MINOR (v0.2.3 -> v0.3.0)
# 	MINOR or PATCH -> PATCH (v0.2.3 -> v0.2.4)
# effectively avoiding incrementing the MAJOR version number while the first
# stable version (v1.0.0) is not released
	cz bump --dry-run --increment $(shell cz bump --dry-run | grep -q "MAJOR" && echo "MINOR" || echo "PATCH")
	git push
	git push --tags
