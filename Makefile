.PROJECT = boiling_learning
.TESTS_FOLDER = tests

.PHONY: test
test:
	@bash scripts/test.sh -vv $(.PROJECT) $(.TESTS_FOLDER)

.PHONY: test-cov
test-cov:
	@bash scripts/test-cov.sh -vv $(.PROJECT) $(.TESTS_FOLDER)

.PHONY: format
format:
	@bash scripts/format.sh $(.PROJECT) $(.TESTS_FOLDER)

.PHONY: lint
lint:
	@bash scripts/lint.sh $(.PROJECT) $(.TESTS_FOLDER)

.PHONY: typecheck
typecheck:
	@bash scripts/typecheck.sh $(.PROJECT)

.PHONY: vulture
vulture:
	@bash scripts/vulture.sh $(.PROJECT)

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
