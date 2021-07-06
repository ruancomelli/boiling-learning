pre-commit run && cz bump --increment $(. ./scripts/adapt_increment.sh) $@ && git push && git push --tags
