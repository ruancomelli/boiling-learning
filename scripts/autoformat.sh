isort .
autoflake \
	-i -r \
	--expand-star-imports \
	--remove-all-unused-imports \
	--ignore-init-module-imports \
	--remove-unused-variables \
	.