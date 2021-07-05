bump_output=$(cz bump --dry-run)

if echo "$bump_output" | grep -q "PATCH"; then
	echo "PATCH"
elif echo "$bump_output" | grep -q "MINOR"; then
	echo "PATCH"
else
	echo "MINOR"
fi
