SHFMT_VERSION ?= v3.13.1

SHELL_SCRIPTS := \
	examples/script.sh

# Install all lint tools (requires pip and curl).
.PHONY: install-tools
install-tools:
	curl --fail --silent --show-error --location \
	    "https://github.com/mvdan/sh/releases/download/$(SHFMT_VERSION)/shfmt_$(SHFMT_VERSION)_linux_amd64" \
	    -o /tmp/shfmt
	sudo install -m 755 /tmp/shfmt /usr/local/bin/shfmt
	pip install shellcheck-py ruff mypy types-PyYAML google-yamlfmt yamllint actionlint-py

# Run all linters.
.PHONY: lint
lint: lint-sh lint-py lint-yaml lint-actions

# Lint shell scripts.
.PHONY: lint-sh
lint-sh:
	# --indent 2 + --case-indent approximates the Google Shell Style Guide.
	# --diff exits non-zero on drift instead of rewriting in place.
	shfmt --indent 2 --case-indent --diff $(SHELL_SCRIPTS)
	# --external-sources lets shellcheck follow `# shellcheck source=...`
	# directives; --source-path=SCRIPTDIR resolves them relative to each
	# script's directory rather than the cwd shellcheck was invoked from.
	shellcheck --external-sources --source-path=SCRIPTDIR $(SHELL_SCRIPTS)

# Lint Python sources.
.PHONY: lint-py
lint-py:
	ruff check furiosa_perf
	mypy

# Lint YAML files.
.PHONY: lint-yaml
lint-yaml:
	yamlfmt -lint .github/
	yamllint --strict .github/

# Lint GitHub Actions workflows.
.PHONY: lint-actions
lint-actions:
	actionlint -color

# Run the test suite.
.PHONY: test
test:
	pytest

# Remove generated artifacts and tool caches.
.PHONY: clean
clean:
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
