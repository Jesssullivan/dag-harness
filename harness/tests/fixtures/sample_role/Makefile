.PHONY: help test lint molecule converge verify clean

help:
	@echo "Available targets:"
	@echo "  test      - Run all tests"
	@echo "  lint      - Run linting"
	@echo "  molecule  - Run molecule tests"
	@echo "  converge  - Run molecule converge"
	@echo "  verify    - Run molecule verify"
	@echo "  clean     - Clean up test artifacts"

test: lint molecule

lint:
	ansible-lint .
	yamllint .

molecule:
	molecule test

converge:
	molecule converge

verify:
	molecule verify

clean:
	molecule destroy
	rm -rf .molecule
