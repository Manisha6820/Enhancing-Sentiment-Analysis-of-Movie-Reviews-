# Contributing Guide

First — thank you for wanting to contribute! Contributions make open-source projects better and help you stand out in placements and interviews.

This document describes step-by-step contribution workflows for:
- Bug reports
- Feature requests
- Adding new models or benchmarks
- Code style and PR submission flow

## Code of conduct
Be respectful and constructive. Treat maintainers and other contributors kindly.

## Reporting a bug
1. Search existing issues to ensure it isn't already reported.
2. Open a new issue using the bug report template (.github/ISSUE_TEMPLATE/bug_report.md).
3. Provide:
   - A clear title and short summary.
   - Steps to reproduce (commands to run).
   - Environment (OS, Python version, pip freeze, CPU/GPU).
   - Attach logs, small reproducible script, or dataset subset if possible.

Example:
- Title: "Training script fails on tokenization with non-UTF8 characters"
- Body: include dataset snippet, full traceback, and command executed.

## Requesting a feature
1. Check existing feature requests.
2. Open a feature request using the feature_request template (.github/ISSUE_TEMPLATE/feature_request.md).
3. Provide the motivation, proposed API/design, and expected benefits.

## Adding a new model
If you want to add a new classifier (classical or deep):
1. Fork the repository and create a feature branch: git checkout -b feature/my-model
2. Implement model code under:
   - Classical: src/models/classical/{my_model}.py
   - Deep: src/models/deep/{my_model}.py
3. Add CLI wrapper/entry in scripts/run_classical.py or scripts/run_deep.py.
4. Add unit tests (recommended) under tests/.
5. Update docs:
   - README.md (add to comparison table)
   - docs/model_cards/{my_model}.md — include model description, hyperparameters, expected runtime.
6. Submit a pull request (see PR process below).

Model implementation checklist:
- Input signature: (train_data, val_data, config) -> returns trained_model object and metrics dict.
- Save checkpoint under results/{model_name}/checkpoints.
- Save metrics to results/{model_name}/metrics.json.

Suggested hyperparameters should be configurable via YAML or CLI flags. Keep deterministic seeds where possible.

## Code style & tests
- Follow PEP8 for Python.
- Use docstrings for public functions and modules.
- Add tests for non-trivial logic. Tests should be runnable with pytest.
- Format with black / isort if used; include a pre-commit hook if possible.

## Pull request flow
1. Fork -> Branch -> Commit -> Push
2. Open PR against main with descriptive title.
3. Link related issue(s) with GitHub keywords (e.g., "Fixes #123").
4. In PR description include:
   - Short summary of changes.
   - How to run/test (commands).
   - Any new dependencies.
   - Performance/regression notes (if applicable).
5. Maintain CI green (if CI set up).
6. Wait for maintainers to review; respond to requested changes.

## Review checklist for PRs
- Are tests added or updated?
- Is the documentation updated?
- Are there any breaking changes?
- Is the code modular and reusable?

## License & copyright
By contributing, you agree your contributions will be licensed under the project's MIT License.

## Contact / Questions
If you have questions, open a discussion or create a new issue tagged "question".

Thank you — contributions are valued and will be acknowledged in CONTRIBUTORS.md.