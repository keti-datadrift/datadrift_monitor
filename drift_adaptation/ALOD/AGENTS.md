# AGENTS.md

## Scope

These instructions apply to the entire repository unless a more specific
`AGENTS.md` exists in a subdirectory.

## Working Principles

* Read the relevant source files, configuration files, and documentation before editing.
* Make the smallest complete change that satisfies the task.
* Preserve the existing architecture, naming conventions, and code style.
* Do not modify unrelated files or behavior.
* Do not overwrite or revert existing user changes.
* Resolve minor implementation details by following existing repository patterns.
* Do not hide failures with broad exception handling or silent fallback behavior.
* Do not add new dependencies unless they are necessary for the requested task.
* Explain any required dependency, public API, configuration, or file-format change.

## Repository Safety

* Treat datasets, checkpoints, logs, experiment outputs, generated files,
  external repositories, and reference implementations as read-only unless
  the task explicitly targets them.
* In particular, do not modify directories such as:

  * `datasets/`
  * `checkpoints/`
  * `runs/`
  * `results/`
  * `logs/`
  * `wandb/`
  * `outputs/`
  * `code_refs/`
  * `third_party/`
* Do not introduce hard-coded absolute paths, credentials, tokens, or private URLs.
* Preserve backward compatibility unless the task explicitly requires a breaking change.

## Python Guidelines

* Follow the style already used in the repository.
* Keep functions focused and avoid unnecessary abstractions.
* Use clear variable and function names.
* Write comments and docstrings in English.
* Comments should explain intent or non-obvious reasoning, not restate the code.
* Remove unused imports, variables, and dead code introduced by the change.
* Prefer configuration files or command-line arguments over hard-coded parameters.
* Use `pathlib` or the repository's existing path-handling convention.
* Preserve existing type annotations and add annotations when they improve clarity.

## Machine Learning Guidelines

* Preserve tensor shape, dtype, device, and gradient behavior.
* Do not add unnecessary `.detach()`, `.cpu()`, `.numpy()`, or in-place operations.
* Avoid moving tensors between CPU and GPU inside performance-critical loops.
* Keep training, validation, and inference behavior clearly separated.
* Preserve reproducibility when modifying sampling, splitting, or initialization.
* Use explicit random seeds when introducing randomized behavior.
* Do not silently change dataset splits, preprocessing, evaluation metrics,
  checkpoint formats, or default hyperparameters.
* Avoid loading an entire dataset into memory unless the existing implementation does so intentionally.
* Maintain compatibility with the repository's existing configuration and checkpoint loading behavior.

## Workflow

1. Inspect the relevant files and existing implementation.
2. Identify the minimum set of files that must change.
3. Implement the requested behavior.
4. Review the modified files for unrelated changes.
5. Run the most relevant available validation commands.
6. Report what changed and what validation was performed.

For broad refactors or changes spanning several modules, establish a brief
implementation plan before editing, then complete the work without stopping
for trivial decisions.

## Documentation Records

* Record every implementation change, refactor, bug fix, configuration or
  experiment-protocol decision, and experiment-result analysis under `docs/`.
* Create one timestamped directory per work item using
  `MM-DD-YYYY_HH;mm_<short-topic>` and place a concise Markdown record inside it.
* Each change record must state the objective, affected behavior and files,
  validation performed, and any compatibility or follow-up notes.
* Each result analysis must identify the source experiment/run paths and clearly
  distinguish measured evidence from interpretation.
* Update the same record throughout one task instead of scattering partial notes
  across unrelated documents. Preserve historical records rather than rewriting
  them to match later behavior.

## Test and Temporary Artifact Policy

* Do not add or retain project-owned test suites, `tests/` directories,
  `test_*.py` files, pytest fixtures, or test-only dependencies unless the
  user explicitly requests them.
* Do not create or run smoke, dry-run, synthetic, benchmark, or disposable
  experiment workflows unless the user explicitly requests that validation.
* Do not create validation outputs such as `work_dirs/.smoke/` inside the
  repository.
* Do not leave `.pytest_cache/`, `__pycache__/`, `.pyc`, coverage output, or
  other tool-generated validation artifacts in the repository. If a permitted
  command creates them, remove them before handoff.
* Treat runtime implementation and real user experiment outputs separately
  from validation artifacts. Never delete or overwrite a real run while
  cleaning temporary files.

## Validation

* Prefer read-only source inspection, configuration checks, Git diff checks,
  and the actual user-requested runtime path.
* Run formatters or linters only when they are already configured.
* Do not claim that a command passed unless it was actually executed.
* If validation cannot be completed, state exactly what was not run and why.

## Final Response

Provide a concise final response containing:

1. A summary of the implemented changes.
2. The validation commands that were run and their results.
3. Any remaining limitation, skipped validation, or important compatibility note.

Do not include unrelated recommendations or lengthy explanations.
