# Project guide

## Purpose

Vestigia (`vestigia` on PyPI and in Python imports) compares retrieved text
chunks with an LLM-generated answer. The GitHub repository is `vestigia`.
Labels describe observable use, omission, or conflict, not the generating
model's hidden reasoning. It offers structured chat-model classification and a
TypeSafe system-one approach. The public usage guide and installation options
are in [README.md](README.md).

## Architecture

- `src/vestigia/chunk_classification/models.py` defines Pydantic
  models for chunks, statuses, prompt templates, and classifier results.
- `src/vestigia/chunk_classification/prompts/` contains the packaged Markdown
  and YAML prompts. `prompts.py` loads and validates them.
- `src/vestigia/chunk_classification/classifier.py` defines the generic async
  `ChunkClassifier` protocol.
- `src/vestigia/chunk_classification/langchain/` contains the chat-model and
  TypeSafe adapters. The package initializer loads each adapter lazily because
  their optional dependencies are separate.
- `tests/` contains pytest checks for package exports, model contracts, prompt
  loading, and classifier behavior.

## Domain contracts

- Require unique `Chunk.id` values per call and associate backend answers by ID.
  Return validated results in input order, built from the original `Chunk`
  objects. Reject missing, extra, or duplicate answer IDs with `ValueError`.
  The chat model's structured output carries only `chunk_id`, never chunk text.
- Empty chunk input returns an empty result without a backend invocation.
- Classification statuses are `used`, `partially_used`, `contradicted`,
  `not_used`, and `uncertain`. Omission alone is not contradiction.
- `EvidenceModel` freezes attribute assignment and rejects extra fields. Nested
  dictionaries remain mutable; ordinary Pydantic coercion is enabled.
- Give every Pydantic field a useful `Field` description. Required fields use
  `Field(..., description=...)`; optional fields such as `explanation` keep
  `default=None`. Nullable and optional are distinct concepts in Pydantic.
- Require finite probability and confidence values in `[0, 1]`; preserve scores
  without normalization or an exact-sum constraint.
- Keep packaged prompt resources addressed through `importlib.resources` so
  they work from built distributions.

## Development

- Python support starts at 3.12; dependencies and optional extras are declared
  in `pyproject.toml` and locked in `uv.lock`.
- Add tests for behavior changes. The test suite runs without provider API
  credentials, using LangChain runnables and real TypeSafe response types with
  mocked backend calls. Keep provider exceptions visible to callers.
- Run the full suite locally (CI's `all` profile) with:

  ```bash
  uv sync --locked --group dev --all-extras
  uv run --locked --group dev --all-extras pytest
  ```

- GitHub Actions runs this suite on pushes and pull requests using
  `.github/workflows/tests.yml`, including isolated optional-dependency profiles.
- Add practical examples to Python docstrings when documenting public
  interfaces. Executable docstrings and README Python snippets are checked by
  `tests/test_documentation.py`; external provider calls are mocked there.
- Read [pyproject.toml](pyproject.toml) for dependency declarations and
  [LICENSE](LICENSE) for the Apache-2.0 terms.

## Commit messages

Every commit subject must be `<type>: <imperative summary>` with one of these
types:

| Type | Use for | Example |
| --- | --- | --- |
| `feat` | A new feature | `feat: add user login via OAuth` |
| `fix` | A bug fix | `fix: correct null pointer exception in user service` |
| `chore` | Build process or auxiliary tasks, no production code | `chore: update CI configuration` |
| `docs` | Documentation only | `docs: update README with installation instructions` |
| `style` | Formatting with no functional change | `style: format code with prettier` |
| `refactor` | Code change that neither fixes a bug nor adds a feature | `refactor: simplify authentication middleware` |
| `perf` | Performance improvement | `perf: improve query speed by adding index` |
| `test` | Adding or fixing tests | `test: add unit tests for payment service` |
| `build` | Build system or dependencies | `build: update webpack to v5` |
| `ci` | CI configuration files and scripts | `ci: add GitHub Actions workflow for tests` |
| `revert` | Reverting a previous commit | `revert: revert "feat: add user profile page"` |
| `release` | Releasing a new version | `release: v1.0.1` |

## Packaging and releases

- For installation or release changes, follow
  [Releasing to PyPI](README.md#releasing-to-pypi), which owns the publishing
  configuration and release checklist.
- Keep distribution and import names aligned as `vestigia`, including the
  self-referencing `all` extra and `importlib.resources` package names.
- Keep version metadata in `pyproject.toml` and update `uv.lock` with `uv lock`.
  Build wheel and sdist with `uv build --no-sources`; validate them using
  `uvx --from twine twine check --strict dist/*` from a clean output directory.
- `.github/workflows/tests.yml` also validates an isolated wheel installation.
  `.github/workflows/publish.yml` publishes those tested artifacts only for a
  published GitHub Release whose tag matches the package version.
- Publishing requires an explicitly requested release; preparing packaging or
  pushing code alone is not authorization to create a release or upload packages.
