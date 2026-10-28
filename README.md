<div align="center">

<img src="https://raw.githubusercontent.com/Lorenzo-Gardini/evidence/main/assets/vestigia-banner.svg" alt="Vestigia - Trace answers back to evidence. An illustrated magnifying glass connects source passages to an answer." width="100%">

Classify how retrieved text supports, partially supports, contradicts, or does
not contribute to an answer.

<p>
  <a href="https://github.com/Lorenzo-Gardini/evidence/actions/workflows/tests.yml">
    <img src="https://github.com/Lorenzo-Gardini/evidence/actions/workflows/tests.yml/badge.svg?branch=main" alt="Tests">
  </a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12 or newer">
</p>

<p>
  <a href="#features">Features</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#classification-statuses">Statuses</a> ·
  <a href="#development">Development</a> ·
  <a href="#releasing-to-pypi">Releases</a>
</p>

</div>

## Features

- **Two classification approaches:** structured chat-model output and a
  TypeSafe system-one classifier.
- **Traceable results:** each result retains the original chunk and its stable
  identifier.
- **Typed data:** Pydantic models with frozen attributes validate inputs and outputs.
- **Packaged prompts:** Markdown and YAML prompt templates are available through
  Python APIs.
- **Optional integrations:** install only the provider dependencies you need.

## Quick start

Requires Python 3.12 or newer. Install the OpenAI integration with your package
manager:

```bash
# pip (inside your virtual environment)
python -m pip install "vestigia[openai]"

# uv (in a uv-managed project)
uv add "vestigia[openai]"

# Poetry (in a Poetry-managed project)
poetry add "vestigia[openai]"
```

Set `OPENAI_API_KEY` in your environment. Save the following example as
`classify.py` and run it with `python classify.py` (or `uv run classify.py`,
`poetry run python classify.py`).
This example sends data to the configured provider and may incur API charges.

```python
import asyncio

from langchain_openai import ChatOpenAI

from vestigia.chunk_classification import Chunk
from vestigia.chunk_classification.langchain import ChatModelChunkClassifier

classifier = ChatModelChunkClassifier(
    ChatOpenAI(model="gpt-4o-mini"),
)


async def classify_answer():
    result = await classifier.classify(
        query="What is the repayment term?",
        answer="The plan lasts 60 months.",
        chunks=[
            Chunk(
                id="repayment-term",
                text="The repayment plan lasts 60 months.",
            ),
            Chunk(
                id="early-repayment",
                text="Early repayment has no additional fee.",
            ),
        ],
    )

    for item in result.classifications:
        print(item.chunk.id, item.status.value, item.explanation)


if __name__ == "__main__":
    asyncio.run(classify_answer())
```

`ChatModelChunkClassifier` accepts any LangChain `BaseChatModel` that supports
structured output; the OpenAI integration above is one option.

### Use a TypeSafe system-one model

Install the extra with `python -m pip install "vestigia[typesafe]"`
(`uv add` or `poetry add` work the same way) and set `TYPESAFE_API_KEY`.
Save this example as `classify.py` and run it with
`python classify.py`:

```python
import asyncio

from langchain_typesafe import TypeSafeClassifier

from vestigia.chunk_classification import Chunk
from vestigia.chunk_classification.langchain import SystemOneModelChunkClassifier


async def classify_answer():
    classifier = SystemOneModelChunkClassifier(TypeSafeClassifier())
    result = await classifier.classify(
        query="What is the repayment term?",
        answer="The plan lasts 60 months.",
        chunks=[Chunk(id="source-1", text="The plan lasts 60 months.")],
    )
    for item in result.classifications:
        print(item.chunk.id, item.status.value, item.probabilities, item.confidence)


if __name__ == "__main__":
    asyncio.run(classify_answer())
```

Like the chat-model example, this makes a real provider request. System-one
results include per-status probabilities and confidence.

### Use the data models directly

The package exports its domain models from `vestigia.chunk_classification`:

```python
from vestigia.chunk_classification import (
    ChatModelChunkClassification,
    ChatModelChunksClassificationResult,
    Chunk,
    ChunkStatus,
)

chunk = Chunk(id="source-1", text="The plan lasts 60 months.")
result = ChatModelChunksClassificationResult(
    classifications=(
        ChatModelChunkClassification(
            chunk=chunk,
            status=ChunkStatus.USED,
            explanation="The answer states the same repayment term.",
        ),
    )
)
```

## Classification statuses

| Status | Meaning |
| --- | --- |
| `used` | The answer states or faithfully paraphrases meaningful information from the chunk. |
| `partially_used` | The answer uses meaningful information but omits or incompletely represents other significant details. |
| `contradicted` | The answer makes a claim incompatible with information in the chunk. |
| `not_used` | No meaningful information from the chunk appears in the answer, and it is not contradicted. |
| `uncertain` | The available text does not support a reliable classification. |

Omission alone is not a contradiction. Both packaged prompts instruct the model
to assess each chunk against the answer and avoid relying on external knowledge.
The labels describe observable overlap and conflict, not proof of the generating
model's internal reasoning or actual use of a retrieved chunk.

## Classifiers and prompts

| API | Purpose | Optional dependency |
| --- | --- | --- |
| `ChatModelChunkClassifier` | Classify chunks using a LangChain chat model with structured output. | `openai`, `anthropic`, or another installed LangChain chat integration |
| `SystemOneModelChunkClassifier` | Classify chunks using a TypeSafe system-one model and per-status criteria. | `typesafe` |

The adapters are exported from `vestigia.chunk_classification.langchain`.
`ChunkClassifier` is exported from `vestigia.chunk_classification.classifier`.

Load the packaged prompt templates directly when needed:

```python
from vestigia.chunk_classification.prompts import (
    load_chat_model_prompt,
    load_system_one_prompt,
)

chat_prompt_text = load_chat_model_prompt()
system_one_template = load_system_one_prompt()
```

`load_system_one_prompt()` requires the `typesafe` extra, which includes
PyYAML.

### Custom prompts

Pass a `langchain_core.prompts.PromptTemplate` as `prompt=` to the chat
classifier. Its available variables are `query`, `answer`, and `chunks`; the
last contains XML-escaped source text and identifiers. The model returns only
each chunk's `chunk_id`, `status`, and optional `explanation`; results reference
the original `Chunk` objects.

For the system-one classifier, pass a `SystemOneModelTemplate`:

```python
from vestigia.chunk_classification import ChunkStatus, SystemOneModelTemplate

template = SystemOneModelTemplate(
    state="Question: {query}\nAnswer: {answer}",
    instruction="Evaluate the following evidence: {chunk}",
    criteria={
        ChunkStatus.USED: "Meaningful information appears in the answer.",
        ChunkStatus.NOT_USED: "No meaningful information appears in the answer.",
    },
)
```

The state template receives `query` and `answer`; the instruction template
receives each chunk's text as `chunk`. Criteria require at least two statuses.

## Input and output contracts

- Chunk IDs must be unique within each call. Duplicates raise `ValueError`
  before a provider is invoked.
- An empty chunk sequence returns `classifications=()` without a provider call.
- Results follow input order and contain the original `Chunk` objects. Missing,
  extra, or duplicate response IDs raise `ValueError`.
- System-one answers must be choice answers whose labels belong to the prompt's
  criteria. Confidence and probability values must be finite and in `[0, 1]`.
  The package does not normalize the distribution or require an exact sum of 1.
- Backend, prompt-formatting, and Pydantic validation errors propagate to the
  caller; the package does not substitute partial or empty success responses.
- Model attributes are frozen and extra fields are rejected. Freezing is
  shallow: nested dictionaries remain mutable, and normal Pydantic type
  coercion applies.

## Installation options

The PyPI distribution and Python import package are both named `vestigia`.
The GitHub repository is still named `evidence`; the former Python import
`evidence` has been replaced without a compatibility alias.

Install the core models without provider integrations:

```bash
python -m pip install vestigia
```

Install optional integrations with `python -m pip install "vestigia[<extra>]"`,
`uv add "vestigia[<extra>]"`, or `poetry add "vestigia[<extra>]"`:

| Extra | Includes |
| --- | --- |
| `openai` | LangChain OpenAI integration |
| `typesafe` | TypeSafe classifier and YAML prompt support |
| `anthropic` | LangChain Anthropic integration |
| `all` | All optional integrations |

`ChatModelChunkClassifier` accepts any LangChain `BaseChatModel`, so the
`openai` and `anthropic` extras are conveniences rather than requirements.

## Development

Install [uv](https://docs.astral.sh/uv/), clone the repository, and install the
test and classifier dependencies:

```bash
git clone https://github.com/Lorenzo-Gardini/evidence.git
cd evidence
uv sync --locked --group dev --all-extras
```

Run the test suite:

```bash
uv run --locked --group dev --all-extras pytest
```

The tests use real LangChain runnables and TypeSafe response models with mocked
providers. They require neither API keys nor network access to a model.
Documentation examples are tested without making real provider requests.
GitHub Actions runs on pushes and pull requests, checking the base installation,
each classifier extra separately, and the combined configuration on Python 3.12.
It also builds distributions, validates their metadata and README, and checks
imports and packaged resources from an isolated wheel installation.

For architecture and contribution guidance, see [AGENTS.md](AGENTS.md).

## Releasing to PyPI

Publishing uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/):
no PyPI API token needs to be stored in GitHub.

### One-time setup

Create a GitHub environment named `pypi` in this repository's settings. Configure
required reviewers if you want manual approval before an upload.

In [PyPI account publishing settings](https://pypi.org/manage/account/publishing/),
add a pending GitHub publisher for the first release with these exact values:

| Setting | Value |
| --- | --- |
| PyPI project name | `vestigia` |
| Owner | `Lorenzo-Gardini` |
| Repository | `evidence` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

For an existing PyPI project, add the publisher in that project's publishing
settings instead. A pending publisher does not reserve the project name.

### Publish a version

1. Update `project.version` in [pyproject.toml](pyproject.toml), run `uv lock`,
   and commit the version and lockfile together. The initial version is `0.1.0`.
2. Verify the tests and packaging locally:

   ```bash
   uv run --locked --group dev --all-extras pytest
   uv build --no-sources
   uvx --from twine twine check --strict dist/*
   ```

   Use a clean `dist/` directory so old artifacts are not included.
3. Merge the changes into the default branch and push them before releasing.
4. Publish a GitHub Release targeting that commit, with a tag exactly matching
   `v<project.version>` (for example, `v0.1.0`). Creating a tag alone or saving a
   draft release does not publish to PyPI.
5. The [publishing workflow](.github/workflows/publish.yml) checks the version,
   runs all CI profiles, builds and validates the packages, then uploads those
   same artifacts to PyPI after any environment approval.
6. Confirm the new version on [PyPI](https://pypi.org/project/vestigia/) and
   install it in a fresh virtual environment:

   ```bash
   python -m pip install "vestigia[openai]==0.1.0"
   ```

   Replace `0.1.0` with the version being verified.

PyPI releases cannot be overwritten. For a corrected artifact, bump the version
and publish a new release.

## License

Licensed under [Apache-2.0](LICENSE).
