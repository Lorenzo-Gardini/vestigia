<div align="center">

<img src="https://raw.githubusercontent.com/Lorenzo-Gardini/vestigia/main/assets/vestigia-banner.svg" alt="Vestigia - Trace answers back to evidence. An illustrated magnifying glass connects source passages to an answer." width="100%">

Classify how retrieved text supports, partially supports, contradicts, or does
not contribute to an answer.

<p>
  <a href="https://pypi.org/project/vestigia/">
    <img src="https://img.shields.io/pypi/v/vestigia.svg?label=PyPI" alt="PyPI version">
  </a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12 or newer">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-Apache--2.0-brightgreen.svg" alt="Apache-2.0 license">
  </a>
</p>

<p>
  <a href="#features">Features</a> ·
  <a href="#installation-options">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#dashboard">Dashboard</a> ·
  <a href="#classification-statuses">Statuses</a> ·
  <a href="#development">Development</a>
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

## Installation options

The PyPI distribution, Python import package, and GitHub repository are all
named `vestigia`. Choose the install that matches the integrations you need:

| Configuration | pip command | Includes |
| --- | --- | --- |
| Base | `python -m pip install vestigia` | Core models and classifiers |
| OpenAI with LangChain | `python -m pip install "vestigia[openai]"` | LangChain OpenAI integration |
| TypeSafe with LangChain | `python -m pip install "vestigia[typesafe]"` | TypeSafe classifier and YAML prompt support |
| Anthropic with LangChain | `python -m pip install "vestigia[anthropic]"` | LangChain Anthropic integration |
| Dashboard | `python -m pip install "vestigia[dashboard]"` | Local Streamlit dashboard and all integrations |
| Everything | `python -m pip install "vestigia[all]"` | All optional integrations, including the dashboard |

There is no separate `langchain` extra: each optional extra installs its
corresponding LangChain integration. With uv or Poetry, replace
`python -m pip install` with `uv add` or `poetry add`.

`ChatModelChunkClassifier` accepts any LangChain `BaseChatModel`, so the
`openai` and `anthropic` extras are conveniences rather than requirements.

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

## Dashboard

A local browser dashboard runs the classifiers without writing code. It
installs every integration:

```bash
python -m pip install "vestigia[dashboard]"
vestigia-dashboard
```

From a clone, run `uv run vestigia-dashboard`. The dashboard lets you:

- choose an OpenAI-compatible, Anthropic, or TypeSafe model, with optional base
  URL and temperature, so local servers such as LM Studio
  (`http://localhost:1234/v1`) work too;
- enter the input, output, and chunks, or import chunks from YAML or JSON as an
  ID-to-text mapping, a list of `{id, text}`, or a list of strings; blank IDs
  become `1`…`N` by position;
- edit the prompt, including system-one criteria;
- evaluate and review color-coded results per chunk;
- export the experiment as YAML (model settings, prompt, inputs, chunks, and
  results) and import it later to restore everything.

To explore the dashboard without a model, choose **Import** and load
[`examples/dashboard-experiment.yaml`](https://github.com/Lorenzo-Gardini/vestigia/blob/main/examples/dashboard-experiment.yaml).
It restores the setup and illustrative results shown below:

<img src="https://raw.githubusercontent.com/Lorenzo-Gardini/vestigia/main/assets/dashboard-setup.png" alt="Dashboard Setup tab with an OpenAI-compatible qwen/qwen3.5-9b model, a question and answer about the Colosseum, and five chunks." width="100%">

<img src="https://raw.githubusercontent.com/Lorenzo-Gardini/vestigia/main/assets/dashboard-results.png" alt="Dashboard Results tab showing five Colosseum chunks classified as used, partially used, not used, uncertain, and contradicted, each with an explanation." width="100%">

### API keys

Leave the **API key** field empty to use the provider's environment variable:

| Provider | Environment variable |
| --- | --- |
| OpenAI-compatible | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| TypeSafe | `TYPESAFE_API_KEY` |

Set it in the shell that starts the dashboard:

```bash
export OPENAI_API_KEY="sk-..."
vestigia-dashboard
```

A key typed into the field overrides the variable for the current browser
session only and is never exported. Local OpenAI-compatible servers such as LM
Studio ignore the key, but the client still requires one, so any placeholder
works: `OPENAI_API_KEY=lm-studio vestigia-dashboard`.

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

## Development

Install [uv](https://docs.astral.sh/uv/), clone the repository, and install the
test and classifier dependencies:

```bash
git clone https://github.com/Lorenzo-Gardini/vestigia.git
cd vestigia
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

## License

Licensed under [Apache-2.0](LICENSE).
