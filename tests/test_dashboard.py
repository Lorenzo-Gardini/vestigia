"""Dashboard experiment logic and a Streamlit smoke test, without provider calls."""

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("yaml")
pytest.importorskip("langchain_core")

import yaml  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from vestigia.chunk_classification import (  # noqa: E402
    ChatModelChunkClassification,
    ChatModelChunksClassificationResult,
    Chunk,
    ChunkStatus,
    SystemOneModelChunkClassification,
    SystemOneModelChunksClassificationResult,
    SystemOneModelTemplate,
)
from vestigia.chunk_classification.dashboard import experiment as ex  # noqa: E402

CHAT_TEMPLATE = "Q: {query}\nA: {answer}\n{chunks}"
SYSTEM_ONE = SystemOneModelTemplate(
    state="Q: {query}\nA: {answer}",
    instruction="Chunk: {chunk}",
    criteria={ChunkStatus.USED: "Used.", ChunkStatus.NOT_USED: "Not used."},
)


def _experiment(**changes: object) -> ex.Experiment:
    values: dict[str, object] = {
        "model": ex.ModelConfig(
            provider="openai",
            model="qwen/qwen3.5-9b",
            base_url="http://localhost:1234/v1",
            temperature=0.0,
        ),
        "prompt": ex.ChatPrompt(template=CHAT_TEMPLATE),
        "input": "What is the term?",
        "output": "The plan lasts 60 months.",
        "chunks": {"term": "The plan lasts 60 months.\n\nNo fees.", "2": "Unrelated."},
    }
    return ex.Experiment(**(values | changes))


class FakeClassifier:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def classify(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.result


def test_blank_ids_are_numbered_by_row_position() -> None:
    chunks = ex.normalize_chunks([("a", "x"), ("", "y"), (None, None), (None, "z")])

    assert [(c.id, c.text) for c in chunks] == [("a", "x"), ("2", "y"), ("3", "z")]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([("2", "x"), ("", "y")], "Duplicate chunk ID '2'"),
        ([("a", "x"), ("a", "y")], "Duplicate chunk ID 'a'"),
        ([("a", "  ")], "has no text"),
    ],
)
def test_invalid_chunk_rows_are_rejected(rows: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ex.normalize_chunks(rows)


@pytest.mark.parametrize(
    "content",
    [
        "a: first\n2: second\n",
        '[{"id": "a", "text": "first"}, {"text": "second"}]',
        "- id: a\n  text: first\n- second\n",
    ],
)
def test_chunk_files_accept_documented_shapes(content: str) -> None:
    assert ex.parse_chunk_file(content) == [
        Chunk(id="a", text="first"),
        Chunk(id="2", text="second"),
    ]


@pytest.mark.parametrize(
    "content", ["just text", "[]", "- {id: a}", "- [nested]", "a: 3", "- {id: a, text: x, extra: y}"]
)
def test_chunk_files_reject_other_shapes(content: str) -> None:
    with pytest.raises(ValueError):
        ex.parse_chunk_file(content)


@pytest.mark.parametrize(
    "prompt",
    [ex.ChatPrompt(template=CHAT_TEMPLATE), SYSTEM_ONE],
)
def test_yaml_round_trip_preserves_experiment(prompt: object) -> None:
    provider = "openai" if isinstance(prompt, ex.ChatPrompt) else "typesafe"
    experiment = _experiment(
        model=ex.ModelConfig(provider=provider, model="m", base_url="http://x"),
        prompt=prompt,
        evaluations={
            "term": ex.Evaluation(
                status="used",
                confidence=0.9,
                probabilities={"used": 0.9, "not_used": 0.1},
            ),
            "2": ex.Evaluation(status="not_used", explanation="Not mentioned."),
        },
    )

    exported = ex.to_yaml(experiment)

    assert ex.from_yaml(exported) == experiment
    assert "api_key" not in exported
    assert "template: |" in exported or "state: |" in exported
    assert list(ex.from_yaml(exported).chunks) == ["term", "2"]


def test_import_accepts_unquoted_numeric_ids() -> None:
    exported = ex.to_yaml(_experiment()).replace("'2':", "2:")

    assert list(ex.from_yaml(exported).chunks) == ["term", "2"]


def test_readme_example_experiment_imports_with_results() -> None:
    path = Path(__file__).parents[1] / "examples" / "dashboard-experiment.yaml"

    experiment = ex.from_yaml(path.read_text(encoding="utf-8"))

    assert experiment.evaluations.keys() == experiment.chunks.keys()
    assert {evaluation.status for evaluation in experiment.evaluations.values()} == set(
        ChunkStatus
    )


@pytest.mark.parametrize(
    "content",
    [
        "version: 2",
        "not: an experiment",
        ex.to_yaml(_experiment()).replace("provider: openai", "provider: typesafe"),
        ex.to_yaml(_experiment()).replace("provider: openai", "provider: other"),
        yaml.safe_dump(
            _experiment().model_dump(mode="json")
            | {"evaluations": {"term": {"status": "used"}}}
        ),
        ex.to_yaml(
            _experiment(
                evaluations={
                    "term": ex.Evaluation(status="used"),
                    "2": ex.Evaluation(status="used"),
                }
            )
        ).replace("status: used", "status: maybe", 1),
    ],
)
def test_invalid_imports_are_rejected(content: str) -> None:
    with pytest.raises(ValidationError):
        ex.from_yaml(content)


def test_temperature_is_rejected_for_system_one() -> None:
    with pytest.raises(ValidationError, match="Temperature"):
        ex.ModelConfig(provider="typesafe", model="m", temperature=0.5)


@pytest.mark.parametrize(
    ("prompt", "message"),
    [
        (ex.ChatPrompt(template="{query} {answer}"), "missing {chunks}"),
        (ex.ChatPrompt(template="{query} {answer} {chunks} {extra}"), "unknown"),
        (SYSTEM_ONE.model_copy(update={"instruction": "{text}"}), "instruction"),
        (SYSTEM_ONE.model_copy(update={"state": "{chunk}"}), "state"),
    ],
)
def test_prompts_with_unfillable_placeholders_are_rejected(
    prompt: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ex.validate_prompt(prompt)


def test_evaluate_stores_chat_results(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _experiment()
    fake = FakeClassifier(
        ChatModelChunksClassificationResult(
            classifications=tuple(
                ChatModelChunkClassification(
                    chunk=Chunk(id=key, text=text),
                    status=status,
                    explanation="Because.",
                )
                for (key, text), status in zip(
                    experiment.chunks.items(), [ChunkStatus.USED, ChunkStatus.NOT_USED]
                )
            )
        )
    )
    built: list[tuple[ex.ModelConfig, str | None]] = []
    monkeypatch.setattr(
        ex, "build_classifier", lambda config, key: built.append((config, key)) or fake
    )

    evaluated = asyncio.run(ex.evaluate(experiment, api_key="lm-studio"))

    assert built == [(experiment.model, "lm-studio")]
    assert fake.calls[0]["prompt"].template == CHAT_TEMPLATE
    assert [c.id for c in fake.calls[0]["chunks"]] == ["term", "2"]
    assert evaluated.evaluations == {
        "term": ex.Evaluation(status="used", explanation="Because."),
        "2": ex.Evaluation(status="not_used", explanation="Because."),
    }


def test_evaluate_stores_system_one_results(monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = _experiment(
        model=ex.ModelConfig(provider="typesafe", model="m"),
        prompt=SYSTEM_ONE,
        chunks={"1": "Evidence"},
    )
    fake = FakeClassifier(
        SystemOneModelChunksClassificationResult(
            classifications=(
                SystemOneModelChunkClassification(
                    chunk=Chunk(id="1", text="Evidence"),
                    status=ChunkStatus.USED,
                    probabilities={ChunkStatus.USED: 0.8, ChunkStatus.NOT_USED: 0.2},
                    confidence=0.8,
                ),
            )
        )
    )
    monkeypatch.setattr(ex, "build_classifier", lambda config, key: fake)

    evaluated = asyncio.run(ex.evaluate(experiment, api_key=None))

    assert fake.calls[0]["prompt"] == SYSTEM_ONE
    assert evaluated.evaluations["1"] == ex.Evaluation(
        status="used", confidence=0.8, probabilities={"used": 0.8, "not_used": 0.2}
    )


def test_build_classifier_configures_local_openai_model() -> None:
    pytest.importorskip("langchain_openai")

    classifier = ex.build_classifier(
        ex.ModelConfig(
            provider="openai",
            model="qwen/qwen3.5-9b",
            base_url="http://localhost:1234/v1",
            temperature=0.2,
        ),
        api_key="lm-studio",
    )

    model = classifier._model
    assert model.model_name == "qwen/qwen3.5-9b"
    assert model.openai_api_base == "http://localhost:1234/v1"
    assert model.temperature == 0.2
    assert model.openai_api_key.get_secret_value() == "lm-studio"


def test_dashboard_keeps_prompt_edits_across_provider_switches() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(ex.__file__).with_name("app.py")), default_timeout=30)
    app.run()
    app.text_area(key="chat_template").input(CHAT_TEMPLATE).run()
    app.number_input(key="temperature").set_value(0.3).run()
    app.selectbox(key="provider").select("typesafe").run()
    app.text_area(key="state").input("Q: {query}").run()
    app.selectbox(key="provider").select("openai").run()

    assert app.text_area(key="chat_template").value == CHAT_TEMPLATE
    assert app.number_input(key="temperature").value == 0.3
    app.selectbox(key="provider").select("typesafe").run()
    assert app.text_area(key="state").value == "Q: {query}"


def test_chunk_ids_sort_naturally() -> None:
    chunk_ids = ["chunk-10", "chunk-1a", "chunk-2", "chunk-1"]

    assert sorted(chunk_ids, key=ex.natural_sort_key) == [
        "chunk-1",
        "chunk-1a",
        "chunk-2",
        "chunk-10",
    ]


def test_dashboard_sends_edited_system_one_criteria(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    fake = FakeClassifier(
        SystemOneModelChunksClassificationResult(
            classifications=(
                SystemOneModelChunkClassification(
                    chunk=Chunk(id="1", text="Evidence"),
                    status=ChunkStatus.USED,
                    probabilities={ChunkStatus.USED: 0.7, ChunkStatus.NOT_USED: 0.3},
                    confidence=0.7,
                ),
            )
        )
    )
    monkeypatch.setattr(ex, "build_classifier", lambda config, key: fake)
    app = AppTest.from_file(str(Path(ex.__file__).with_name("app.py")), default_timeout=30)
    app.run()
    app.text_area(key="chunk_text_0").input("Evidence")
    app.selectbox(key="provider").select("typesafe").run()
    app.text_input(key="model").input("jev-latest")
    for status in ("partially_used", "contradicted", "uncertain"):
        app.checkbox(key=f"criterion_on_{status}").uncheck()
    app.text_area(key="criterion_used").input("Clearly used.")
    app.button(key="evaluate").click().run()

    assert not app.exception
    assert fake.calls[0]["prompt"].criteria == {
        ChunkStatus.USED: "Clearly used.",
        ChunkStatus.NOT_USED: ex.default_prompt("typesafe")
        .criteria[ChunkStatus.NOT_USED]
        .strip(),
    }


def test_dashboard_evaluates_and_renders_results(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    fake = FakeClassifier(
        ChatModelChunksClassificationResult(
            classifications=(
                ChatModelChunkClassification(
                    chunk=Chunk(id="1", text="The plan lasts 60 months."),
                    status=ChunkStatus.CONTRADICTED,
                    explanation="The answer says 48 months.",
                ),
            )
        )
    )
    monkeypatch.setattr(ex, "build_classifier", lambda config, key: fake)
    app = AppTest.from_file(str(Path(ex.__file__).with_name("app.py")), default_timeout=30)
    app.run()
    app.text_area(key="chunk_text_0").input("The plan lasts 60 months.")
    app.text_input(key="model").input("gpt-4o-mini")
    app.text_area(key="input").input("What is the term?")
    app.text_area(key="output").input("The plan lasts 48 months.")
    app.button(key="evaluate").click().run()

    assert not app.exception
    assert not app.error
    assert fake.calls[0]["answer"] == "The plan lasts 48 months."
    assert app.session_state["tab"] == "Results"
    result = app.session_state["result"]
    assert result.evaluations["1"].status is ChunkStatus.CONTRADICTED
    assert "api_key" not in ex.to_yaml(result)
