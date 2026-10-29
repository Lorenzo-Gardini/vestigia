"""Streamlit interface for chunk classification; start it with ``vestigia-dashboard``."""

import asyncio
import base64
from collections import Counter
from collections.abc import Iterable
from html import escape

import streamlit as st
from pydantic import ValidationError

from vestigia.chunk_classification import ChunkStatus, SystemOneModelTemplate
from vestigia.chunk_classification.dashboard import experiment as ex

PROVIDERS = {
    "openai": "OpenAI-compatible · chat",
    "anthropic": "Anthropic · chat",
    "typesafe": "TypeSafe · system-one",
}
MODEL_PLACEHOLDERS = {
    "openai": "gpt-4o-mini or qwen/qwen3.5-9b",
    "anthropic": "claude-sonnet-4-5",
    "typesafe": "jev-latest",
}
# (text, pastel background, pastel fill); status is always shown as text, never by color alone.
STATUS_COLORS = {
    ChunkStatus.USED: ("#1E6B45", "#DDF3E6", "#8FD4AE"),
    ChunkStatus.PARTIALLY_USED: ("#7A4E0A", "#FCEBD0", "#F4C27F"),
    ChunkStatus.CONTRADICTED: ("#9B2C2C", "#FBE0DE", "#F2A39C"),
    ChunkStatus.NOT_USED: ("#4A5568", "#E8ECF1", "#B8C2CF"),
    ChunkStatus.UNCERTAIN: ("#5B3A99", "#ECE4FA", "#BBA6EC"),
}
STATUS_MEANINGS = {
    ChunkStatus.USED: "Stated or faithfully paraphrased in the output.",
    ChunkStatus.PARTIALLY_USED: "Used, but significant details are missing.",
    ChunkStatus.CONTRADICTED: "The output makes a conflicting claim.",
    ChunkStatus.NOT_USED: "Nothing from the chunk appears in the output.",
    ChunkStatus.UNCERTAIN: "The text does not support a reliable call.",
}
MARK = '<img class="v-mark" alt="" width="22" height="22" src="data:image/svg+xml;base64,{}">'.format(
    base64.b64encode(
        b'<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 22 22">'
        b'<rect x="1" y="4" width="10" height="2.5" rx="1.25" fill="#9AA5B1"/>'
        b'<rect x="1" y="9.75" width="14" height="2.5" rx="1.25" fill="#1D5FB0"/>'
        b'<rect x="1" y="15.5" width="7" height="2.5" rx="1.25" fill="#9AA5B1"/>'
        b'<circle cx="19" cy="11" r="2.25" fill="#1D5FB0"/></svg>'
    ).decode()
)
MONO = '"IBM Plex Mono", ui-monospace, monospace'
CRITERIA_KEYS = [f"criterion{part}_{s}" for s in ChunkStatus for part in ("", "_on")]

CSS = f"""
<style>
::selection {{ background: #CFE0F7; }}
.block-container {{ max-width: 1440px; padding: 1.5rem 3rem 4rem; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="InputInstructions"] {{ display: none; }}
.st-key-chat_template textarea, .st-key-state textarea, .st-key-instruction textarea {{
  font-family: {MONO}; font-size: .875rem; line-height: 1.6;
}}
[class*="st-key-chunk_id_"] input {{ font-family: {MONO}; font-size: .9rem; }}
.v-appbar {{ display: flex; align-items: center; gap: .65rem; min-height: 2.75rem; }}
.v-brand {{ font-size: 1.25rem; font-weight: 600; color: #1F2933; }}
.v-sep {{ width: 1px; height: 1.2rem; background: #D9DEE4; margin: 0 .35rem; }}
.v-model {{ font-size: .95rem; color: #3E4C59; }}
.v-model b {{ font-weight: 500; color: #1F2933; }}
.v-muted {{ color: #7B8794; }}
.v-badge {{
  display: inline-block; padding: .12rem .55rem; border-radius: 999px; white-space: nowrap;
  font-size: .8rem; font-weight: 600; font-variant-numeric: tabular-nums;
}}
.v-dist {{ display: flex; gap: 2px; height: 10px; margin: .3rem 0 .8rem; }}
.v-dist > div {{ border-radius: 2px; }}
.v-legend {{ display: flex; flex-wrap: wrap; gap: .4rem 1.25rem; font-size: .9rem;
  color: #3E4C59; margin-bottom: 1.2rem; font-variant-numeric: tabular-nums; }}
.v-legend span::before {{ content: ""; display: inline-block; width: .6rem; height: .6rem;
  border-radius: 2px; margin-right: .45rem; background: var(--c); }}
.v-legend b {{ color: #1F2933; font-weight: 600; margin-left: .3rem; }}
.v-total {{ font-size: 1rem; color: #1F2933; margin: 0; }}
.v-card {{
  border: 1px solid #E3E7EC; border-radius: 8px; background: #FFFFFF;
}}
.v-cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(26rem, 1fr));
  gap: .9rem; align-items: start; }}
.v-card > summary {{
  display: flex; justify-content: space-between; align-items: center; gap: 1rem;
  padding: .8rem 1.1rem; cursor: pointer; list-style: none;
}}
.v-card > summary::-webkit-details-marker {{ display: none; }}
.v-card > summary::before {{ content: "▸"; color: #7B8794; margin-right: .5rem; }}
.v-card[open] > summary::before {{ content: "▾"; }}
.v-card > summary:hover {{ background: #F7F8FA; }}
.v-card > summary:focus-visible {{ outline: 2px solid #1D5FB0; outline-offset: -2px; }}
.v-body {{ padding: 0 1.1rem 1rem; border-top: 1px solid #EEF0F3; }}
.v-id {{ font-family: {MONO}; font-size: .9rem; color: #1F2933; overflow-wrap: anywhere;
  flex: 1; }}
.v-text {{ color: #323F4B; white-space: pre-wrap; margin: .8rem 0 0; line-height: 1.6;
  font-size: 1rem; }}
.v-label {{ color: #616E7C; font-size: .8rem; font-weight: 600; margin: .9rem 0 .25rem; }}
.v-expl {{ color: #1F2933; margin: 0; line-height: 1.6; font-size: 1rem; }}
.v-prob {{ display: grid; grid-template-columns: 9rem 1fr 3rem; align-items: center;
  gap: .7rem; font-size: .9rem; color: #3E4C59; line-height: 1.7; }}
.v-bar {{ height: 6px; background: #EEF0F3; border-radius: 3px; overflow: hidden; }}
.v-bar > div {{ height: 100%; }}
.v-num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.v-empty {{ color: #3E4C59; font-size: 1rem; margin: 0 0 1rem; line-height: 1.6; }}
.v-key {{ display: grid; grid-template-columns: 9.5rem 1fr; gap: .6rem .8rem;
  align-items: baseline; font-size: .95rem; color: #3E4C59; }}
</style>
"""

state = st.session_state


def _chunk_keys(uid: int) -> tuple[str, str, str]:
    return f"chunk_id_{uid}", f"chunk_text_{uid}", f"chunk_open_{uid}"


def _add_chunk(chunk_id: str = "", text: str = "") -> None:
    uid = state.next_uid
    state.next_uid += 1
    id_key, text_key, open_key = _chunk_keys(uid)
    state[id_key], state[text_key], state[open_key] = chunk_id, text, True
    state.chunk_uids = [*state.chunk_uids, uid]


def _remove_chunk(uid: int) -> None:
    state.chunk_uids = [u for u in state.chunk_uids if u != uid]
    for key in _chunk_keys(uid):
        state.pop(key, None)


def _set_chunks(rows: Iterable[tuple[str, str]]) -> None:
    for uid in state.chunk_uids:
        _remove_chunk(uid)
    for chunk_id, text in rows:
        _add_chunk(chunk_id, text)


def _set_chunks_open(value: bool) -> None:
    for uid in state.chunk_uids:
        state[_chunk_keys(uid)[2]] = value


def _set_prompt(prompt: ex.ChatPrompt | SystemOneModelTemplate) -> None:
    if isinstance(prompt, ex.ChatPrompt):
        state.chat_template = prompt.template
        return
    state.state = prompt.state
    state.instruction = prompt.instruction
    for status in ChunkStatus:
        state[f"criterion_on_{status}"] = status in prompt.criteria
        state[f"criterion_{status}"] = prompt.criteria.get(status, "").strip()


def _init_state() -> None:
    for key, value in {
        "provider": "openai",
        "model": "",
        "base_url": "",
        "api_key": "",
        "temperature": None,
        "input": "",
        "output": "",
        "chunk_uids": [],
        "next_uid": 0,
        "results_open": True,
        "running": False,
        "result": None,
        "error": None,
        "import_error": None,
        "chunk_error": None,
    }.items():
        state.setdefault(key, value)
    if "chat_template" not in state:
        _set_prompt(ex.default_prompt("openai"))
    if "state" not in state:
        _set_prompt(ex.default_prompt("typesafe"))
    if state.next_uid == 0:
        _add_chunk()
    # Streamlit drops values of widgets not rendered this run; re-setting keeps them.
    chunk_keys = [key for uid in state.chunk_uids for key in _chunk_keys(uid)]
    for key in (
        "chat_template",
        "state",
        "instruction",
        "temperature",
        *CRITERIA_KEYS,
        *chunk_keys,
    ):
        state[key] = state[key]


def _message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(
            (
                f"{'.'.join(map(str, error['loc']))}: {error['msg']}"
                if error["loc"]
                else error["msg"]
            )
            for error in exc.errors()
        )
    if isinstance(exc, ValueError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def _load_experiment() -> None:
    upload = state.experiment_file
    if upload is None:
        return
    try:
        loaded = ex.from_yaml(upload.getvalue().decode("utf-8"))
    except Exception as exc:
        state.import_error = f"Could not import this file. {_message(exc)}"
        return
    state.import_error = None
    state.provider = loaded.model.provider
    state.model = loaded.model.model
    state.base_url = loaded.model.base_url or ""
    state.temperature = loaded.model.temperature
    _set_prompt(loaded.prompt)
    state.input = loaded.input
    state.output = loaded.output
    _set_chunks(loaded.chunks.items())
    state.result = loaded if loaded.evaluations else None
    state.error = None
    state.tab = "Results" if loaded.evaluations else "Setup"


def _load_chunks() -> None:
    upload = state.chunk_file
    if upload is None:
        return
    try:
        chunks = ex.parse_chunk_file(upload.getvalue().decode("utf-8"))
    except Exception as exc:
        state.chunk_error = f"Could not import chunks. {_message(exc)}"
        return
    state.chunk_error = None
    _set_chunks((chunk.id, chunk.text) for chunk in chunks)


def _reset_prompt() -> None:
    _set_prompt(ex.default_prompt(state.provider))


def _current_experiment() -> ex.Experiment:
    parsed = ex.normalize_chunks(
        (state[_chunk_keys(uid)[0]], state[_chunk_keys(uid)[1]])
        for uid in state.chunk_uids
    )
    if not parsed:
        raise ValueError("Add at least one chunk with text before evaluating")
    provider = state.provider
    if provider != "typesafe":
        prompt: ex.ChatPrompt | SystemOneModelTemplate = ex.ChatPrompt(
            template=state.chat_template
        )
    else:
        criteria = {
            status: state[f"criterion_{status}"].strip()
            for status in ChunkStatus
            if state[f"criterion_on_{status}"]
        }
        if missing := [status.value for status, text in criteria.items() if not text]:
            raise ValueError(f"Add a definition for {', '.join(missing)}")
        if len(criteria) < 2:
            raise ValueError("Enable at least two criteria")
        prompt = SystemOneModelTemplate(
            state=state.state, instruction=state.instruction, criteria=criteria
        )
    return ex.Experiment(
        model=ex.ModelConfig(
            provider=provider,
            model=state.model.strip(),
            base_url=state.base_url.strip() or None,
            temperature=state.get("temperature") if provider != "typesafe" else None,
        ),
        prompt=prompt,
        input=state.input,
        output=state.output,
        chunks={chunk.id: chunk.text for chunk in parsed},
    )


def _render_model() -> None:
    st.subheader("Model")
    is_chat = state.provider != "typesafe"
    provider, model, base_url, api_key, *temperature = st.columns(
        [2.2, 2.4, 2.6, 2.2, 1.4] if is_chat else [2.2, 2.4, 2.6, 2.2]
    )
    provider.selectbox(
        "Provider",
        list(PROVIDERS),
        format_func=PROVIDERS.__getitem__,
        key="provider",
    )
    model.text_input(
        "Model", key="model", placeholder=MODEL_PLACEHOLDERS[state.provider]
    )
    base_url.text_input(
        "Base URL",
        key="base_url",
        placeholder="Provider default",
        help="For local servers, e.g. http://localhost:1234/v1 (LM Studio).",
    )
    api_key.text_input(
        "API key",
        key="api_key",
        type="password",
        placeholder="From environment",
        help="Kept in this session only; exports never include it.",
    )
    if is_chat:
        temperature[0].number_input(
            "Temperature",
            key="temperature",
            min_value=0.0,
            max_value=2.0,
            step=0.1,
            placeholder="Default",
        )


def _appbar(result: ex.Experiment | None) -> None:
    brand, actions = st.columns([5, 2], vertical_alignment="center")
    model = state.model.strip()
    brand.html(
        f'<div class="v-appbar">{MARK}<span class="v-brand">Vestigia</span>'
        '<span class="v-sep"></span><span class="v-model">'
        + (
            f"<b>{escape(model)}</b> · {PROVIDERS[state.provider]}"
            if model
            else '<span class="v-muted">No model selected</span>'
        )
        + "</span></div>"
    )
    load, save = actions.columns(2)
    with load.popover("Import", width="stretch"):
        st.file_uploader(
            "Experiment YAML",
            type=["yaml", "yml"],
            key="experiment_file",
            on_change=_load_experiment,
            help="Restores model settings, prompt, input, output, chunks, and results.",
        )
    save.download_button(
        "Export",
        data=ex.to_yaml(result) if result else "",
        file_name="vestigia-evaluation.yaml",
        mime="application/x-yaml",
        disabled=result is None,
        width="stretch",
        help="Download settings, prompt, inputs, chunks, and results as YAML.",
    )
    if state.import_error:
        st.error(state.import_error)


def _section_header(title: str, on_expand, on_collapse, key: str):
    heading, expand, collapse, extra = st.columns(
        [8, 1.3, 1.3, 1.3], vertical_alignment="center"
    )
    heading.subheader(title)
    expand.button(
        "Expand all", key=f"{key}_expand", type="tertiary", on_click=on_expand
    )
    collapse.button(
        "Collapse all", key=f"{key}_collapse", type="tertiary", on_click=on_collapse
    )
    return extra


def _render_chunks() -> None:
    extra = _section_header(
        "Chunks",
        lambda: _set_chunks_open(True),
        lambda: _set_chunks_open(False),
        "chunks",
    )
    with extra.popover("Import", width="stretch"):
        st.file_uploader(
            "YAML or JSON",
            type=["yaml", "yml", "json"],
            key="chunk_file",
            on_change=_load_chunks,
            help="A mapping of ID to text, a list of {id, text}, or a list of strings.",
        )
    if state.chunk_error:
        st.error(state.chunk_error)
    grid = st.columns(2, gap="medium")
    ordered_chunks = sorted(
        enumerate(state.chunk_uids, start=1),
        key=lambda item: ex.natural_sort_key(
            state[_chunk_keys(item[1])[0]].strip() or str(item[0])
        ),
    )
    for position, uid in ordered_chunks:
        id_key, text_key, open_key = _chunk_keys(uid)
        chunk_id = state[id_key].strip() or str(position)
        preview = " ".join(state[text_key].split())[:60] or "Empty chunk"
        with (
            grid[(position - 1) % 2],
            st.expander(
                f"**{chunk_id}**\u2003{preview}", key=open_key, on_change="rerun"
            ),
        ):
            id_col, _, remove_col = st.columns([3, 2, 1.6], vertical_alignment="center")
            id_col.text_input(
                "Chunk ID",
                key=id_key,
                placeholder=f"ID (default {position})",
                label_visibility="collapsed",
            )
            remove_col.button(
                "Remove",
                key=f"remove_{uid}",
                type="tertiary",
                on_click=_remove_chunk,
                args=(uid,),
            )
            st.text_area(
                "Chunk text",
                key=text_key,
                height="content",
                placeholder="Chunk text",
                label_visibility="collapsed",
            )
    st.button("Add chunk", key="add_chunk", on_click=_add_chunk, icon=":material/add:")


def _render_criteria_and_prompt() -> None:
    if state.provider == "typesafe":
        _render_criteria()
    _render_prompt()


def _render_criteria() -> None:
    with st.expander("Criteria"):
        st.caption(
            "Definitions from the default template. The model chooses among the "
            "enabled statuses; enable at least two."
        )
        grid = st.columns(2, gap="medium")
        for index, status in enumerate(ChunkStatus):
            with grid[index % 2]:
                st.checkbox(status.value, key=f"criterion_on_{status}")
                st.text_area(
                    f"Definition of {status.value}",
                    key=f"criterion_{status}",
                    height="content",
                    label_visibility="collapsed",
                    disabled=not state[f"criterion_on_{status}"],
                )


def _render_prompt() -> None:
    with st.expander("Prompt"):
        if state.provider == "typesafe":
            left, right = st.columns(2, gap="medium")
            left.text_area(
                "State", key="state", height=420, help="Uses {query} and {answer}."
            )
            right.text_area(
                "Instruction", key="instruction", height=420, help="Uses {chunk}."
            )
        else:
            st.text_area(
                "Template",
                key="chat_template",
                height=360,
                help="Must use {query}, {answer}, and {chunks}.",
            )
        st.button("Reset to default", on_click=_reset_prompt)


def _badge(status: ChunkStatus, suffix: str = "") -> str:
    text, background, _ = STATUS_COLORS[status]
    label = status.value.replace("_", " ")
    return (
        f'<span class="v-badge" style="color:{text};background:{background}">'
        f"{label}{suffix}</span>"
    )


def _card(chunk_id: str, text: str, evaluation: ex.Evaluation, is_open: bool) -> str:
    parts = [
        f'<details class="v-card"{" open" if is_open else ""}><summary>'
        f'<span class="v-id">{escape(chunk_id)}</span>{_badge(evaluation.status)}</summary>'
        f'<div class="v-body"><p class="v-text">{escape(text)}</p>'
    ]
    if evaluation.explanation:
        parts.append(
            f'<p class="v-label">Explanation</p>'
            f'<p class="v-expl">{escape(evaluation.explanation)}</p>'
        )
    if evaluation.confidence is not None:
        parts.append(f'<p class="v-label">Confidence {evaluation.confidence:.2f}</p>')
    for status, probability in (evaluation.probabilities or {}).items():
        bar = STATUS_COLORS[status][2] if status == evaluation.status else "#D9DEE4"
        parts.append(
            f'<div class="v-prob"><span>{status.value.replace("_", " ")}</span>'
            f'<div class="v-bar"><div style="width:{probability * 100:.1f}%;'
            f'background:{bar}"></div></div>'
            f'<span class="v-num">{probability:.2f}</span></div>'
        )
    parts.append("</div></details>")
    return "".join(parts)


def _render_results() -> None:
    result: ex.Experiment | None = state.result
    if result is None:
        st.subheader("Results")
    else:
        _section_header(
            "Results",
            lambda: state.update(results_open=True),
            lambda: state.update(results_open=False),
            "results",
        )
    if result is None:
        st.html(
            '<p class="v-empty">No evaluation yet. Add the input, the output, and '
            "the chunks, then run Evaluate. Each chunk receives one status:</p>"
            '<div class="v-key">'
            + "".join(
                f"<span>{_badge(status)}</span><span>{meaning}</span>"
                for status, meaning in STATUS_MEANINGS.items()
            )
            + "</div>"
        )
        return
    counts = Counter(evaluation.status for evaluation in result.evaluations.values())
    total = len(result.evaluations)
    present = [status for status in ChunkStatus if counts[status]]
    st.html(
        f'<p class="v-total">{total} chunk{"s" if total != 1 else ""} evaluated</p>'
        '<div class="v-dist" role="img" aria-label="Status distribution">'
        + "".join(
            f'<div style="flex:{counts[s]};background:{STATUS_COLORS[s][2]}"></div>'
            for s in present
        )
        + '</div><div class="v-legend">'
        + "".join(
            f'<span style="--c:{STATUS_COLORS[s][2]}">'
            f'{s.value.replace("_", " ")}<b>{counts[s]}</b></span>'
            for s in present
        )
        + '</div><div class="v-cards">'
        + "".join(
            _card(chunk_id, result.chunks[chunk_id], evaluation, state.results_open)
            for chunk_id, evaluation in sorted(
                result.evaluations.items(),
                key=lambda item: ex.natural_sort_key(item[0]),
            )
        )
        + "</div>"
    )


def _start() -> None:
    state.running = True


def main() -> None:
    st.set_page_config(page_title="Vestigia", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    _init_state()
    _appbar(state.result)

    if state.pop("show_results", False):
        state.tab = "Results"
    setup, results = st.tabs(["Setup", "Results"], key="tab", on_change="rerun")
    with setup:
        _render_model()
        st.subheader("Answer")
        query, answer = st.columns(2, gap="medium")
        query.text_area(
            "Input", key="input", height=150, placeholder="The user's query"
        )
        answer.text_area(
            "Output", key="output", height=150, placeholder="The generated answer"
        )

        _render_chunks()
        _render_criteria_and_prompt()

        st.button(
            "Evaluate",
            type="primary",
            key="evaluate",
            width="stretch",
            disabled=state.running,
            on_click=_start,
        )
        if state.error:
            st.error(state.error)
    if state.running:
        state.result = state.error = None
        try:
            with setup, st.spinner("Evaluating chunks…"):
                current = _current_experiment()
                state.result = asyncio.run(ex.evaluate(current, state.api_key or None))
            state.show_results = True
        except Exception as exc:
            state.error = f"Evaluation failed. {_message(exc)}"
        state.running = False
        st.rerun()

    with results:
        _render_results()


main()
