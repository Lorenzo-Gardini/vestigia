"""Load packaged prompts for the chunk classification models.

Examples
--------
>>> chat_prompt = load_chat_model_prompt()
>>> isinstance(chat_prompt, str) and bool(chat_prompt)
True
"""

from importlib.resources import files

from vestigia.chunk_classification import SystemOneModelTemplate


def load_chat_model_prompt() -> str:
    """Read the default prompt template for the chat model classifier.

    Returns
    -------
    str
        Template text with ``query``, ``answer``, and ``chunks`` placeholders.

    Examples
    --------
    >>> prompt = load_chat_model_prompt()
    >>> isinstance(prompt, str)
    True
    """
    return (
        files("vestigia.chunk_classification")
        .joinpath("prompts", "chat_model_prompt.md")
        .read_text(encoding="utf-8")
    )


def load_system_one_prompt() -> SystemOneModelTemplate:
    """Load and validate the default system-one model prompt.

    Requires PyYAML, included in the optional ``typesafe`` extra.

    Returns
    -------
    SystemOneModelTemplate
        Validated state, per-chunk instruction, and classification criteria.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    pydantic.ValidationError
        If the packaged YAML does not match the template model.

    Examples
    --------
    >>> prompt = load_system_one_prompt()
    >>> len(prompt.criteria) >= 2
    True
    """
    import yaml

    configs = yaml.safe_load(
        files("vestigia.chunk_classification")
        .joinpath("prompts", "system_one_model_prompt.yaml")
        .read_text(encoding="utf-8")
    )
    return SystemOneModelTemplate.model_validate(configs)
