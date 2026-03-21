"""Exceptions for the chains module."""


class MissingPromptVariablesError(ValueError):
    """Input model has fields not present in the prompt template.

    Raised during ``StructuredLLMChain`` construction when the input model
    declares fields that have no corresponding ``{{ variable }}`` in the
    Jinja2 prompt template.

    Args:
        missing: Field names present in the input model but absent from the
            prompt template.

    Example:
        ::

            raise MissingPromptVariablesError({"recipe_text", "author"})
    """

    def __init__(self, missing: set[str] | frozenset[str]) -> None:
        """Initialise with the set of missing field names."""
        super().__init__(
            f"Input model fields missing from prompt template: {sorted(missing)}"
        )


class ExtraPromptVariablesError(ValueError):
    """Prompt template has variables not present in the input model.

    Raised during ``StructuredLLMChain`` construction when the Jinja2 prompt
    template references variables that do not correspond to any field on the
    input model.

    Args:
        extra: Variable names present in the prompt template but absent from
            the input model.

    Example:
        ::

            raise ExtraPromptVariablesError({"unknown_var"})
    """

    def __init__(self, extra: set[str] | frozenset[str]) -> None:
        """Initialise with the set of extra variable names."""
        super().__init__(
            f"Prompt template variables missing from input model: {sorted(extra)}"
        )
