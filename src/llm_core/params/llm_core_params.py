"""LlmCore project params.

Parameters are actual value of the config.

The class is a singleton, so it can be accessed from anywhere in the code.

There is a parameter regarding the environment type (stage and location), which
is used to load different paths and other parameters based on the environment.
"""

from loguru import logger as lg

from llm_core.metaclasses.singleton import Singleton
from llm_core.params.env_type import EnvType
from llm_core.params.llm_core_paths import LlmCorePaths
from llm_core.params.sample_params import SampleParams


class LlmCoreParams(metaclass=Singleton):
    """LlmCore project parameters."""

    def __init__(self) -> None:
        """Load the LlmCore params."""
        lg.info("Loading LlmCore params")
        self.set_env_type()

    def set_env_type(self, env_type: EnvType | None = None) -> None:
        """Set the environment type.

        Args:
            env_type (EnvType | None): The environment type.
                If None, it will be set from the environment variables.
                Defaults to None.
        """
        if env_type is not None:
            self.env_type = env_type
        else:
            self.env_type = EnvType.from_env_var()
        self.load_config()

    def load_config(self) -> None:
        """Load the llm_core configuration."""
        self.paths = LlmCorePaths(env_type=self.env_type)
        self.sample = SampleParams()

    def __str__(self) -> str:
        """Return the string representation of the object."""
        s = "LlmCoreParams:"
        s += f"\n{self.paths}"
        s += f"\n{self.sample}"
        return s

    def __repr__(self) -> str:
        """Return the string representation of the object."""
        return str(self)


def get_llm_core_params() -> LlmCoreParams:
    """Get the llm_core params."""
    return LlmCoreParams()


def get_llm_core_paths() -> LlmCorePaths:
    """Get the llm_core paths."""
    return get_llm_core_params().paths
