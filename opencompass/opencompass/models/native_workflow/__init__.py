"""Optional native tool calling for the GTA workflow inferencer."""

from opencompass.models.lagent import LagentAgent

from .native_agent import SYSTEM, native_chat


class NativeLagentAgent(LagentAgent):
    """Call the task's tools via OpenRouter Chat Completions."""

    def __init__(self, *args, native_system_prompt=SYSTEM, **kwargs):
        if not isinstance(native_system_prompt, str) or not native_system_prompt.strip():
            raise ValueError('native_system_prompt must be a nonempty string')
        super().__init__(*args, **kwargs)
        self.native_system_prompt = native_system_prompt

    chat = native_chat
