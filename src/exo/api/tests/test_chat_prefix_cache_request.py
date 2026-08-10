"""Wire-contract tests for the explicit OpenAI chat prefix-cache opt-in."""

import pytest
from fastapi.responses import StreamingResponse

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.main import API
from exo.api.types import (
    BenchChatCompletionRequest,
    ChatCompletionMessage,
    ChatCompletionRequest,
)
from exo.shared.types.commands import TextGeneration as TextGenerationCommand
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams


def _request(*, use_prefix_cache: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=ModelId("test-model"),
        messages=[ChatCompletionMessage(role="user", content="hello")],
        use_prefix_cache=use_prefix_cache,
    )


class TestChatPrefixCacheRequest:
    def test_openai_wire_extension_parses_only_when_explicit(self) -> None:
        request = ChatCompletionRequest.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "use_prefix_cache": True,
            }
        )

        assert request.use_prefix_cache is True

    async def test_default_keeps_public_chat_requests_cold(self) -> None:
        request = _request()

        task = await chat_request_to_text_generation(request)

        assert request.use_prefix_cache is False
        assert task.use_prefix_cache is False

    async def test_explicit_opt_in_reaches_internal_task_contract(self) -> None:
        request = _request(use_prefix_cache=True)

        task = await chat_request_to_text_generation(request)

        assert request.use_prefix_cache is True
        assert task.use_prefix_cache is True

    @pytest.mark.parametrize("use_prefix_cache", [False, True])
    async def test_public_chat_endpoint_forwards_the_cache_policy(
        self,
        use_prefix_cache: bool,
    ) -> None:
        api = object.__new__(API)
        captured: list[TextGenerationTaskParams] = []

        async def validate_model(model_id: ModelId) -> ModelId:
            return model_id

        async def send_task(
            task_params: TextGenerationTaskParams,
        ) -> TextGenerationCommand:
            captured.append(task_params)
            return TextGenerationCommand(
                command_id=CommandId("prefix-cache-test"),
                task_params=task_params,
            )

        api._validate_model_has_instance = validate_model  # pyright: ignore[reportPrivateUsage]
        api._send_text_generation_with_images = send_task  # pyright: ignore[reportPrivateUsage]

        response = await api.chat_completions(
            _request(use_prefix_cache=use_prefix_cache).model_copy(
                update={"stream": True}
            )
        )

        assert isinstance(response, StreamingResponse)
        assert len(captured) == 1
        assert captured[0].use_prefix_cache is use_prefix_cache

    def test_benchmark_request_keeps_the_same_inherited_wire_field(self) -> None:
        request = BenchChatCompletionRequest(
            model=ModelId("test-model"),
            messages=[ChatCompletionMessage(role="user", content="hello")],
            use_prefix_cache=True,
        )

        assert request.use_prefix_cache is True
        assert request.model_dump()["use_prefix_cache"] is True
