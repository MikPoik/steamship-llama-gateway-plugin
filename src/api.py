import json
import logging
from typing import Any, Dict, List, Optional, Type

import openai
from pydantic import Field
from steamship import Steamship, Block, Tag, SteamshipError
from steamship.data.tags.tag_constants import TagKind, RoleTag
from steamship.invocable import Config, InvocableResponse, InvocationContext
from steamship.plugin.generator import Generator
from steamship.plugin.inputs.raw_block_and_tag_plugin_input import (
    RawBlockAndTagPluginInput, )
from steamship.plugin.outputs.raw_block_and_tag_plugin_output import (
    RawBlockAndTagPluginOutput, )
from steamship.plugin.request import PluginRequest
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    before_sleep_log,
    wait_exponential_jitter,
)
import requests
from steamship import Block, File, Steamship, MimeTypes, Tag
from steamship.data import TagKind
from steamship.data.tags.tag_constants import RoleTag

VALID_MODELS_FOR_BILLING = [
    "NousResearch/Nous-Hermes-Llama2-13b", "HuggingFaceH4/zephyr-7b-alpha",
    "NousResearch/Redmond-Puffin-13B", "teknium/OpenHermes-2-Mistral-7B"
]


class LlamagatewayPlugin(Generator):
    """
    Plugin for generating text using LLama model.
    """

    class LlamagatewayPluginConfig(Config):
        api_key: str = Field(
            "",
            description=
            "An openAI API key to use. If left default, will use Steamship's API key.",
        )
        max_tokens: int = Field(
            256,
            description=
            "The maximum number of tokens to generate per request. Can be overridden in runtime options.",
        )
        model: Optional[str] = Field(
            "NousResearch/Nous-Hermes-Llama2-13b",
            description=
            "The OpenAI model to use. Can be a pre-existing fine-tuned model.",
        )
        temperature: Optional[float] = Field(
            0.9,
            description=
            "Controls randomness. Lower values produce higher likelihood / more predictable results; "
            "higher values produce more variety. Values between 0-1.",
        )
        top_p: Optional[int] = Field(
            0.6,
            description=
            "Controls the nucleus sampling, where the model considers the results of the tokens with "
            "top_p probability mass. Values between 0-1.",
        )
        presence_penalty: Optional[int] = Field(
            0,
            description=
            "Control how likely the model will reuse words. Positive values penalize new tokens based on "
            "whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Number between -2.0 and 2.0.",
        )
        frequency_penalty: Optional[int] = Field(
            0,
            description=
            "Control how likely the model will reuse words. Positive values penalize new tokens based on "
            "their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Number between -2.0 and 2.0.",
        )
        moderate_output: bool = Field(
            False,
            description=
            "Pass the generated output back through OpenAI's moderation endpoint and throw an exception "
            "if flagged.",
        )
        max_retries: int = Field(
            5,
            description="Maximum number of retries to make when generating.")
        request_timeout: Optional[float] = Field(
            600,
            description=
            "Timeout for requests to OpenAI completion API. Default is 600 seconds.",
        )
        n: Optional[int] = Field(
            1, description="How many completions to generate for each prompt.")

        default_role: str = Field(
            RoleTag.USER.value,
            description=
            "The default role to use for a block that does not have a Tag of kind='role'",
        )

        default_system_prompt: str = Field(
            "",
            description=
            "System prompt that will be prepended before every request")

    @classmethod
    def config_cls(cls) -> Type[Config]:
        return cls.LlamagatewayPluginConfig

    config: LlamagatewayPluginConfig

    def __init__(
        self,
        client: Steamship = None,
        config: Dict[str, Any] = None,
        context: InvocationContext = None,
    ):
        # Load original api key before it is read from TOML, so we know to restrict models for billing
        #original_api_key = config.get("openai_api_key", "")
        super().__init__(client, config, context)
        if self.config.model not in VALID_MODELS_FOR_BILLING:
            raise SteamshipError(
                f"This plugin cannot be used with model {self.config.model}. Valid models are {VALID_MODELS_FOR_BILLING}"
            )
        openai.api_base = "https://api.llamagateway.com/sizes/40/v1"
        openai.api_key = "dummy"

    def prepare_message(self, block: Block) -> Dict[str, str]:
        role = None
        name = None
        for tag in block.tags:
            if tag.kind == TagKind.ROLE:
                role = tag.name

            if tag.kind == "name":
                name = tag.name

        if role is None:
            role = self.config.default_role

        if name:
            return {"role": role, "content": block.text, "name": name}
        else:
            return {"role": role, "content": block.text}

    def prepare_messages(self, blocks: List[Block]) -> List[Dict[str, str]]:
        messages = []
        if self.config.default_system_prompt != "":
            messages.append({
                "role": RoleTag.SYSTEM,
                "content": self.config.default_system_prompt
            })
        # TODO: remove is_text check here when can handle image etc. input
        messages.extend([
            self.prepare_message(block) for block in blocks
            if block.text is not None and block.text != ""
        ])
        return messages

    def generate_with_retry(self, user: str, messages: List[Dict[str, str]],
                            options: Dict) -> (List[Block]):
        """Call the API to generate the next section of text."""
        logging.info(
            f"Making LLama chat completion call on behalf of user with id: {user}"
        )
        options = options or {}
        stopwords = options.get("stop", None)
        functions = options.get("functions", None)

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(jitter=5),
            before_sleep=before_sleep_log(logging.root, logging.INFO),
            retry=(
                retry_if_exception_type(openai.error.Timeout)
                | retry_if_exception_type(openai.error.APIError)
                | retry_if_exception_type(openai.error.APIConnectionError)
                | retry_if_exception_type(openai.error.RateLimitError)
                | retry_if_exception_type(
                    ConnectionError
                )  # handle 104s that manifest as ConnectionResetError
                | retry_if_exception_type(requests.exceptions.HTTPError
                                          )  # Added retry check for HTTPError
            ),
            after=after_log(logging.root, logging.INFO),
        )
        def _generate_with_retry() -> Any:
            kwargs = dict(model=self.config.model,
                          messages=messages,
                          user=user,
                          presence_penalty=self.config.presence_penalty,
                          frequency_penalty=self.config.frequency_penalty,
                          max_tokens=self.config.max_tokens,
                          stop=stopwords,
                          n=self.config.n,
                          temperature=self.config.temperature,
                          headers={"x-api-key": self.config.api_key})
            if functions:
                kwargs = {**kwargs, "functions": functions}
            try:
                return openai.ChatCompletion.create(**kwargs)
            except requests.exceptions.HTTPError as http_err:
                # Check if response code is 504 and raise for retry if so
                if http_err.response.status_code == 504:
                    raise
                else:
                    # If not 504, raise a retry-agnostic error to break retry loop
                    raise openai.error.OpenAIError(http_err)

        openai_result = _generate_with_retry()
        #logging.info(openai_result)
        #logging.info(
        #    "Retry statistics: " + json.dumps(_generate_with_retry.retry.statistics)
        #)

        # Fetch text from responses
        generations = []
        for choice in openai_result["choices"]:
            message = choice["message"]
            role = message["role"]
            if function_call := message.get("function_call"):
                content = json.dumps({"function_call": function_call})
            else:
                content = message.get("content", "")

            generations.append((content, role))

        return [
            Block(
                text=text,
                tags=[
                    Tag(kind=TagKind.ROLE, name=RoleTag(role)),
                ],
            ) for text, role in generations
        ]

    def run(
        self, request: PluginRequest[RawBlockAndTagPluginInput]
    ) -> InvocableResponse[RawBlockAndTagPluginOutput]:
        """Run the text generator against all the text, combined"""

        self.config.extend_with_dict(request.data.options, overwrite=True)

        messages = self.prepare_messages(request.data.blocks)

        user_id = self.context.user_id if self.context is not None else "testing"
        generated_blocks = self.generate_with_retry(
            messages=messages, user=user_id, options=request.data.options)

        return InvocableResponse(data=RawBlockAndTagPluginOutput(
            blocks=generated_blocks))
