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
import sys
from pathlib import Path

# Make sure `test` and '../src' are on the PYTHONPATH. Otherwise, cross-test imports (e.g. to util libraries) will fail.
sys.path.append(str(Path(__file__).parent.absolute()))
sys.path.append(str(Path(__file__).parent.parent.absolute() / "src"))
from api import LlamagatewayPlugin
from steamship import Block, File, Steamship, MimeTypes, Tag
from steamship.data import TagKind
from steamship.data.tags.tag_constants import RoleTag

llama = LlamagatewayPlugin(config={
    "api_key": "",
    "model": "teknium/OpenHermes-2-Mistral-7B",
    "temperature": 0.8
})

blocks = [
    Block(
        text="You are a helpful AI assistant, answer politely.",
        tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
        mime_type=MimeTypes.TXT,
    ),
    Block(
        text="Im bob,tell me about you?",
        tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
        mime_type=MimeTypes.TXT,
    ),
]

new_blocks = llama.run(
    PluginRequest(data=RawBlockAndTagPluginInput(
        blocks=blocks,
        options={},
    )))
print(new_blocks.data.blocks[0].text)
#blocks=[
#    Block(
#        text="You are an assistant who loves to count",
#        tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)],
#        mime_type=MimeTypes.TXT,
#    ),
#    Block(
#        text="Continue this series: 1 2 3 4",
#        tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
#        mime_type=MimeTypes.TXT,
#    ),
#]

#request = PluginRequest(
#    data=RawBlockAndTagPluginInput(blocks=blocks)
#)
#response = llama.run(request)
#print(response.data.blocks[0].text)
options = {
    "functions": [{
        "name": "Search",
        "description": "useful if you need to search for weather conditions.",
        "parameters": {
            "properties": {
                "query": {
                    "title": "query",
                    "type": "string"
                }
            },
            "required": ["query"],
            "type": "object",
        },
    }]
},
