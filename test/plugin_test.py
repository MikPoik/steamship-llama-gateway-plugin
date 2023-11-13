"""Test gpt-4 generation """
import json
from typing import Optional

from steamship import Block, File, Steamship, MimeTypes, Tag
from steamship.data import TagKind
from steamship.data.tags.tag_constants import RoleTag

GENERATOR_HANDLE = "llamagateway-plugin-dev"

with Steamship.temporary_workspace() as steamship:
    llama = steamship.use_plugin(GENERATOR_HANDLE,
                                 config={
                                     "api_key": "",
                                     "model":
                                     "teknium/OpenHermes-2-Mistral-7B",
                                     "max_tokens": 500,
                                     "temperature": 0.9
                                 })
    file = File.create(
        steamship,
        blocks=[
            Block(
                text="""You are a creative character generator.""",
                tags=[Tag(kind=TagKind.ROLE, name=RoleTag.SYSTEM)],
                mime_type=MimeTypes.TXT,
            ),
            Block(
                text="""write me fictional female tinder character
""",
                tags=[Tag(kind=TagKind.ROLE, name=RoleTag.USER)],
                mime_type=MimeTypes.TXT,
            ),
        ],
    )
    generate_task = llama.generate(input_file_id=file.id, )
    generate_task.wait()
    output = generate_task.output
    #print(output)
    print(output.blocks[0].text)
