# Llamagateway Plugin for Steamship

This plugin provides access to llamagateway

## Usage


### Examples

#### Basic

```python
llama = steamship.use_plugin("llamagateway-plugin")
task = llama.generate(text=prompt)
task.wait()
for block in task.output.blocks:
    print(block.text)
```

#### With agent

completions = self.llm.complete(prompt=prompt,                               
                                max_retries=4)
#Log agent raw output
logging.warning("\n\nOutput form Llama: " + completions[0].text + "\n\n")
```



