from typing import Callable
from pydantic import BaseModel, ValidationError
from anthropic import AsyncAnthropic
import anthropic
import time
from .utils import ToolSchema
from .models_common import LLMConfig, InputModel


aclient = AsyncAnthropic()

# takes a task object and returns the messages for the prompt
GetPrompt = Callable[[BaseModel], list[dict[str, str]]]

# signature for running tasks that take a task object and return an ToolSchema
ApplyTask = Callable[[BaseModel, GetPrompt, ToolSchema, LLMConfig | None], ToolSchema] 


async def apply_task(task_input: InputModel, 
                     get_prompt: GetPrompt, 
                     result_class: ToolSchema, 
                     llm_config: LLMConfig | None = None) -> ToolSchema:
    """Gets the result of applying an NLP task to a comment, list of comments, or some other unit of work."""
    if llm_config is None:
        llm_config = LLMConfig()

    # if the task_input has no content (is a None equivalent), return early with
    # an empty classification (filled in with defaults)
    if task_input.is_empty():
        return result_class()
    
    # expect partial application of get_prompt if needs something like the tags_list
    messages = get_prompt(task_input)
    fn_schema = result_class.tool_schema()
    # replace 'parameters' with 'input_schema'
    # fn_schema['input_schema'] = fn_schema.pop('parameters')

    # print(messages[0]['content'])

    try:
        response = await aclient.messages.create(
            model=llm_config.model,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
            messages=messages,
            tools=[fn_schema],
            tool_choice={"type": "tool", "name": fn_schema['name']},
        )
    except anthropic.RateLimitError as e:
        print(f"Rate limit error: {e}")
        print("Sleeping for 30 seconds and trying again...")
        time.sleep(30)
        # try again
        response = await aclient.messages.create(
            model=llm_config.model,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
            messages=messages,
            tools=[fn_schema],
            tool_choice={"type": "tool", "name": fn_schema['name']},
        )

    args = [content for content in response.model_dump()['content'] if content['type'] == 'tool_use'][0]['input']

    try: 
        result = result_class(**args)
    except ValidationError as e:
        # sometimes the tool call doesn't always go right (the args are malformed) so we'll try fixing once
        # with another tool call (!)
        print(f"Error: {e} for {args}.\nCalling the Fixer...")
        try:
            # use the Fixer tool to correct the ill-formed arguments
            result = await fix_tool_call(args, result_class)
        except ValidationError as e:
            print(f"Error: {e} for {args}.\nFailed to fix the ill-formed arguments. Returning default result.")
            # return the default result if there is one
            # this will appropriately raise if there is no default
            result = result_class()

    return result


async def fix_tool_call(bad_tool_args: dict, result_class: ToolSchema):
    MODEL_NAME_HAIKU = "claude-3-haiku-20240307"
    ILLFORMED_TOOL_ARGS = bad_tool_args
    fn_schema = result_class.tool_schema()
    # replace 'parameters' with 'input_schema'
    # fn_schema['input_schema'] = fn_schema.pop('parameters')

    message = await aclient.messages.create(
        model=MODEL_NAME_HAIKU,
        max_tokens=4096,
        tools=[fn_schema],
        tool_choice={"type": "tool", "name": fn_schema['name']},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""\
Here are arguments that were ill-formed and need to be fixed. They are intended for the `{fn_schema['name']}` tool. \
<illformed_tool_arguments>
{ILLFORMED_TOOL_ARGS!r}
</illformed_tool_arguments>

Figure out the intended arguments and call the `{fn_schema['name']}` tool with the corrected arguments."""},
                ],
            }
        ],
    )
    
    args = [content for content in message.model_dump()['content'] if content['type'] == 'tool_use'][0]['input']
    result = result_class(**args)
    return result

