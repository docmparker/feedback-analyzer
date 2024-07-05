from .utils import ToolSchema
from typing import Callable
import time
import asyncio
from .single_input_task import InputModel

RunTask = Callable[[InputModel], ToolSchema]

# TODO: generalize this to take a list of task inputs 
async def process_tasks(task_inputs: list[InputModel],
                                      run_task: RunTask,
                                      batch_size: int=25, 
                                      batch_sleep_interval: int=20) -> list[ToolSchema]:
    """Takes a list of inputs and processes them in parallel, returning a list of 
    pydantic model responses. This is done in batches, with the batches having their 
    model calls processed in parallel. The default limits above are pretty low because Anthropic
    uses a token bucket algorithm and "Short bursts of requests at a high volume can surpass 
    the rate limit and result in rate limit errors."
    """

    print(f"processing {len(task_inputs)} inputs in batches of {batch_size}")
    print(f"sleeping for {batch_sleep_interval} seconds between batches")
    response_list: list[ToolSchema] = []
    for i in range(0, len(task_inputs), batch_size):
        input_batch = task_inputs[i:i+batch_size]

        print(f"starting {i} to {i+batch_size}")
        start_time = time.time()

        responses = await asyncio.gather(*[run_task(input) for input in input_batch])
        response_list.extend(responses)
        print(f"completed {i} to {i+batch_size}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"elapsed time: {elapsed_time}")

        if i < len(task_inputs) - batch_size:
            time_to_next_minute = batch_sleep_interval - (elapsed_time % batch_sleep_interval)
            print(f"sleeping for {time_to_next_minute} seconds")
            time.sleep(time_to_next_minute)

    return response_list
