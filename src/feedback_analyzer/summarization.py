import textwrap
from .utils import ToolSchema, escape_xml
from .models_common import InputModel, SurveyTaskProtocol, CommentModel, CommentBatch, LLMConfig
from .single_input_task import apply_task
from pydantic import Field, validate_call
from typing import Type
from functools import partial


class SummarizationResult(ToolSchema):
    """Store the results of summarizing a group of survey comments"""
    summary: str = Field("The comments had no content", description="The summary of the comments")


class Summarization(SurveyTaskProtocol):
    """Class for comment summarization task"""
    def __init__(self, question: str):
        self.question = question

    @property
    def input_class(self) -> Type[InputModel]:
        return CommentBatch

    def prompt_messages(self, task_input: CommentBatch) -> list[dict[str, str]]:
        """Creates the messages for the summarization prompt"""

        # question = "What could be improved about the course?"

        comment_list = "\n".join([f"<comment>\n{escape_xml(comment.comment)}\n</comment>" 
                                      for comment in task_input.comments 
                                      if not comment.is_empty()])

        user_message = f"""You are a skilled assistant that summarizes the themes of \
a set of student course feedback comments.

You will be provided with a group of comments from a student course feedback survey. \
Each comment will be in <comment> tags. \
Each original comment was in response to the question: "{self.question}".

First, read the comments carefully to understand the feedback provided by the students:
<comments>
{textwrap.indent(comment_list, prefix='    ')}
</comments>

Then summarize the major themes of feedback shared by the students. \
Your summary should be comprehensive (in other words, capture all ideas that are \
emphasized by multiple students). When you are done with your summary, call the `SummarizationResult` tool."""

        messages =  [  
            {'role':'user', 'content': user_message}
        ]

        return messages

    @property
    def result_class(self) -> Type[ToolSchema]:
        """Returns the result class for summarization"""
        return SummarizationResult
    

@validate_call
async def summarize_comments(*, comments: list[str | float | None], question: str, llm_config: LLMConfig | None = None) -> ToolSchema:
    """Summarize the themes of a group of comments, based on a particular question
    
    Returns a SummarizationResult object
    """

    if not llm_config:
        llm_config = LLMConfig()

    comment_list = [CommentModel(comment=escape_xml(comment)) for comment in comments]
    comments_to_test: CommentBatch = CommentBatch(comments=comment_list)
    survey_task: SurveyTaskProtocol = Summarization(question=question)
    summarization_task = partial(apply_task, 
                      get_prompt=survey_task.prompt_messages, 
                      result_class=survey_task.result_class,
                      llm_config=llm_config)

    summarization_result = await summarization_task(comments_to_test)

    return summarization_result