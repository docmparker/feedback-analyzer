from ast import literal_eval
import asyncio
from .utils import ToolSchema, escape_xml
from .models_common import InputModel, LLMConfig, SurveyTaskProtocol, CommentModel
from .single_input_task import apply_task
from pydantic import Field, validate_arguments
from typing import Type
from functools import partial
from . import batch_runner as br


# Create the model - here we do it outside the class so it can also be used elsewhere if desired
# without instantiating the class
class ExcerptExtractionResult(ToolSchema):
    """Store excerpts containing a particular goal focus extracted from a student comment"""
    excerpts: list[str] = Field([], description="A list of excerpts related to the goal focus")


class ExcerptExtraction(SurveyTaskProtocol):
    """Class for excerpt extraction"""
    def __init__(self, goal_focus: str, question: str):
        self.goal_focus = goal_focus
        self.question = question

    @property
    def input_class(self) -> Type[InputModel]:
        return CommentModel

    def prompt_messages(self, task_input: CommentModel) -> list[dict[str, str]]:
        """Creates the messages for the extraction prompt"""

        # Examples of questions and goal_focus
        # question = "What could be improved about the course?"
        # goal_focus = "suggestions for improvement"

        user_message = f"""You are a skilled assistant that extracts {self.goal_focus!r} from \
student course feedback comments.

You will be provided with a comment from a student course feedback survey in <comment> tags. \
The comment was in response to the survey question: "{self.question}". \
Your task is to only select excerpts which pertain to the goal focus: {self.goal_focus!r}. \

First read the user comment:
<comment>
{escape_xml(task_input.comment)}
</comment>

Then select excerpts. Follow these rules:
- Excerpts should only be exact quotes taken from the comment; do not add or alter words \
under any circumstances.
- If you cannot extract excerpts for any reason, for example if the comment is \
not relevant to the question, you should return an empty list for excerpts.
- If there are relevant excerpts, ensure that excerpts contain all relevant text needed to interpret them - \
in other words don't extract small snippets that are missing important context.

Before finalizing excerpts, review your excerpts to see if any consecutive excerpts \
are actually about the same suggestion or part of the same thought. If so, combine them \
into a single excerpt. When you are done with your excerpts, call the `ExcerptExtractionResult` tool."""

        messages =  [  
            {'role':'user', 'content': user_message}
        ]

        return messages

    @property
    def result_class(self) -> Type[ToolSchema]:
        """Returns the result class for extraction"""
        return ExcerptExtractionResult
    

@validate_arguments
async def extract_excerpts(*, comments: list[str | float | None], 
                           question: str, 
                           goal_focus: str, 
                           llm_config: LLMConfig | None = None) -> list[ExcerptExtractionResult]:
    """Extract excerpts from a list of comments, based on a particular question and goal_focus
    
    Returns a list of ExcerptExtractionResult objects
    """

    if not llm_config:
        llm_config = LLMConfig()

    comments_to_test: list[CommentModel] = [CommentModel(comment=comment) for comment in comments]
    survey_task: SurveyTaskProtocol = ExcerptExtraction(goal_focus=goal_focus, question=question)
    ex_task = partial(apply_task, 
                      get_prompt=survey_task.prompt_messages, 
                      result_class=survey_task.result_class,
                      llm_config=llm_config)

    extractions = await br.process_tasks(comments_to_test, ex_task)

    return extractions


# ====================================================================================================
# for agent tool use

class ExcerptExtractionTool(ToolSchema):
    """Tool to extract excerpts from a list of comments, given the list of comments, the survey question \
that the comments are in response to, and the goal focus of interest. Returns a list of ExcerptExtractionResult objects."""
    comments: list[str | float | None] = Field(..., description="List of comments to extract excerpts from")
    question: str = Field(..., description="The survey question that the comments are in response to")
    goal_focus: str = Field(..., description="The goal focus of interest for the extraction")

    @staticmethod
    def execute(comments: list[str | float | None], question: str, goal_focus: str) -> list[ExcerptExtractionResult]:
        # sometimes this gets called with a string representation of a list, rather than an actual list
        if isinstance(comments, str):
            comments = literal_eval(comments)
        return asyncio.run(extract_excerpts(comments=comments, question=question, goal_focus=goal_focus))