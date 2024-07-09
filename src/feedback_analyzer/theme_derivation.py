from ast import literal_eval
import asyncio
import textwrap
from .utils import ToolSchema, escape_xml
from .models_common import SurveyTaskProtocol, InputModel, CommentModel, CommentBatch, LLMConfig
from pydantic import Field, validate_call, conint, BaseModel
from typing import Type
from .single_input_task import apply_task
from feedback_analyzer.summarization import summarize_comments
from functools import partial
import logging
import sys

logger = logging.getLogger("feedback_analyzer")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Create the models

class Theme(ToolSchema):
    """A theme and relevant extracted quotes derived from a batch of comments"""
    theme_title: str = Field("", description="A short name for the theme (5 words or less)")
    description: str = Field("", description="A description of the theme (a short bulleted list of key points)")
    citations: list[str] = Field([], description="A list of citations (exact extracted quotes) related to the theme")

class DerivedThemes(ToolSchema, InputModel):
    """Store the themes derived from a batch of comments"""
    themes: list[Theme] = Field([], description="A list of themes")

    def is_empty(self) -> bool:
        """Returns True if all themes are empty"""
        return len(self.themes) == 0

class StepInput(BaseModel, InputModel):
    """Holds the comment batch and the themes narrative as input for the refinement step"""
    themes_narrative: str = Field("", description="The themes narrative")
    comment_batch: CommentBatch = Field(CommentBatch(), description="The comment batch")

    def is_empty(self) -> bool:
        """Returns True if the themes narrative is empty"""
        return not self.themes_narrative

class RefinementResult(ToolSchema):
    """Store the results of refining summary of themes from a group of survey comments"""
    reasoning: str = Field("", description="The reasoning for any themes added or subtracted")
    summary: str = Field("The comments had no content", description="The updated summary of the comments")

class RefineThemes(SurveyTaskProtocol):
    """Class for refining themes narrative against the comments they were derived from.
    This is essentially a refinement pass."""

    def __init__(self, question: str):
        self.question = question

    @property
    def input_class(self) -> Type[InputModel]:
        return StepInput

    def prompt_messages(self, task_input: StepInput) -> list[dict[str, str]]:
        """Creates the messages for the refinement prompt"""

        comment_list = "\n".join([f"<comment>\n{escape_xml(comment.comment)}\n</comment>" 
                                      for comment in task_input.comment_batch.comments 
                                      if not comment.is_empty()])

        user_message = f"""You are a skilled assistant that checks and, if necessary, \
revises a summary of the themes of a set of student course feedback comments.

You will be provided with the group of comments from a student course feedback survey \
that the summary was derived from. Each comment will be in <comment> tags. \
Each original comment was in response to the question: "{self.question}".

First, read the comments carefully to understand the feedback provided by the students:
<comments>
{textwrap.indent(comment_list, prefix='    ')}
</comments>

Then review the summary of the major themes of feedback that were previously found: 
<themes_summary>
{task_input.themes_narrative}
<themes_summary>

Check this summary and revise it if necessary. Here is what you are looking for:
- The summary should be comprehensive (in other words, capture all ideas that are \
emphasized by multiple students). 
- Every theme should be reflected in at least two comments.

If the summary already fulfills these criteria, you can respond with the same summary. \
You should also concisely capture your reasoning for any themes you added or subtracted. \
When you are done with your checking, making any refinements, and capturing your reasoning, \
call the `RefinementResult` tool."""

        messages =  [  
            {'role':'user', 'content': user_message}
        ]

        return messages

    @property
    def result_class(self) -> Type[ToolSchema]:
        """Returns the result class for theme refinement."""
        return RefinementResult 


class DeriveThemes(SurveyTaskProtocol):
    """Class for extracting themes with titles, descriptions, and citations from a batch of comments
    and a themes narrative"""
    
    def __init__(self, question: str):
        self.question = question

    @property
    def input_class(self) -> Type[InputModel]:
        return StepInput

    def prompt_messages(self, task_input: StepInput) -> list[dict[str, str]]:
        """Creates the messages for the refinement prompt"""

        comment_list = "\n".join([f"<comment>\n{escape_xml(comment.comment)}\n</comment>" 
                                      for comment in task_input.comment_batch.comments 
                                      if not comment.is_empty()])

        user_message = f"""You are a skilled assistant that extracts themes with citations \
based on a summary of the themes of a set of student course feedback comments and the comments themselves.

Here is the summary of the major themes of feedback that were previously found: 
<themes_summary>
{task_input.themes_narrative}
<themes_summary>

For each theme, come up with a title and a brief description. The title should be a short phrase \
(2-5 words) that captures the essence of the theme. The description should be a bulleted list \
that describes the key points of the theme. If the summary already had the format of titles and descriptions, \
you can use them as is.

Next, you will come up with citations (exact quotes from the student comments) that support the themes. \
You will be provided with the group of comments from a student course feedback survey \
that the themes were derived from. Each comment will be in <comment> tags. \
Each original comment was in response to the question: "{self.question}".

First, read the comments carefully:
<comments>
{textwrap.indent(comment_list, prefix='    ')}
</comments>

Then add citations to each theme. For each theme, provide 3 exact quotes from distinct survey comments \
that support this theme. Each quote should have enough context to be understood. Do not add or alter words \
in the quotes under any circumstances. If there are less than 3 quotes, then include as many as you can.

When you are done with coming up with titles, descriptions, and citations, \
call the `DerivedThemes` tool to record your results."""

        messages =  [  
            {'role':'user', 'content': user_message}
        ]

        return messages

    @property
    def result_class(self) -> Type[ToolSchema]:
        """Returns the result class for theme derivation."""
        return DerivedThemes


@validate_call
async def find_themes(comments: list[str | float | None], 
                      question: str, 
                      llm_config: LLMConfig | None = None) -> DerivedThemes:
    """Finds themes from a batch of comments
    1. [comments] --> summarize the themes --> [themes narrative] 
    2. [themes narrative, comments] --> check to see if captured all themes --> [revised themes narrative] 
    3. [revised themes narrative, comments] --> add citations to themes, descriptions, titles --> [revised themes, citations]
    """

    if not llm_config:
        llm_config = LLMConfig()

    logger.info("Summarizing comments to find themes")
    summarization_result = await summarize_comments(comments=comments, 
                                                    question=question,
                                                    llm_config=llm_config)
    comment_batch = CommentBatch(comments=[CommentModel(comment=comment) for comment in comments])
    
    step2_input = StepInput(themes_narrative=summarization_result.summary,
                                 comment_batch=comment_batch)
    survey_task2: SurveyTaskProtocol = RefineThemes(question=question)
    refinement_task = partial(apply_task, 
                      get_prompt=survey_task2.prompt_messages, 
                      result_class=survey_task2.result_class,
                      llm_config=llm_config)

    logger.info("Checking and refining themes summary")
    refinement_result = await refinement_task(step2_input)

    step3_input = StepInput(themes_narrative=refinement_result.summary,
                                    comment_batch=comment_batch)
    survey_task3: SurveyTaskProtocol = DeriveThemes(question=question)
    derivation_task = partial(apply_task, 
                      get_prompt=survey_task3.prompt_messages, 
                      result_class=survey_task3.result_class,
                      llm_config=llm_config)

    logger.info("Extracting and citing")
    derivation_result = await derivation_task(step3_input)

    return derivation_result


# ====================================================================================================
# for agent tool use

class ThemeDerivationTool(ToolSchema):
    """Tool to find the themes from a list of comments, given the list of comments and the survey question \
that the comments are in response to. Returns a DerivedThemes object."""
    comments: list[str | float | None] = Field(..., description="List of comments to analyze")
    question: str = Field(..., description="The survey question that the comments are in response to")

    @staticmethod
    def execute(comments: list[str | float | None], question: str) -> str:
        # sometimes this gets called with a string representation of a list, rather than an actual list
        if isinstance(comments, str):
            comments = literal_eval(comments)
        return asyncio.run(find_themes(comments=comments, question=question))


class ThemeDisplayTool(ToolSchema):
    """Tool to display the themes from a DerivedThemes object. Returns a string."""
    derived_themes: DerivedThemes = Field(..., description="DerivedThemes object to display")

    @staticmethod
    def execute(derived_themes: DerivedThemes) -> str:
        if isinstance(derived_themes, dict):
            derived_themes_typed = DerivedThemes(themes=[Theme(**theme) for theme in derived_themes['themes']])

        output = []
        for theme in derived_themes_typed.themes:
            output.append(f"Theme: {theme.theme_title}  ")
            output.append(f"Description:\n{theme.description}  ")
            output.append("Citations:  ")
            for citation in theme.citations:
                output.append(f"- \"{citation}\"  ")
            output.append("  ")  # Empty line between themes
        return "\n".join(output)