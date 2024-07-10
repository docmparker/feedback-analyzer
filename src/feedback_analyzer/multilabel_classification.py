from ast import literal_eval
import asyncio
from enum import Enum
from pydantic import Field, ConfigDict, validate_call
from pydantic.main import create_model
from functools import partial
from .utils import ToolSchema, escape_xml
import yaml
from .models_common import SurveyTaskProtocol, InputModel, CommentModel, LLMConfig
from .single_input_task import apply_task
from . import batch_runner as br
from typing import Type
import textwrap

# load the tags as a list of dicts, each with a 'topic' and 'description'
# in the yaml, these are all under the root of 'tags'
# It is loaded here for the default set of tags we developed for course feedback
with open('../data/tags_8.yaml', 'r') as file:
    data = yaml.safe_load(file)

default_tags_list: list[dict[str, str]] = data['tags']


class MultiLabelClassification(SurveyTaskProtocol):
    """Class for multilabel classification"""
    def __init__(self, tags_list: list[dict[str, str]]):
        """Initialize the multilabel classification task with a list of tags, 
        each a dict with a 'topic' and 'description' key"""
        self.tags_list = tags_list
        self._result_class = None
        self._tags_for_prompt = None

    @property
    def input_class(self) -> Type[InputModel]:
        return CommentModel
    
    @property
    def tags_for_prompt(self) -> str:
        """tags_for_prompt is a list of tags that will be used in the prompt, each
        a dict with a 'topic' and 'description' key.
        """

        def format_topic_name(topic: str):
            """Format a topic name for use in the prompt"""
            return topic.replace(' ', '_').lower()
        
        def format_tag(tag: dict):
            """Format a tag for use in the prompt"""
            return textwrap.dedent(f"""    <category>
        <topic>{format_topic_name(tag['topic'])}</topic>
        <description>{tag['description']}</description>
    </category>""")

        if not self._tags_for_prompt: 
            tags_for_prompt = '\n'.join([format_tag(tag) for tag in self.tags_list])
            self._tags_for_prompt = tags_for_prompt

        return self._tags_for_prompt


    def prompt_messages(self, task_input:CommentModel) -> list[dict[str, str]]:
        """Creates the messages for the multilabel classification prompt"""

        user_message = f"""You are a skilled assistant that classifies student course \
feedback comments.

You will be provided with a comment from a student course feedback survey in <comment> tags. \
The goal is to categorize the comment with as many of the following categories as apply:

<categories>
{textwrap.indent(self.tags_for_prompt, prefix='    ')}
</categories>

First read the user comment:
<comment>
{escape_xml(task_input.comment)}
</comment>

Then reason through which categories apply to the comment, justifying each choice. \
When you are done reasoning and categorizing the comment, call the `MultiLabelClassificationResult` \
tool.""" 

        messages =  [  
            {'role':'user', 
             'content': user_message}
        ]

        return messages

    @property
    def result_class(self) -> Type[ToolSchema]:
        """Returns the result class for multilabel classification, dynamically creating it if necessary"""

        if not self._result_class:
            # only need to dynamically create the model once
            class CategoryValue(Enum):
                ZERO = 0
                ONE = 1

            aliases = {'_'.join(tag['topic'].split()): tag['topic'] for tag in self.tags_list}
            def tag_alias_generator(name: str) -> str:
                """Generate aliases based on what was passed with the tags"""
                return aliases[name]

            tag_fields = {'_'.join(tag['topic'].split()): (CategoryValue, CategoryValue.ZERO) for tag in self.tags_list}
            # CategoriesModel = create_model('Categories', **tag_fields, __base__=ToolSchema)
            # note that the tool schema uses the aliases to create the json schema for the model
            CategoriesModel = create_model('Categories', **tag_fields, __config__=ConfigDict(alias_generator=tag_alias_generator, __base__=ToolSchema))

            # I am deciding to only include the descriptions of the categories spelled out in the system message to the 
            # model but not to repeat those in the field descriptions for the schema.

            # Create the model
            class MultilabelClassificationResult(ToolSchema):
                """Store the multilabel classification and reasoning of a comment"""
                reasoning: str = Field("The comment had no content", description="The reasoning for the classification")
                categories: CategoriesModel = Field(CategoriesModel(), description="The categories that the comment belongs to")

            self._result_class = MultilabelClassificationResult

        return self._result_class


@validate_call
async def multilabel_classify(*, comments: list[str | float | None], 
                              tags_list: list[dict[str, str]] | None = None, 
                              llm_config: LLMConfig | None = None) -> ToolSchema:
    """Multilabel classify a list of comments, based on a list of categories (tags)
    
    Returns a list of MultiLabelClassificationResult objects
    """

    if not tags_list:
        tags_list = default_tags_list

    if not llm_config:
        llm_config = LLMConfig()

    survey_task: SurveyTaskProtocol = MultiLabelClassification(tags_list=tags_list)
    comments_to_test: list[CommentModel] = [CommentModel(comment=comment) for comment in comments]
    mlc_task = partial(apply_task, 
                       get_prompt=survey_task.prompt_messages, 
                       result_class=survey_task.result_class, 
                       llm_config=llm_config)
    classifications = await br.process_tasks(comments_to_test, mlc_task)

    return classifications


# ====================================================================================================
# for agent tool use

class MultlabelClassificationTool(ToolSchema):
    """Tool to classify a list of comments, given the list of comments and a list of tags. \
Returns a list of MultiLabelClassificationResult objects."""
    comments: list[str | float | None] = Field(..., description="List of comments to extract excerpts from")
    tags_list: list[dict[str, str]] | None = Field(None, description="The list of tags and descriptions to classify the comments with")

    @staticmethod
    def execute(comments: list[str | float | None], tags_list: list[dict[str, str]] | None = None) -> list[ToolSchema]:
        # sometimes this gets called with a string representation of a list, rather than an actual list
        if isinstance(comments, str):
            comments = literal_eval(comments)
        return asyncio.run(multilabel_classify(comments=comments, tags_list=tags_list))