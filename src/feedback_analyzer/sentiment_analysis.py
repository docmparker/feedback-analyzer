from .utils import ToolSchema, escape_xml
from .models_common import InputModel, SurveyTaskProtocol, CommentModel, LLMConfig
from .single_input_task import apply_task
from pydantic import Field, validate_call, BaseModel
from typing import Type, Literal
from functools import partial
from . import batch_runner as br
from textwrap import fill
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import asyncio
from ast import literal_eval


class Sentiment(BaseModel):
    overall_sentiment_score: float
    confidence: float
    overall_sentiment: Literal['positive', 'negative', 'neutral']

class SentimentAnalysisResult(ToolSchema):
    """Record the reasoning and sentiment scores for a survey comment"""
    reasoning: str = Field("The comment had no content", description="The reasoning for the sentiment score assignments")
    positive_score: float = Field(default=0.0, ge=0.0, le=1.0, description="The positive sentiment score for the comment, ranging from 0.0 to 1.0.")
    negative_score: float = Field(default=0.0, ge=0.0, le=1.0, description="The negative sentiment score for the comment, ranging from 0.0 to 1.0.")
    neutral_score: float = Field(default=0.0, ge=0.0, le=1.0, description="The neutral sentiment score for the comment, ranging from 0.0 to 1.0.")

    def _overall_sentiment_score(self):
        """Takes a weighted average of the sentiment scores to get an overall sentiment score.
        Translates to a 0-1 scale."""
        combined_score = (self.positive_score * 1) + (self.neutral_score * 0) + (self.negative_score * -1)
        normalized_score = (combined_score + 1) / 2
        return normalized_score

    def _confidence(self):
        """Calculate the confidence in the sentiment by taking the difference between the top two sentiment scores"""
        scores = [self.positive_score, self.neutral_score, self.negative_score]
        top_score = max(scores)
        scores.remove(top_score)
        second_score = max(scores)
        return top_score - second_score

    @property
    def overall_sentiment(self):
        """Calculate the overall sentiment based on the overall sentiment score"""
        sentiment_score = self._overall_sentiment_score()
        confidence = self._confidence()
        
        if sentiment_score > 0.6:
            sentiment = "positive"
        elif sentiment_score < 0.4:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return Sentiment(overall_sentiment_score=sentiment_score, confidence=confidence, overall_sentiment=sentiment)
    

class SentimentAnalysis(SurveyTaskProtocol):
    """Class for sentiment analysis task"""
    def __init__(self, question: str):
        self.question = question

    @property
    def input_class(self) -> Type[InputModel]:
        return CommentModel

    def prompt_messages(self, task_input: CommentModel) -> list[dict[str, str]]:
        """Creates the messages for the sentiment analysis prompt"""

        # question = "What could be improved about the course?"

        user_message = f"""You are a skilled assistant that determines the sentiment of \
student course feedback comments.

You will be provided with a comment from a student course feedback survey in <comment> tags.
The comment was in response to the survey question: "{self.question}".
Your goal is to determine sentiment scores for the comment and provide your reasoning for the sentiment scores.

First read the user comment:
<comment>
{escape_xml(task_input.comment)}
</comment>

Then reason through what the sentiment (positive, negative, neutral) scores for the comment are and why. \
Remember that all your sentiment scores must be floating point numbers between 0.0 and 1.0. \
When you are done reasoning and scoring the comment, call the `SentimentAnalysisResult` \
tool."""

        messages =  [  
        {'role':'user', 'content': user_message}]

        return messages

    @property
    def result_class(self) -> Type[ToolSchema]:
        """Returns the result class for sentiment analysis"""
        return SentimentAnalysisResult
    

# for direct use
@validate_call
async def classify_sentiment(*, comments: list[str | float | None], question: str, llm_config: LLMConfig | None = None) -> list[ToolSchema]:
    """Classify the sentiment for each of a list of comments, based on a particular question 
    
    Returns a list of SentimentAnalysisResult objects
    """

    if not llm_config:
        llm_config = LLMConfig()

    comments_to_test: list[CommentModel] = [CommentModel(comment=comment) for comment in comments]
    survey_task: SurveyTaskProtocol = SentimentAnalysis(question=question)
    sentiment_task = partial(apply_task, 
                      get_prompt=survey_task.prompt_messages, 
                      result_class=survey_task.result_class,
                      llm_config=llm_config)
    sentiment_results = await br.process_tasks(comments_to_test, sentiment_task)

    return sentiment_results

# ====================================================================================================
# for agent tool use

class SentimentAnalysisTool(ToolSchema):
    """Tool to analyze the sentiment of a list of comments, given the list of comments and the survey question \
that the comments are in response to. Returns a list of SentimentAnalysisResult objects."""
    comments: list[str | float | None] = Field(..., description="List of comments to analyze")
    question: str = Field(..., description="The survey question that the comments are in response to")

    @staticmethod
    def execute(comments: list[str | float | None], question: str) -> list[SentimentAnalysisResult]:
        # sometimes this gets called with a string representation of a list, rather than an actual list
        if isinstance(comments, str):
            comments = literal_eval(comments)
        return asyncio.run(classify_sentiment(comments=comments, question=question))


class SentimentAnalysisVisualizeTool(ToolSchema):
    """Tool to visualize the sentiment analysis results, given a list of SentimentAnalysisResult objects.
    Returns a visualization of the sentiment analysis results."""
    sentiment_results: list[SentimentAnalysisResult] = Field(..., description="List of SentimentAnalysisResult objects")

    @staticmethod
    def execute(sentiment_results: list[SentimentAnalysisResult]):
        # make sure this is a list and not just a string representation of a list
        if isinstance(sentiment_results, str):
            sentiment_results = literal_eval(sentiment_results)

        # make sure the results are SentimentAnalysisResult objects and not just plain dicts
        if not all(isinstance(result, SentimentAnalysisResult) for result in sentiment_results):
            sentiment_results = [SentimentAnalysisResult(**result) for result in sentiment_results]

        visualize_sentiment_results(sentiment_results)

        return "Visualization displayed"


# ====================================================================================================
# some convenience functions for display

def display_top_comments(comments: list[str], sentiment_results: SentimentAnalysisResult, n=2, sentiment="positive"):
    comment_result_pairs = list(zip(comments, sentiment_results))
    def format_sentiment(result):
        return (f"(Overall score: {result.overall_sentiment.overall_sentiment_score:.2f})\n"
                f"Distribution: +{result.positive_score:.2f}, "
                f"{result.neutral_score:.2f}, "
                f"-{result.negative_score:.2f}")

    sorted_pairs = sorted(comment_result_pairs, key=lambda pair: pair[1].overall_sentiment.overall_sentiment_score, reverse=True)
    selected_pairs = sorted_pairs[:n] if sentiment == "positive" else sorted_pairs[-n:]

    print(f"\nTop {n} {sentiment.capitalize()} Comments:")
    print("=" * 80)

    for comment, result in selected_pairs:
        wrapped_comment = fill(str(comment), width=70)
        formatted_sentiment = format_sentiment(result)
        
        print(f"Comment: {wrapped_comment}")
        print(f"Sentiment: {formatted_sentiment}")
        print("-" * 80)


def visualize_sentiment_results(sentiment_results: list[SentimentAnalysisResult], 
                                figsize=(16, 8), 
                                max_comments=None):
    """
    Create visualizations for sentiment analysis results.
    
    Parameters:
    sentiment_results (list): List of SentimentAnalysisResult objects
    figsize (tuple): Figure size for the plot (width, height)
    max_comments (int): Maximum number of comments to show in the breakdown graph. If None, show all.
    """

    # Close any existing figures to start fresh
    plt.close('all')

    sentiment_colors = {
        'positive': '#4575B4',  # Warm blue
        'neutral': '#BDBDBD',   # Light gray
        'negative': '#D73027'   # Muted orange
    }

    # Convert the results to a DataFrame
    data = []
    for i, result in enumerate(sentiment_results):
        sentiment = result.overall_sentiment
        data.append({
            'id': i,
            'positive_score': result.positive_score,
            'neutral_score': result.neutral_score,
            'negative_score': result.negative_score,
            'overall_score': sentiment.overall_sentiment_score,
            'sentiment': sentiment.overall_sentiment
        })

    df = pd.DataFrame(data)

    # Sort the DataFrame by overall_score in descending order
    df_sorted = df.sort_values('overall_score', ascending=False).reset_index(drop=True)

    # Limit the number of comments if specified
    if max_comments is not None:
        df_sorted = df_sorted.head(max_comments)

    # Create a new figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 1. Distribution of Overall Sentiment Scores
    sns.histplot(data=df, x='overall_score', hue='sentiment', kde=False, palette=sentiment_colors, ax=ax1, bins=20, binrange=(0.0, 1.0))
    ax1.set_title('Distribution of Overall Sentiment Scores')
    ax1.set_xlabel('Overall Sentiment Score')
    ax1.set_ylabel('Count')
    ax1.set_xlim(0.0, 1.0)  # Set x-axis range from 0.0 to 1.0

    handles, labels = ax1.get_legend_handles_labels()
    all_handles = [plt.Rectangle((0,0),1,1, color=sentiment_colors[sent]) for sent in sentiment_colors]
    all_labels = list(sentiment_colors.keys())
    ax1.legend(all_handles, all_labels, title='Sentiment')

    # 2. Breakdown of Sentiment Scores (sorted)
    df_sorted[['positive_score', 'neutral_score', 'negative_score']].plot(
        kind='bar', 
        stacked=True, 
        ax=ax2,
        color=[sentiment_colors['positive'], sentiment_colors['neutral'], sentiment_colors['negative']]
    )
    ax2.set_title('Breakdown of Sentiment Scores (Sorted by Overall Score)')
    ax2.set_xlabel('Comment ID (Sorted)')
    ax2.set_ylabel('Score')
    ax2.legend(title='Sentiment Type')

    # Adjust layout
    plt.tight_layout()

    plt.show()