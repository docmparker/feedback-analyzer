from pydantic import BaseModel

def comment_has_content(comment: str) -> bool:
    """Check if a comment has content"""
    none_equivalents = ['n/a', None, 'none', 'null', '', 'na', 'nan']
    return False if ((not comment) or (comment.lower() in none_equivalents)) else True

def escape_xml(text: str) -> str:
    """Escape XML characters to sanitize student comments

    Example usage: `escaped_comment = escape_xml(student_comment)`
    """
    if type(text) != str:
        return text
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

class ToolSchema(BaseModel):
    """This goes from a pydantic model to Anthropic function/tool schema"""
    @classmethod
    def tool_schema(cls):
        assert cls.__doc__, f"{cls.__name__} is missing a docstring."
        assert (
            "title" not in cls.model_fields
        ), "`title` is a reserved keyword and cannot be used as a field name."
        schema_dict = cls.model_json_schema()
        cls.remove_a_key(schema_dict, "title")

        # bizarre, but it does better (fewer malformed responses) with the extra description key still in there
        # schema_dict.pop('description')

        return {
            "name": cls.__name__,
            "description": cls.__doc__,
            "input_schema": schema_dict,
        }

    @classmethod
    def remove_a_key(cls, d, remove_key):
        if isinstance(d, dict):
            for key in list(d.keys()):
                if key == remove_key:
                    del d[key]
                else:
                    cls.remove_a_key(d[key], remove_key)