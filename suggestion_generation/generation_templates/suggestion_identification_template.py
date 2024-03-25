import json
from pydantic import Field, BaseModel
from typing import List


def get_user_prompt_template():
    user_prompt_template = """You are presented with a Question asking for a process improvement suggestion. Generate one corresponding suggestion based on the provided Context Information.

Context Information:
{context}

Question: {query}
"""
    
    return user_prompt_template


def get_system_prompt():
    class Improvement_Suggestion_Generation(BaseModel):
        improvement_suggestion: str = Field(description="A process improvement suggestion that addresses the Question")
    main_model_schema = Improvement_Suggestion_Generation.model_json_schema()
    json_schema_suggestions = json.dumps(main_model_schema)

    system_prompt_template = '''As an assistant dedicated to supporting airline operations, you are presented with Questions asking for improvement suggestions for specific process weaknesses. For each Question, generate a corresponding suggestion based on the provided Context Information. Do not make up any suggestions but only use the Context Information to generate suggestions. Your answer must adhere to the following JSON Schema.

JSON Schema:
{json_schema_suggestions}'''

    system_prompt = system_prompt_template.format(json_schema_suggestions = json_schema_suggestions)

    return system_prompt
