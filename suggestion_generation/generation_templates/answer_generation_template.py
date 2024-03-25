import json
from pydantic import Field, BaseModel
from typing import List


def get_user_prompt_template():
    user_prompt_template = """Use the Tweet and the Context Information to generate a very concise text that provides an improvement suggestion for each process weakness described in the Tweet.

Context Information:
{context}

Tweet: {prompt_feedback}
"""
    
    return user_prompt_template


def get_system_prompt():
    class Improvement_Suggestion_Generation(BaseModel):
        improvement_suggestions_text: str = Field(description="A text that provides an improvement suggestion for each process weakness described in the Tweet")
    main_model_schema = Improvement_Suggestion_Generation.model_json_schema()
    json_schema_suggestions = json.dumps(main_model_schema)

    system_prompt_template = '''You are an assistant dedicated to supporting airline operations. Your task is to generate very concise texts that provide improvement suggestions for the process weaknesses described in Tweets. Do not make up any suggestions but only use the provided Tweets and Context Information to generate suggestions. Moreover, do not provide multiple suggestions for a weakness. Your answer must adhere to the following JSON Schema.

JSON Schema:
{json_schema_suggestions}'''

    system_prompt = system_prompt_template.format(json_schema_suggestions = json_schema_suggestions)

    return system_prompt
