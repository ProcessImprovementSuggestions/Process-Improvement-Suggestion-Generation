import json
import pandas as pd
from pydantic import Field, BaseModel
from typing import List


def get_user_prompt_template():
    user_prompt_template = '''Create a concise search query aimed at finding improvement suggestions that address the common process weakness mentioned in the following Texts.

Texts:
{texts}'''
    
    return user_prompt_template


def get_system_prompt():
    class Search_Query_Generation(BaseModel):
        search_query: str = Field(description="A short query to search for improvement suggestions that address the common process weakness mentioned in the Texts")
    main_model_schema = Search_Query_Generation.model_json_schema()
    json_schema_query = json.dumps(main_model_schema)

    system_prompt_template = '''You are an assistant dedicated to supporting airline operations. Based on Texts about a process weakness, create a concise search query aimed at finding improvement suggestions. The search query must be phrased as a question. Your answer must adhere to the following JSON Schema.

JSON Schema:
{json_schema_query}'''

    system_prompt = system_prompt_template.format(json_schema_query = json_schema_query)

    return system_prompt

