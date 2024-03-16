import json
import random
import pandas as pd
from typing import List
from pydantic import Field, BaseModel


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



def generate_response(client, model, user_prompt, system_prompt):
    
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    response = client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=messages,
        temperature=0,
    )

    return response


def format_context(context):
    context = "\n\n".join(context)
    return context


def get_suggestions(query, reranked_query_results, client, model):

    system_prompt = get_system_prompt()
    user_prompt_template = get_user_prompt_template()

    context = [reranked_query_result[0] for reranked_query_result in reranked_query_results]

    #Generate suggestions
    context = format_context(context)

    response = generate_response(client, model, user_prompt_template.format(context=context, query=query), system_prompt)

    try:
        improvement_suggestions = json.loads(response.choices[0].message.content)['improvement_suggestion']
    except:
        improvement_suggestions = 'N/A'

    return improvement_suggestions
