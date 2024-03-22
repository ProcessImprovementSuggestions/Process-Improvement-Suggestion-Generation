import json
import random
import pandas as pd
from typing import List
from pydantic import Field, BaseModel


def get_user_prompt_template():
    user_prompt_template = """Use the Tweet and the Context Information to generate a very concise text that provides an improvement suggestion for each process weakness described in the Tweet.

Context Information:
{context}

Tweet: {tweet}
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


def get_answer(tweet_weakness_batch, client, model):

    system_prompt = get_system_prompt()
    user_prompt_template = get_user_prompt_template()

    tweets = tweet_weakness_batch["tweet"].to_list()
    suggestions = tweet_weakness_batch["suggestions"].to_list()
    improvement_suggestions_texts = []

    for idx, tweet_i in enumerate(tweets):
        if suggestions[idx] != []:

            context = format_context(suggestions[idx])

            #Generate answers
            response = generate_response(client, model, user_prompt_template.format(context=context, tweet=tweet_i), system_prompt)

            try:
                improvement_suggestions_texts.append(json.loads(response.choices[0].message.content)['improvement_suggestions_text'])
            except:
                improvement_suggestions_texts.append('N/A')
        else:
            improvement_suggestions_texts.append('N/A')

    return improvement_suggestions_texts
