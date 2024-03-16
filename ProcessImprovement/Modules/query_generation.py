import json
import random
import pandas as pd
from typing import List
from pydantic import Field, BaseModel


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


def format_weaknesses(weaknesses):
    context = "\n\n".join(weaknesses)
    return context


def get_search_queries(weakness_cluster_batch, client, model, cluster_max_size = None):

    system_prompt = get_system_prompt()
    user_prompt_template = get_user_prompt_template()

    clusters = set(weakness_cluster_batch['cluster'].to_list())
    clusters = sorted(list(clusters))

    #Generate search queries
    cluster_queries = []

    for cluster_i in clusters:
        if cluster_i != -1:
            context = format_weaknesses(weakness_cluster_batch[weakness_cluster_batch['cluster']==cluster_i]['weakness'].to_list()[:cluster_max_size])

            response = generate_response(client, model, user_prompt_template.format(texts=context), system_prompt)

            try:
                cluster_queries.append([cluster_i, json.loads(response.choices[0].message.content)['search_query']])
            except:
                cluster_queries.append([cluster_i, ''])
        else:
            cluster_queries.append([cluster_i, ''])


    return pd.DataFrame(cluster_queries, columns=['cluster', 'search_query'])