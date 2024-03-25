import json
import pandas as pd
from processing_utils import resource_preprocessing
from pydantic import Field, BaseModel
from typing import List


def get_user_prompt_template():
    user_prompt_template = '''Identify each process weakness mentioned in the Tweet.
    
Tweet: {prompt_feedback}'''
    
    return user_prompt_template


def get_system_prompt():
    class Process_Weakness_Identification(BaseModel):
        process_weaknesses: List[str] = Field(description="A list of all the process weaknesses mentioned in the Tweet")
    main_model_schema = Process_Weakness_Identification.model_json_schema()
    json_schema_weaknesses = json.dumps(main_model_schema)

    system_prompt_template = '''As an assistant dedicated to supporting airline operations, your task is to identify process weaknesses mentioned in Tweets. You do not make up any process weaknesses. Your answer must adhere to the following JSON Schema.

JSON Schema:
{json_schema_weaknesses}'''

    system_prompt = system_prompt_template.format(json_schema_weaknesses = json_schema_weaknesses)

    return system_prompt


def get_system_few_shot_prompts():

    user_prompt_template = get_user_prompt_template()
    system_prompt = get_system_prompt()

    few_shot_feedback = [
        '''First, I was rebooked on a different flight and now I received my suitcase completely damaged.''',
        '''I had a wonderful flight!''',
        '''YOU LOST OUR GODDAMN BAGS! We have to go to a conference tomorrow. Gonna have to buy a nice outfit in the morning.''',
        '''My flight to Miami was delayed by 5 hours. You never fail to disappoint.''',
        '''I've gone thru unnecessary foolishness with you all day and I'm over it!'''
        
    ]

    few_shot_weaknesses = [
        {"process_weaknesses": ["I was rebooked on a different flight.", "I received my suitcase completely damaged."]},
        {"process_weaknesses": []},
        {"process_weaknesses": ["You lost our bags."]},
        {"process_weaknesses": ["My flight to Miami was delayed by 5 hours."]},
        {"process_weaknesses": []}
    ]

    few_shot_user_prompts = [user_prompt_template.format(prompt_feedback = feedback_i) for feedback_i in few_shot_feedback]
    few_shot_assistant_prompts = [json.dumps(weakness_i) for weakness_i in few_shot_weaknesses]

    system_few_shot_prompts=[]
    system_few_shot_prompts.append({"role": "system", "content": system_prompt})
    for few_shot_user_prompt_i, few_shot_assistant_prompt_i in zip(few_shot_user_prompts, few_shot_assistant_prompts):
            system_few_shot_prompts.append({"role": "user", "content": few_shot_user_prompt_i})
            system_few_shot_prompts.append({"role": "assistant", "content": few_shot_assistant_prompt_i})

    return system_few_shot_prompts
