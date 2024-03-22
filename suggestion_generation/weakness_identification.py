import json
import pandas as pd
from typing import List
import resource_preprocessing
from pydantic import Field, BaseModel


def get_user_prompt_template():
    user_prompt_template = '''Identify each process weakness mentioned in the Tweet.
    
Tweet: {tweet}'''
    
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

    few_shot_tweets = [
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

    few_shot_user_prompts = [user_prompt_template.format(tweet = tweet_i) for tweet_i in few_shot_tweets]
    few_shot_assistant_prompts = [json.dumps(weakness_i) for weakness_i in few_shot_weaknesses]

    system_few_shot_prompts=[]
    system_few_shot_prompts.append({"role": "system", "content": system_prompt})
    for few_shot_user_prompt_i, few_shot_assistant_prompt_i in zip(few_shot_user_prompts, few_shot_assistant_prompts):
            system_few_shot_prompts.append({"role": "user", "content": few_shot_user_prompt_i})
            system_few_shot_prompts.append({"role": "assistant", "content": few_shot_assistant_prompt_i})

    return system_few_shot_prompts


def generate_response(client, model, user_prompt, few_shot_prompts):
    
    messages = list(few_shot_prompts)
    messages.append({"role": "user", "content": user_prompt})
    
    response = client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=messages,
        temperature=0,
    )

    return response


def identify_weaknesses(tweets, client, model):
    '''
    Identify the weaknesses of a batch of tweets
    '''

    corpus_tweets_batch = tweets.values.tolist()

    user_prompt_template = get_user_prompt_template()
    user_prompts = [user_prompt_template.format(tweet = tweet_i[1]) for tweet_i in corpus_tweets_batch]
    system_few_shot_prompts = get_system_few_shot_prompts()

    number_excepts=0
    weaknesses_batch = []
    for idx, user_prompt_i in enumerate(user_prompts):
        try:
            response = generate_response(client, model, user_prompt_i, system_few_shot_prompts)
            response = json.loads(response.choices[0].message.content)
            process_weaknesses = response['process_weaknesses']

            for process_weakness_i in process_weaknesses:
                weaknesses_batch.append([corpus_tweets_batch[idx][0], process_weakness_i])

            corpus_tweets_batch[idx].append(process_weaknesses)
        except:
            number_excepts+=1
            corpus_tweets_batch[idx].append([])
    

    return pd.DataFrame(corpus_tweets_batch, columns=['tweetid', 'tweet', 'weaknesses']), pd.DataFrame(weaknesses_batch, columns=['tweetid', 'weakness']), number_excepts
