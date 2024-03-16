import os
import re
import pandas as pd


def prepare_tweet_dataset(path_tweets_raw, path_tweets_raw_shuffled, start_year, end_year, source_column, text_column, date_column):
    tweets = pd.read_csv(path_tweets_raw)
    tweets['created_at_clean'] = pd.to_datetime(tweets[date_column])
    tweets_p2014b2022 = tweets[(tweets.created_at_clean.dt.year > start_year) & (tweets.created_at_clean.dt.year < end_year)]
    tweets_p2014b2022_shuffled = tweets_p2014b2022.sample(frac=1, random_state=41)
    tweets_p2014b2022_shuffled.drop(columns=tweets_p2014b2022.columns.difference([source_column, text_column, date_column]), inplace=True)

    tweets_p2014b2022_shuffled.to_pickle(path_tweets_raw_shuffled)

def clean_tweets(path_tweets_raw_shuffled, path_tweets_clean_shuffled, text_column):
    tweets = pd.read_pickle(path_tweets_raw_shuffled)
    tweets['text_clean'] = tweets[text_column].str.replace(r'http\S+', '', regex=True)
    tweets['text_clean'] = tweets['text_clean'].str.replace(r'@\w+', '', regex=True)
    tweets.to_pickle(path_tweets_clean_shuffled)


def preprocessing(corpus_data, source_column, text_column):
    corpus_data.drop(columns=corpus_data.columns.difference([source_column, text_column]), inplace=True)
    corpus_data.dropna(inplace=True)
    corpus_data.drop_duplicates(subset=[source_column], inplace=True)
    corpus_data[text_column] = corpus_data[text_column].str.strip()
    corpus_data = corpus_data[corpus_data[text_column] != '']
    corpus_data.reset_index(drop=True, inplace=True)

    return corpus_data


def create_split_documents(texts, metadata, embedder):
    split_documents = []

    inputs = embedder.tokenizer(texts, truncation=True, padding=True, return_overflowing_tokens=True, max_length=embedder.get_max_seq_length(), return_tensors='pt')
    decoded_inputs = embedder.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)

    for content_i, meta_i in zip(decoded_inputs, inputs["overflow_to_sample_mapping"]):
        split_documents.append({'page_content': content_i, 'metadata': metadata[meta_i]})
    
    del inputs

    return split_documents