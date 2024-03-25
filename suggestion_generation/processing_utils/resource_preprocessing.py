import pandas as pd


def tweet_date_range(corpus_data, date_column, date_range):
    start_year = date_range[0]
    end_year = date_range[1]
    corpus_data['created_at_clean'] = pd.to_datetime(corpus_data[date_column])
    corpus_data = corpus_data[(corpus_data.created_at_clean.dt.year > start_year) & (corpus_data.created_at_clean.dt.year < end_year)]
    corpus_data.reset_index(drop=True, inplace=True)
    corpus_data = corpus_data.drop(columns=['created_at_clean'])
    return corpus_data

def tweet_shuffle(corpus_data, frac, random_state):
    corpus_data = corpus_data.sample(frac=frac, random_state=random_state) #41
    corpus_data.reset_index(drop=True, inplace=True)
    return corpus_data

def rm_links_handles(corpus_data, text_column):
    corpus_data['text_clean'] = corpus_data[text_column].str.replace(r'http\S+', '', regex=True)
    corpus_data['text_clean'] = corpus_data['text_clean'].str.replace(r'@\w+', '', regex=True)
    return corpus_data

def cross_dataset_preprocessing(corpus_data, text_column, source_column):
    corpus_data = corpus_data.drop(columns=corpus_data.columns.difference([source_column, text_column]))
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
