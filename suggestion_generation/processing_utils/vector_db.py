import os
import pandas as pd
import time
from processing_utils import resource_preprocessing
from qdrant_client import models


def recreate_db(qdrantdb_client, collection_name, embedder):
    #recreate collection
    qdrantdb_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=embedder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
            distance=models.Distance.COSINE,
            on_disk=True,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0,
        ),
    )

def create_db_collection(path_resources, source_column, text_column, qdrantdb_client, collection_name, embedder, cross_dataset_preprocess):
    print(f'Create {collection_name}')
    recreate_db(qdrantdb_client, collection_name, embedder)

    upload_batch_size = 100000
    uploaded_sources = set()
    len_chunks = 0

    for resource_file in os.scandir(path_resources):
        if resource_file.name.endswith('.jsonl') == True:
            print(f'Loading file: {resource_file.name}')
            corpus_data = pd.read_json(resource_file.path, lines=True)
        elif resource_file.name.endswith('.pkl') == True:
            print(f'Loading file: {resource_file.name}')
            corpus_data = pd.read_pickle(resource_file.path)
        else:
            print("Must be jsonl or pkl file")

        if cross_dataset_preprocess == True:
            corpus_data = resource_preprocessing.cross_dataset_preprocessing(corpus_data, text_column, source_column)

        corpus_texts_raw = corpus_data[text_column].tolist()
        corpus_sources_raw = corpus_data[source_column].tolist()

        corpus_texts=[]
        corpus_sources=[]

        #Check for duplicates
        for text_i, source_i in zip(corpus_texts_raw, corpus_sources_raw):
            if source_i not in uploaded_sources:
                uploaded_sources.add(source_i)
                corpus_sources.append(source_i)
                corpus_texts.append(text_i)

        corpus_sources = [{'source': source_i} for source_i in corpus_sources]
    
        start = time.time()
        for batch_i in range(0, len(corpus_texts), upload_batch_size):
            documents = resource_preprocessing.create_split_documents(corpus_texts[batch_i:batch_i+upload_batch_size], corpus_sources[batch_i:batch_i+upload_batch_size], embedder)
            documents_texts = [document_i['page_content'] for document_i in documents]
            documents_metadata = [{'source': document_i['metadata']['source'], 'page_content': document_i['page_content']} for document_i in documents]
            len_chunks+=len(documents_texts)

            documents_texts_embeddings = embedder.encode(documents_texts, batch_size=256, device='cuda', convert_to_tensor=True)
            documents_texts_embeddings = documents_texts_embeddings.cpu().numpy()

            qdrantdb_client.upload_collection(
                collection_name=collection_name,
                ids=None,
                payload=documents_metadata,
                vectors=documents_texts_embeddings
            )

        end = time.time()
        print("Finished | Time required: ", end - start)

    qdrantdb_client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )

    print(f'{collection_name} uploaded')
    print(f'Length texts and chunks: {len(uploaded_sources)} | {len_chunks}')


###

def search_kb(query, qdrantdb_client, collection_name, embedder, limit_results):
    vector = embedder.encode(query).tolist()

    search_results = qdrantdb_client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=limit_results,
    )

    search_results = [search_result.payload for search_result in search_results]
    
    return search_results