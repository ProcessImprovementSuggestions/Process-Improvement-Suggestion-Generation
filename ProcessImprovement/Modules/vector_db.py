import os
import time
import itertools
import pandas as pd
from qdrant_client import models
import resource_preprocessing


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

def create_db(path_resources, from_directory, source_column, text_column, qdrantdb_client, collection_name, embedder):
    source_alias='source'
    text_alias='page_content'

    print(f'Create {collection_name}')
    recreate_db(qdrantdb_client, collection_name, embedder)

    len_documents=0
    if from_directory == True:
        upload_batch_size = 100000
        uploaded_sources = set()

        id_generator = itertools.count()

        for resource_file in os.scandir(path_resources):
            print(f'Loading file: {resource_file.name}')
            corpus_data = pd.read_json(resource_file.path,  lines=True)
            print(f'Length corpus: {len(corpus_data)}')

            corpus_data = resource_preprocessing.preprocessing(corpus_data, source_column, text_column)

            corpus_texts_raw = corpus_data[text_column].tolist()
            corpus_sources_raw = corpus_data[source_column].tolist()

            corpus_texts=[]
            corpus_sources=[]
            #Check for duplicates
            for idx, source_i in enumerate(corpus_sources_raw):
                if source_i not in uploaded_sources:
                    uploaded_sources.add(source_i)
                    corpus_sources.append(source_i)
                    corpus_texts.append(corpus_texts_raw[idx])

            corpus_sources = [{source_alias: source_i} for source_i in corpus_sources]
            print(f'Length corpus after preprocessing: {len(corpus_texts)}')
        
            start = time.time()
            for batch_i in range(0, len(corpus_texts), upload_batch_size):
                documents = resource_preprocessing.create_split_documents(corpus_texts[batch_i:batch_i+upload_batch_size], corpus_sources[batch_i:batch_i+upload_batch_size], embedder)
                documents_texts = [document_i['page_content'] for document_i in documents]
                documents_metadata = [{source_alias: document_i['metadata'][source_alias], text_alias: document_i['page_content']} for document_i in documents]
                len_documents+=len(documents_texts)

                documents_texts_embeddings = embedder.encode(documents_texts, batch_size=256, device='cuda', convert_to_tensor=True)
                documents_texts_embeddings = documents_texts_embeddings.cpu().numpy()

                document_batch_ids = [next(id_generator) for _ in documents_texts_embeddings]

                print(f'Start_batch_id: {document_batch_ids[0]} | End_batch_id: {document_batch_ids[-1]}')

                qdrantdb_client.upload_collection(
                    collection_name=collection_name,
                    ids=document_batch_ids,
                    payload=documents_metadata,
                    vectors=documents_texts_embeddings
                )

            end = time.time()
            print("Finished | Time required: ", end - start)

        qdrantdb_client.update_collection(
            collection_name=collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
            )

    elif from_directory == False:
        print('Loading data from pkl')
        corpus_data = pd.read_pickle(path_resources)
        print(f'Length data: {len(corpus_data)}')

        corpus_data = resource_preprocessing.preprocessing(corpus_data, source_column, text_column)

        corpus_texts = corpus_data[text_column].tolist()
        corpus_sources = corpus_data[source_column].tolist()
        corpus_sources = [{source_alias: source_i} for source_i in corpus_sources]
        print(f'Length corpus after preprocessing: {len(corpus_texts)}')

        start = time.time()
        documents = resource_preprocessing.create_split_documents(corpus_texts, corpus_sources, embedder)
        documents_texts = [document_i['page_content'] for document_i in documents]
        documents_metadata = [{source_alias: document_i['metadata'][source_alias], text_alias: document_i['page_content']} for document_i in documents]
        len_documents=len(documents_texts)

        documents_texts_embeddings = embedder.encode(documents_texts, batch_size=256, show_progress_bar=True, device='cuda', convert_to_tensor=True)
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

    print('Collection uploaded')
    print(f'Length documents: {len_documents}')


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