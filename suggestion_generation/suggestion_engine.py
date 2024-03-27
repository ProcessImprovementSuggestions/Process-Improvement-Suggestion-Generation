import json
import numpy as np
import os
import pandas as pd
from config import config_params
from generation_templates import answer_generation_template
from generation_templates import query_generation_template
from generation_templates import suggestion_identification_template
from generation_templates import weakness_identification_template
from googleapiclient.discovery import build
from itertools import chain
from openai import OpenAI
from processing_utils import resource_preprocessing
from processing_utils import retrieval_processing
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sentence_transformers.cross_encoder import CrossEncoder



class SuggestionEngine:
    """Identifies process weaknesses described in texts (e.g., tweets) and generates improvement suggestions using knowledge resources."""

    def __init__(self):
        self.search_embedder = SentenceTransformer(config_params["search_embedding_model"], device="cuda")
        self.cluster_embedder = SentenceTransformer(config_params["cluster_embedding_model"], device="cuda")
        self.cross_encoder = CrossEncoder(f"cross-encoder/{config_params['cross_encoder_model']}", device="cuda")

        self.GOOGLE_CSE_ID = config_params["GOOGLE_CSE_ID"]
        self.GOOGLE_API_KEY = config_params["GOOGLE_API_KEY"]
        self.websearch_service = build("customsearch", "v1", developerKey=self.GOOGLE_API_KEY)

        self.generative_model = config_params["generative_model"]
        os.environ["OPENAI_API_KEY"] = config_params["openai_api_key"]
        self.openAI_client = OpenAI()

        self.grobid_url_setting = '%s/api/processFulltextDocument' % config_params["GROBID_URL"]

        self.scholar_x_api_key = config_params["scholar_x_api_key"]
        self.qdrantdb_client = QdrantClient(host=config_params["qdrant_host"], grpc_port=config_params["qdrant_grpc_port"], prefer_grpc=True)

        self.abstract_collection_name = config_params["abstract_collection"]
        self.tweet_collection_name = config_params["tweet_collection"]


    def load_feedback(self, feedback, source_column, text_column, cross_dataset_preprocess):
        self.feedback = pd.DataFrame({'feedback_id': feedback[source_column].to_list(), 'feedback_text': feedback[text_column].to_list()})
        if cross_dataset_preprocess == True:
            self.feedback = resource_preprocessing.cross_dataset_preprocessing(self.feedback, 'feedback_text', 'feedback_id')


    def _zero_shot_response(self, user_prompt, system_prompt):
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        response = self.openAI_client.chat.completions.create(
            model=self.generative_model,
            response_format={ "type": "json_object" },
            messages=messages,
            temperature=0,
        )
        return response
    

    def _few_shot_response(self, user_prompt, few_shot_prompts):
        messages = list(few_shot_prompts)
        messages.append({"role": "user", "content": user_prompt})
        
        response = self.openAI_client.chat.completions.create(
            model=self.generative_model,
            response_format={ "type": "json_object" },
            messages=messages,
            temperature=0,
        )
        return response


    def _format_context(self, context):
        context = "\n\n".join(context)
        return context


    def weaknesses_identification(self):
        """Identifies the process weaknesses described in the provided feedback."""

        corpus_feedback_batch = self.feedback.values.tolist()
        user_prompt_template = weakness_identification_template.get_user_prompt_template()
        user_prompts = [user_prompt_template.format(prompt_feedback = feedback_i[1]) for feedback_i in corpus_feedback_batch]
        system_few_shot_prompts = weakness_identification_template.get_system_few_shot_prompts()

        weaknesses_batch = []
        for idx, user_prompt_i in enumerate(user_prompts):
            try:
                response = self._few_shot_response(user_prompt_i, system_few_shot_prompts)
                response = json.loads(response.choices[0].message.content)
                process_weaknesses = response['process_weaknesses']
                for process_weakness_i in process_weaknesses:
                    weaknesses_batch.append([corpus_feedback_batch[idx][0], process_weakness_i])
                corpus_feedback_batch[idx].append(process_weaknesses)
            except:
                corpus_feedback_batch[idx].append([])

        self.feedback_weakness_batch = pd.DataFrame(corpus_feedback_batch, columns=['feedback_id', 'feedback_text', 'weaknesses'])
        self.weakness_cluster_batch = pd.DataFrame(weaknesses_batch, columns=['feedback_id', 'weakness'])
        return self.feedback_weakness_batch, self.weakness_cluster_batch


    def weaknesses_clustering(self, cluster_min_size = 1, cluster_threshold=0.75):
        """Clusters the identified process weaknesses."""

        self.weakness_cluster_batch['cluster'] = [-1 for _ in range(self.weakness_cluster_batch.shape[0])]
        corpus_weaknesses = self.weakness_cluster_batch["weakness"].tolist()
        self.weakness_cluster_batch = self.weakness_cluster_batch.values.tolist()

        corpus_weakness_embeddings = self.cluster_embedder.encode(corpus_weaknesses, convert_to_tensor=True)
        clusters = util.community_detection(corpus_weakness_embeddings, min_community_size=cluster_min_size, threshold=cluster_threshold)

        for idx, cluster_i in enumerate(clusters):
            for weakness_id in cluster_i:
                self.weakness_cluster_batch[weakness_id][2] = idx

        self.weakness_cluster_batch = pd.DataFrame(self.weakness_cluster_batch, columns=['feedback_id', 'weakness', 'cluster'])
        return self.weakness_cluster_batch
    

    def cluster_query_generation(self, cluster_max_examples = 10):
        """Generates, for each cluster, a search query aimed at finding improvement suggestions"""

        system_prompt = query_generation_template.get_system_prompt()
        user_prompt_template = query_generation_template.get_user_prompt_template()

        clusters = set(self.weakness_cluster_batch['cluster'].to_list())
        clusters = sorted(list(clusters))

        #Generate search queries
        self.cluster_queries_batch = []

        for cluster_i in clusters:
            if cluster_i != -1:
                context = self._format_context(self.weakness_cluster_batch[self.weakness_cluster_batch['cluster']==cluster_i]['weakness'].to_list()[:cluster_max_examples])
                response = self._zero_shot_response(user_prompt_template.format(texts=context), system_prompt)
                try:
                    self.cluster_queries_batch.append([cluster_i, json.loads(response.choices[0].message.content)['search_query']])
                except:
                    self.cluster_queries_batch.append([cluster_i, ''])
            else:
                self.cluster_queries_batch.append([cluster_i, ''])

        self.cluster_queries_batch = pd.DataFrame(self.cluster_queries_batch, columns=['cluster', 'search_query'])
        return self.cluster_queries_batch
    
    
    def _retrieve(self, query, limit_results_retrieve):
        tweet_search_results = retrieval_processing.get_tweet_documents(query, self.qdrantdb_client, self.tweet_collection_name, self.search_embedder, limit_results_retrieve)
        paper_search_results = retrieval_processing.get_paper_documents(query, self.qdrantdb_client, self.abstract_collection_name, self.search_embedder, self.grobid_url_setting, self.scholar_x_api_key, limit_results_retrieve)
        web_search_results = retrieval_processing.get_web_documents(query, self.websearch_service, self.GOOGLE_CSE_ID, self.search_embedder, limit_results_retrieve)
        
        query_results = [tweet_search_results, paper_search_results, web_search_results]
        query_results = list(chain(*query_results))
        return query_results


    def _rerank(self, query, query_results, limit_results_rerank):
        corpus = [query_result[0] for query_result in query_results]
        query_corpus_combinations = [[query, document] for document in corpus]

        #print('Reranking')
        # Compute the similarity scores for these combinations
        similarity_scores = self.cross_encoder.predict(query_corpus_combinations)

        # Sort the scores in decreasing order
        sim_scores_argsort = reversed(np.argsort(similarity_scores))

        reranked_results = []
        for idx in sim_scores_argsort:
            reranked_results.append(query_results[idx])
        return reranked_results[0:limit_results_rerank]


    def _suggestions_identification(self, query, reranked_query_results):
        system_prompt = suggestion_identification_template.get_system_prompt()
        user_prompt_template = suggestion_identification_template.get_user_prompt_template()

        context = [reranked_query_result[0] for reranked_query_result in reranked_query_results]

        #Generate suggestions
        context = self._format_context(context)
        response = self._zero_shot_response(user_prompt_template.format(context=context, query=query), system_prompt)
        try:
            improvement_suggestions = json.loads(response.choices[0].message.content)['improvement_suggestion']
        except:
            improvement_suggestions = 'N/A'
        return improvement_suggestions


    def _suggestions_postprocessing(self):
        self.weakness_cluster_batch = self.weakness_cluster_batch.values.tolist()
        self.cluster_queries_batch = self.cluster_queries_batch.values.tolist()

        for idx, weakness_cluster_i in enumerate(self.weakness_cluster_batch):
            self.weakness_cluster_batch[idx].append([])
            for cluster_query_i in self.cluster_queries_batch:
                if weakness_cluster_i[2] == cluster_query_i[0]:
                    self.weakness_cluster_batch[idx][3] = cluster_query_i[2]

        self.cluster_queries_batch = pd.DataFrame(self.cluster_queries_batch, columns=['cluster', 'search_query', 'suggestions', 'reranked'])

        self.feedback_weakness_batch = self.feedback_weakness_batch.values.tolist()
        for idx, feedback_weakness_i in enumerate(self.feedback_weakness_batch):
            self.feedback_weakness_batch[idx].append([])
            suggestions_i = []
            for weakness_cluster_i in self.weakness_cluster_batch:
                if feedback_weakness_i[0] == weakness_cluster_i[0]:
                    suggestions_i.append(weakness_cluster_i[3])
            self.feedback_weakness_batch[idx][3] = list(set(suggestions_i))

        self.feedback_weakness_batch = pd.DataFrame(self.feedback_weakness_batch, columns=['feedback_id', 'feedback_text', 'weaknesses', 'suggestions'])
        self.weakness_cluster_batch = pd.DataFrame(self.weakness_cluster_batch, columns=['feedback_id', 'weakness', 'cluster', 'suggestions'])


    def cluster_suggestion_generation(self, limit_results_retrieve, limit_results_rerank):
        """Generates improvement suggestions for each cluster using knowledge resources"""
        
        reranked_query_results = []
        improvement_suggestions = []

        for idx, query in enumerate(self.cluster_queries_batch['search_query'].to_list()):
            query_results = self._retrieve(query, limit_results_retrieve)
            reranked_query_results.append(self._rerank(query, query_results, limit_results_rerank))
            improvement_suggestions.append(self._suggestions_identification(query, reranked_query_results[idx]))

        self.cluster_queries_batch['suggestions'] = improvement_suggestions
        self.cluster_queries_batch['reranked'] = reranked_query_results
        self._suggestions_postprocessing()
        return self.cluster_queries_batch, self.weakness_cluster_batch, self.feedback_weakness_batch


    def feedback_answer_generation(self):
        """Generates, for each feedback, an answer that merges the suggestions relevant for the feedback"""
        
        system_prompt = answer_generation_template.get_system_prompt()
        user_prompt_template = answer_generation_template.get_user_prompt_template()

        feedback_texts = self.feedback_weakness_batch["feedback_text"].to_list()
        suggestions = self.feedback_weakness_batch["suggestions"].to_list()
        improvement_suggestions_texts = []

        for idx, feedback_i in enumerate(feedback_texts):
            if suggestions[idx] != []:
                context = self._format_context(suggestions[idx])

                #Generate answers
                response = self._zero_shot_response(user_prompt_template.format(context=context, prompt_feedback=feedback_i), system_prompt)
                try:
                    improvement_suggestions_texts.append(json.loads(response.choices[0].message.content)['improvement_suggestions_text'])
                except:
                    improvement_suggestions_texts.append('N/A')
            else:
                improvement_suggestions_texts.append('N/A')

        self.feedback_weakness_batch["answer"] = improvement_suggestions_texts
        
        return self.feedback_weakness_batch
