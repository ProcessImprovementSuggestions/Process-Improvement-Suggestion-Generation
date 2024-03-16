import pandas as pd

def postprocessing(weakness_cluster_batch, cluster_queries_batch, tweet_weakness_batch):

    weakness_cluster_batch = weakness_cluster_batch.values.tolist()
    cluster_queries_batch = cluster_queries_batch.values.tolist()

    for idx, weakness_cluster_i in enumerate(weakness_cluster_batch):
        weakness_cluster_batch[idx].append([])
        for cluster_query_i in cluster_queries_batch:
            if weakness_cluster_i[2] == cluster_query_i[0]:
                weakness_cluster_batch[idx][3] = cluster_query_i[2]

    cluster_queries_batch = pd.DataFrame(cluster_queries_batch, columns=['cluster', 'search_query', 'suggestions', 'reranked'])

    tweet_weakness_batch = tweet_weakness_batch.values.tolist()

    for idx, tweet_weakness_i in enumerate(tweet_weakness_batch):
        tweet_weakness_batch[idx].append([])
        suggestions_i = []
        for weakness_cluster_i in weakness_cluster_batch:
            if tweet_weakness_i[0] == weakness_cluster_i[0]:
                suggestions_i.append(weakness_cluster_i[3])
        tweet_weakness_batch[idx][3] = list(set(suggestions_i))

    tweet_weakness_batch = pd.DataFrame(tweet_weakness_batch, columns=['tweetid', 'tweet', 'weaknesses', 'suggestions'])
    weakness_cluster_batch = pd.DataFrame(weakness_cluster_batch, columns=['tweetid', 'weakness', 'cluster', 'suggestions'])

    return weakness_cluster_batch, cluster_queries_batch, tweet_weakness_batch