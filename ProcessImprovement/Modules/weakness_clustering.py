from sentence_transformers import util
import pandas as pd

def get_clusters(weakness_batch, embedder, cluster_min_size = 1, cluster_threshold=0.75):
    
    weakness_batch['cluster'] = [-1 for _ in range(weakness_batch.shape[0])]
    weakness_cluster_batch = weakness_batch.values.tolist()
    corpus_weaknesses = weakness_batch["weakness"].tolist()

    #Create embeddings
    corpus_weakness_embeddings = embedder.encode(corpus_weaknesses, show_progress_bar=True, convert_to_tensor=True)

    #Flat clustering
    clusters = util.community_detection(corpus_weakness_embeddings, min_community_size=cluster_min_size, threshold=cluster_threshold)


    for idx, cluster_i in enumerate(clusters):
        for weakness_id in cluster_i:
            weakness_cluster_batch[weakness_id][2] = idx


    weakness_cluster_batch = pd.DataFrame(weakness_cluster_batch, columns=['tweetid', 'weakness', 'cluster'])

    return weakness_cluster_batch