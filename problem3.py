import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch

np.random.seed(42)
torch.manual_seed(42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

with open('problem3_data.txt', 'r') as file:
    passages = file.readlines()

def get_charge_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_7_output = outputs.hidden_states[7][0]
    
    charge_positions = [i for i, id in enumerate(inputs['input_ids'][0]) if tokenizer.decode([id]) == "charge"]
    
    if not charge_positions:
        return None
    
    charge_embedding = layer_7_output[charge_positions[0]].numpy()
    
    return charge_embedding

embeddings = []
valid_passages = []

for passage in passages:
    embedding = get_charge_embedding(passage)
    if embedding is not None:
        embeddings.append(embedding)
        valid_passages.append(passage)

embeddings = np.array(embeddings)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(embeddings)

def pca_visualize_embeddings(embeddings, passages, clusters):
    pca = PCA(n_components=2)
    twodim = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 12))

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i in range(5):
        cluster_points = twodim[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i}', edgecolors='k')
    
    plt.legend()
    
    sample_size = min(50, len(passages))
    sample_indices = np.random.choice(len(passages), sample_size, replace=False)
    
    for idx in sample_indices:
        x, y = twodim[idx]
        text = passages[idx].strip()[:20] + "..."
        plt.annotate(text, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    plt.title("PCA Visualization of 'charge' Token Embeddings")
    
    plt.show()

pca_visualize_embeddings(embeddings, valid_passages, clusters)

# Print some examples from each cluster
for cluster in range(5):
    print(f"\nCluster {cluster} examples:")
    cluster_passages = [passage for passage, label in zip(valid_passages, clusters) if label == cluster]
    for passage in cluster_passages[:3]: 
        print(passage.strip())
        print("---")

cluster_df = pd.DataFrame({
    'passage': valid_passages,
    'cluster': clusters
})
cluster_df.to_csv('charge_clusters.csv', index=False)


# Problem 3B
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

def get_replacements(sentence, target_word, top_k=20):
    masked_sentence = sentence.replace(target_word, '[MASK]')
    inputs = tokenizer(masked_sentence, return_tensors='pt')
    mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    mask_token_logits = logits[0, mask_token_index, :]
    top_k_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()
    
    replacements = [tokenizer.decode([token]) for token in top_k_tokens]
    return replacements

with open('problem3_data.txt', 'r') as file:
    passages = file.readlines()

all_replacements = []
valid_passages = []
for passage in passages:
    if 'charge' in passage.lower():
        replacements = get_replacements(passage, 'charge')
        if ('charge' in replacements): 
            replacements.remove('charge')
        all_replacements.append(replacements)
        valid_passages.append(passage)

unique_replacements = list(set([word for sublist in all_replacements for word in sublist]))

vectors = []
for replacements in all_replacements:
    vector = [1 if word in replacements else 0 for word in unique_replacements]
    vectors.append(vector)

embeddings = np.array(vectors)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Print some examples from each cluster
for cluster in range(5):
    print(f"\nCluster {cluster} examples:")
    cluster_passages = [passage for passage, label in zip(valid_passages, clusters) if label == cluster]
    for passage in cluster_passages[:3]:
        print(passage.strip())
        print("Replacements:", ', '.join(all_replacements[valid_passages.index(passage)]))
        print("---")

cluster_df = pd.DataFrame({
    'passage': valid_passages,
    'cluster': clusters,
    'replacements': [', '.join(rep) for rep in all_replacements]
})
cluster_df.to_csv('charge_replacement_clusters.csv', index=False)