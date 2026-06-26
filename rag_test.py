from  sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

texts = ["Intelligence artificielle", "Machine learning", "Deep learning", "Réseaux de neurones",
         "Ecommerce", "Marketing digital", "Optimisation des moteurs de recherche", "Publicité en ligne",
         "fashion", "Mode", "Vêtements", "Accessoires", "Chaussures", "Bijoux",
         "Santé", "Médecine", "Bien-être", "Nutrition", "Fitness", "Yoga"]


embeddings = model.encode(texts)
print("Embeddings shape:", embeddings.shape)


def compute_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


i,j,k,l,n=0,1,2,3,4
    

print(f"Similarity between '{texts[i]}' and '{texts[j]}': {compute_similarity(embeddings[i], embeddings[j])}")
print(f"Similarity between '{texts[j]}' and '{texts[k]}': {compute_similarity(embeddings[j], embeddings[k])}")
print(f"Similarity between '{texts[k]}' and '{texts[l]}': {compute_similarity(embeddings[k], embeddings[l])}")
print(f"Similarity between '{texts[l]}' and '{texts[n]}': {compute_similarity(embeddings[l], embeddings[n])}")




textes = [
    "Ce séminaire porte sur le deep learning et machine learning.",
    "Cette conférence explore les architectures de réseaux de neurones artificiels.",
    "Un atelier dédié au e-commerce et aux stratégies de vente en ligne.",
]

embedding = model.encode(textes)


print("Deep learning vs Réseaux de neurones (en phrases):", compute_similarity(embedding[0], embedding[1]))
print("Deep learning vs Ecommerce (en phrases):", compute_similarity(embedding[0], embedding[2]))

