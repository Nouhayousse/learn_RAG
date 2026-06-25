from  sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

texts = ["Intelligence artificielle", "Machine learning", "Deep learning", "Réseaux de neurones",
         "Ecommerce", "Marketing digital", "Optimisation des moteurs de recherche", "Publicité en ligne",
         "fashion", "Mode", "Vêtements", "Accessoires", "Chaussures", "Bijoux",
         "Santé", "Médecine", "Bien-être", "Nutrition", "Fitness", "Yoga"]


embeddings = model.encode(texts)
print("Embeddings shape:", embeddings.shape)


