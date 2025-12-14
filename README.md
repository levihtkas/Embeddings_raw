Word Embeddings from Scratch (Skip-Gram with NumPy)

This project implements a Word2Vec Skip-Gram model from scratch using NumPy, without any deep learning frameworks.
It is designed to demystify how embeddings are learned, showing how words acquire meaning purely from context.

📌 What This Project Does

Builds a vocabulary from a small text corpus

Generates (target, context) pairs using a sliding window

Trains a Skip-Gram Word2Vec model

Learns word embeddings via:

dot products

softmax

cross-entropy loss

gradient descent

Computes cosine similarity between learned word vectors

🧩 Key Concepts Demonstrated

Tokenization and vocabulary indexing

Skip-Gram training objective

Input vs Output embedding matrices
