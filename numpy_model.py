import numpy as np 

#Corpus
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the mouse",
    "the dog chased the cat"
]

#tokenize 

tokens =[]

for sentence in corpus:
  for line in sentence.split(","):
    tokens.extend(line.split())
# print(tokens)

# unique words
vocab = sorted(set(tokens))
vocab_size = len(vocab)

# Word to index and Index to word mapping
word2idx = {word:i for i, word in enumerate(vocab)}
idx2word = {i:word for i, word in enumerate(vocab)}

print(vocab_size)

#Skip Gram
windows_size=1
training_pairs = []

for sentence in corpus:
  words = sentence.lower().split()
  for i,word in enumerate(words):
    target = word2idx[word]
    for j in range(i-windows_size,i+windows_size+1):
      if j!=i and 0<=j<len(words):
        context = word2idx[words[j]]
        training_pairs.append((target,context))

print(training_pairs[:5])

embedding_dim =100
learning_rate = 0.05

#input embeddings
w_in = np.random.randn(vocab_size,embedding_dim)*0.01
# output embeddings
w_out = np.random.randn(embedding_dim,vocab_size)*0.01

def softmax(x):
  e_x = np.exp(x-np.max(x)) # without this the exponenetional can giv us big results
  return e_x/e_x.sum()

epochs =1000

for epoch in range(epochs):
  loss=0
  for (target,context) in training_pairs:
    #one-hot encode
    x= np.zeros(vocab_size)
    x[target]=1

    #forwad pass
    h=w_in[target] # target vector embeddding projection
    u = np.dot(h,w_out) # embedding index of the context window for the target word
    y_pred = softmax(u) # scores for each word in vocab_size


    loss += -np.log(y_pred[context])

    e= y_pred.copy()

    e[context]-=1 

    #gradients
    # dW_out = np.dot(h.T,e) # e_dim x 1 1x vocab_size
    dW_out = np.outer(h, e) 


    dW_in = np.dot(e,w_out.T) #1x vocab  vocab x e_dim

    w_out = w_out - (learning_rate * dW_out)

    w_in[target] = w_in[target] -learning_rate*dW_in
  if epoch % 200 ==0:
    print(f"Epoch {epoch} ,Loss {loss:.4f}") 


def get_embedding(word):
  return w_in[word2idx[word]]

def cosine_similarity(a,b):
  return (np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b)))

def most_similar(word, top_k=3):
    vec = get_embedding(word)
    scores = []

    for w in vocab:
        if w != word:
            sim = cosine_similarity(vec, get_embedding(w))
            scores.append((w, sim))

    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
 
print(most_similar("cat"))

print(get_embedding("cat"))


