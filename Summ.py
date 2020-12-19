### Produced as a school project for CS 490A: Applications of NLP
# Fall 2020
# At the University of Massachusetts Amherst
# Group members: Jeremy Doiron, Akash Munjial, Bhavya Pant


import re
import nltk
import math
import numpy
from gensim.models import Word2Vec
import networkx as nx
from rouge import Rouge


############### Tokenization ###############
# class defined to store sentences, containing the original string (to create the transcription) and a tokenization (list of token strings)
class Sentence:
  def __init__(self, string, tokens):
    self.original = string
    self.tokenization = tokens

# tokenizes an input string into Sentence objects
def tokenize(input_string):
  input_string = re.sub(r'(\b\w)(\.)(\w\b)(\.)',r'\1\3',input_string)
  tokens = nltk.word_tokenize(input_string)
  t = ''
  ret_sentences = []
  ret_tokens = []
  for token in tokens:
    if token == 'n\'t':
      ret_tokens[-1] += token
      t = t[:-1]+token+' '
    elif token in ["'", "`", "''", "``", "”","“"]:
        continue
    elif len(token) > 1:
      ret_tokens.append(token.lower())
      t += token + ' '
    elif len(token) == 1 and token.isalpha():
      ret_tokens.append(token.lower())
      t += token + ' '
    else:
      t = t[:-1]+token
      if token == '.':
        ret_sentences.append(Sentence(t, ret_tokens))
        t = ''
        ret_tokens = []
      else:
        t += ' '
  return ret_sentences


############### Segmentation ###############
def segment(book, num_segments):
  numSents = len(book)
  sents_per_seg = math.floor(numSents/num_segments)
  return [book[i:i+(sents_per_seg)] for i in range(0, numSents, sents_per_seg)]


############### Vectorize ###############
# Vecotrization functions should follow this architecture:
# arg: sents, a list of sentences (represented as lists of tokens)
# return: matrix of sentence similarities

# compute similarities based on the number of shared tokens
def bow_vectorize(sents):
  # create vocabulary for this segment
  v = []
  for sent in sents:
    for tok in sent.tokenization:
      if tok not in v:
        v.append(tok)
  # represent vocabulary as a dict mapping tokens to indices
  v = {v[i]: i for i in range(len(v))}
  vectors = []
  for i in range(len(sents)):
    temp = [0] * len(v)
    for tok in sents[i].tokenization:
      temp[v[tok]] += 1
    vectors.append(temp)
  # dists: matrix of cosine similarities
  return numpy.array([[0 if i == j else numpy.dot(vectors[i], vectors[j]) / (numpy.linalg.norm(vectors[i]) * numpy.linalg.norm(vectors[j])) for j in range(len(vectors))] for i in range(len(vectors))])

# using generic embeddings (embedding of a sentence as the average of its tokens)
# embeddings: dict mapping strings to vectors
def embedding_vectorize(sents, embeddings):
    vectors = []
    for sent in sents:
        temp = [0]*len(embeddings['a'])
		num_toks = 0
        for tok in sent.tokenization:
            if tok in embeddings:
                temp = numpy.add(temp, embeddings[tok])
				num_toks += 1
        vectors.append([val / num_toks for val in temp])
    return numpy.array([[0 if i == j else numpy.dot(vectors[i], vectors[j]) / (numpy.linalg.norm(vectors[i]) * numpy.linalg.norm(vectors[j])) for j in range(len(vectors))] for i in range(len(vectors))])

# Using GLOVE or pretrained embeddings
# Load word embeddings: from pretrained .txt embeddings
# returns a dict mapping strings to embedding vectors
def load_embeddings(filename):
  embeddings = {}
    for line in open(filename, encoding = 'utf8').readlines():
      entry = line.split()
      word = entry[0]
      vec = numpy.array([float(val) for val in entry[1:]])
      embeddings[word] = vec
  return embeddings

# Create tailored embeddings: given a book (array of setences(tokenized)), return the word embeddings
def embed(filename, sents):
  sents = [sent.tokenization for sent in sents]
  return Word2Vec(sents, min_count = 1, workers = 3)


############### Pagerank ###############
# use nx.pagerank algorithm to rank sentences
def rank(similarity_matrix):
  sim_graph = nx.convert_matrix.from_numpy_array(similarity_matrix)
  #The result of pagerank() should be a dict mapping indices to rank values
  return nx.pagerank(sim_graph, tol = .01, max_iter = 1000)

# take in the output ranks for a segment and a number of sentences, return the top n sentences
def topN(ranks, sents, n):
  indices = [key for key, value in sorted(ranks.items(), key = lambda item: item[1], reverse = True)][:n]
  s = ''
  for i in indices:
    s += sents[i].original+' '
  return s


############### Evaluation ###############
def eval_summary(human_summary, machine_summary):
  rouge = Rouge()
  human_summary = open(human_summary, 'r', encoding = 'utf8').read() #I assume ROUGE wants untokenized strings as input?
  machine_summary = open(machine_summary, 'r').read()
  score = score = rouge.get_scores(machine_summary, human_summary) #will return average score over all lines in the file
  return score


############### Testing ###############
def test_model(book_file, sum_file, num_segments, GEN_SUM_DESTINATION = 'outfile.txt', SENTS_PER_SEGMENT = 2):
  book = open(book_file, 'r', encoding = 'utf8').read()
  print('Now summarizing', book_file)
  sents = [sent for sent in tokenize(book)]
  segments = segment(sents, num_segments)
  embeddings = embed('', sents)
  insegment_similarities = [embedding_vectorize(segment, embeddings) for segment in segments]
  rankings = []
  for matrix in insegment_similarities:
    try:
      rankings.append(rank(matrix))
    except:
      print('Error: a segment did not converge. Resuming with other segments...')
      rankings.append([])
  auto_sum = ''
  for i in range(len(segments)):
    if rankings[i] == []:
      continue
    auto_sum += topN(rankings[i], segments[i], SENTS_PER_SEGMENT)

  out = open(GEN_SUM_DESTINATION, 'w')
  out.write(auto_sum)
  out.close()

  print('Evaluating the current model for the book', book_file)
  print(eval_summary(sum_file, GEN_SUM_DESTINATION))


books = [('peterpan.txt', 'peterpan_summary.txt'), ('baskervilles.txt', 'baskervilles_summary.txt'), ('pride_and_prejudice.txt', 'pride_and_prejudice_summary.txt'), ('thesecretgarden.txt', 'thesecretgarden_summary.txt'), ('taleoftwocities.txt', 'taleoftwocities_summary.txt')]
for i in [15]:
  print("Now analyzing Word2Vec model with number of segments:", i)
  for book, summary in books:
    test_model(book, summary, i, SENTS_PER_SEGMENT = 2)

