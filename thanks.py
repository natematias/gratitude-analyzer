import nltk
import json
from nltk.corpus import wordnet
import string
import pickle
import yaml
import os
import itertools

CONFIG = yaml.load(open('config.yaml'))

class Thank:
  def __init__(self,line):

    self.line = line
    self.tokens = nltk.word_tokenize(line)
    self.length = len(self.tokens)
    #self.trigrams = nltk.trigrams(self.tokens)
    # load a brill tagger trained by nltk-trainer
    # https://github.com/japerk/nltk-trainer
    tagger = pickle.load(open(os.path.join(CONFIG["tagger_folder"],"treebank_brill_aubt.pickle"), "rb"))
    self.pos_tokens = tagger.tag(self.tokens)
    # if you don't have a brill tagger, you can use this
    #self.pos_tokens = nltk.pos_tag(self.tokens)

    self.noun_blacklist = ['http']
    self.process()

  def process(self):
    self.pos = {}

    self.pos["nouns"] = [x[0] for x in self.pos_tokens if x[1] in ["NN", "NNS","NNP","NNPS"] and x[0] not in self.noun_blacklist and "//" not in x[0]]
    self.pos["pronouns"] = [x[0] for x in self.pos_tokens if x[1] in ["PRP", "PRP$"]]
    self.pos["verbs"] = [x[0] for x in self.pos_tokens if x[1] in ["VB", "VBG","VBN","VBP","VBZ"]]
    self.pos["predeterminers"] = [x[0] for x in self.pos_tokens if x[1] in ["PDT"]]
    self.pos["interjections"] = [x[0] for x in self.pos_tokens if x[1] in ["UH"]]
    self.pos["modifiers"] = [x[0] for x in self.pos_tokens if x[1] in ["JJ", "JJR","JJS","RB","RBR","RBS", "PDT"] and x[0] not in string.punctuation and "//" not in x[0] and "n't" not in x[0]]

  #possessive pronouns
  #personal pronouns
  #word tree
  # mine
  # yours
  # nouns: what is getting thanked for

class Thanks:
  def __init__(self,lines):
    self.thanks = []
    self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for line in lines.split("\n"):
      self.thanks.append(Thank(line))
      self.thanks[-1].sentences = self.sent_detector.tokenize(line)
      self.thanks[-1].num_sentences = len(self.thanks[-1].sentences)
    self.superset = self.thanks

  def pos_frequency(self, pos):
    if not pos in self.thanks[0].pos.keys():
      return []
    i = [x.pos[pos] for x in self.thanks]
    words = [x.lower() for x in list(itertools.chain(*i))] 
    return nltk.FreqDist(words).items()

  # matches against any of these words
  def word_filter(self, wordlist):
    thx = []
    for thank in self.thanks:
      match = False
      l = [x.lower() for x in thank.tokens]
      for word in wordlist:
        if(not match and word.lower() in l):
          match = True
      if(match):
        thx.append(thank)
    self.thanks = thx
    return {"superset":len(self.superset),"filter":len(self.thanks)}

  # matches messages that include both words
  def word_collocation_filter(self, wordpairs):
    thx = []
    for thank in self.thanks:
      match = False
      l = [x.lower() for x in thank.tokens]
      for wordpair in wordpairs:
        if((not match) and wordpair[0].lower() in l and wordpair[1].lower() in l):
          match = True
      if(match):
        thx.append(thank)
    self.thanks = thx
    return {"superset":len(self.superset),"filter":len(self.thanks)}

  def reset_filter(self):
    self.thanks = self.superset
    return {"superset":len(self.superset),"filter":len(self.thanks)}
