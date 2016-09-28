#!/usr/bin/env python
'''
Doc2Vec baseline for CIKM Cup
Trains a Doc2Vec model
'''
import numpy as np
import random
import sys
import logging
import string 
import os
from collections import Counter,OrderedDict
import json
from tqdm import tqdm
import math
import gensim
import multiprocessing
import cPickle
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from environment import Environment

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

# Setup logger
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':
    ENV = Environment()
    model = Doc2Vec(size=300, window=10, \
                    min_count=5, workers=8,\
                    alpha=0.025, \
                    min_alpha=0.025) # use fixed learning rate
    model.build_vocab(ENV)
    for epoch in range(5):
        model.train(ENV)
        model.alpha -= 0.002 # decrease the learning rate
        model.min_alpha = model.alpha # fix the learning rate, no deca
        model.train(ENV)
    model.save('Models/mdl_urls_300_w10_h3.d2v')


