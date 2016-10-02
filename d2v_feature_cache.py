from __future__ import division
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
from collections import defaultdict
import string
import json
import tqdm
from scipy import spatial
import numpy as np
from gensim.models.doc2vec import *
from gensim.models import Doc2Vec
import math
from xg_lib import *
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import argparse
from shared.ordered_object import *
from shared.utilities import *
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from tqdm import tqdm


def getDoc2VecFeatures(u1, u2, mdl, v1, v2):
	'''
	Gets doc2vec Features 
	returns 
		:Cosine Similarity
		:L1 and L2 Norm Distances
		:Orders of U1 in U2 and U2 in U1
	'''
	try:
		u1_knn = mdl.docvecs.most_similar(u1, topn=100)
		u2_knn = mdl.docvecs.most_similar(u2, topn=100)
		cos = mdl.docvecs.similarity(u1, u2)
		euc = euclidean_distances([v1],[v2])[0][0]
		mhd = manhattan_distances([v1],[v2])[0][0]
		# Get vectors
		orders_1 = [str(x[0]) for x in u1_knn]
		orders_2 = [str(x[0]) for x in u2_knn]
		order_u2_in_u1, order_u1_in_u2 = -1, -1
		if(u2 in orders_1):
			order_u2_in_u1 = orders_1.index(u2)
		if(u1 in orders_2):
			order_u1_in_u2 = orders_2.index(u1)
	except:
		# pass
		return None
	return [order_u2_in_u1, order_u1_in_u2, cos, euc, mhd]

d2v_feature_dict = {}

# Candidate sets
train_candidate_sets=['candidates/candidate_pairs.baseline.nn.100.train-98k.with-orders.tf-scaled.full-hierarchy.3.json.gz']
dev_candidates_sets=['candidates/candidate_pairs.baseline.nn.100.test-98k.with-orders.tf-scaled.full-hierarchy.3.json.gz']
test_candidate_sets=['candidates/candidate_pairs.baseline.nn.100.test.with-orders.tf-scaled.full-hierarchy.3.json.gz']

# Load doc2vec ensemble!
doc2vec_model_h1 = Doc2Vec.load('models/mdl_urls_300_w5_h1.d2v')
doc2vec_model_h2 = Doc2Vec.load('models/mdl_urls_300_w10_h2.d2v')
doc2vec_model_h3 = Doc2Vec.load('models/mdl_urls_300_w10_h3.d2v')
doc2vec_model_concat = Doc2Vec.load('models/mdl_urls_300_concat.d2v')
word_model = Doc2Vec.load('models/mdl_word.d2v')

# Model list 
d2v_model_list = [doc2vec_model_h1,doc2vec_model_h2,doc2vec_model_h3,doc2vec_model_concat,word_model]

nn_pairs_lst = [filter_order_list(dictFromFileUnicode(m),5) for m in train_candidate_sets]
order_objs = [OrderClass(ps) for ps in nn_pairs_lst]

# Build the train and test data xgb1	

nn_pairs= []
for ps in nn_pairs_lst:
	nn_pairs += ps
nn_pairs = filter_nn_pairs(nn_pairs)
random.shuffle(nn_pairs)

golden_edges = set()
positive_samples = []
timer = ProgressBar(title="Reading golden edges")
with open('train_98k.csv','r') as f:
	for line in f:
		timer.tick()
		uid1, uid2 = line.strip().split(',')
		golden_edges.add((min(uid1, uid2),max(uid1, uid2)))


for d2v_model in d2v_model_list:
	try:
		v1 = doc2vec_model_h1.docvecs['USER_'+str(uid1)]
		v2 = doc2vec_model_h1.docvecs['USER_'+str(uid2)]
		_f = getDoc2VecFeatures('USER_'+str(uid1),'USER_'+str(uid2), doc2vec_model_h1, v1, v2)
	except:
		_f = [-1,-1,-1,-1,-1]