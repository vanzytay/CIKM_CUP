#!/usr/bin/env python
# Just for local testing 

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

def getDoc2VecFeatures(u1, u2, mdl, v1, v2):
	try:
		from sklearn.metrics.pairwise import euclidean_distances
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
		return [-1,-1,-1,-1,-1]
	return [order_u2_in_u1, order_u1_in_u2, cos, euc, mhd]

golden_edges = set()

with open('train_100k.csv','r') as f:
	for index,line in enumerate(f):
		if(index>20):
			break
		uid1, uid2 = line.strip().split(',')
		golden_edges.add((min(uid1, uid2),max(uid1, uid2)))

doc2vec_model_h1 = Doc2Vec.load('models/mdl_urls_300_w5_h1.d2v')

for g in golden_edges:
	uid1 = g[0]
	uid2 = g[1]
	# print(uid1)
	# print(uid2)
	
	# TY: My word_level model uses USER_ + ID as label id
	v1 = doc2vec_model_h1.docvecs['USER_'+str(uid1)]
	v2 = doc2vec_model_h1.docvecs['USER_'+str(uid2)]
	d2v_features = getDoc2VecFeatures('USER_'+str(uid1),'USER_'+str(uid2), doc2vec_model_h1, v1, v2)
	print(d2v_features)
	# except:
	# 	cos = -1
	# 	pass