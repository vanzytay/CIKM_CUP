from collections import defaultdict
from myScripts import *
import string
import json
import tqdm
from scipy import spatial
import numpy as np
from gensim.models.doc2vec import *
from gensim.models import Doc2Vec
import math
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

'''
Run MLP version of train/test
'''

# Init Seed for Reproducibility
seed = 7
np.random.seed(seed)

def normalize_click(a):
	return np.true_divide(a,np.sum(a)) if np.sum(a)!=0 else 0


user_features = dictFromFileUnicodeNormal('user_features.json.gz')
all_users = set(user_features.keys())
timer = ProgressBar(title="Normalizing click")
for uid in all_users:
	timer.tick()
	# print(user_features[uid])
	user_features[uid]['click_count_day_time'] = np.array(user_features[uid]['click_count_day_time'])
	user_features[uid]['click_count_time'] = np.array(user_features[uid]['click_count_time'])
	user_features[uid]['click_count_day_time_normalized'] = normalize_click(user_features[uid]['click_count_day_time'])
	user_features[uid]['click_count_time_normalized'] = normalize_click(user_features[uid]['click_count_time'])


def click_distribution_similarity(dt1, dt2):
	return np.linalg.norm(dt1-dt2)

MAX_SIM = 1000000000
def click_distribution_similarity_KL(dt1, dt2):
	'''
	Kullback-Leibler divergence
	'''
	r = np.true_divide(dt1, dt2)
	for i,e in enumerate(r):
		if e==np.inf:
			r[i] = 20
		elif e==np.nan:
			r[i] = 0
	r = np.multiply(r,dt1)
	sum = np.sum(r)
	return min(MAX_SIM,1/sum) if sum!=0 else MAX_SIM 


doc2vec_model = Doc2Vec.load('user-url.d.400.w.8.minf.1.filtered-urls.5trains.doc2vec')

def get_time_overalap(u1,u2,interval, thrs):
	t1 = [e[1] for e in user_facts[u1]]
	t2 = [e[1] for e in user_facts[u2]]
	mint = min(min(t1),min(t2))
	t1 = [e - mint for e in t1]
	t2 = [e - mint for e in t2]
	bin1 = defaultdict(int)
	bin2 = defaultdict(int)
	for e in t1:
		bin1[int(e/interval)]+=1
	for e in t2:
		bin2[int(e/interval)]+=1

	ovl_k = set(bin1.keys())&set(bin2.keys())
	c = defaultdict(int)
	for k in ovl_k:
		for thr in thrs:
			if bin1[k]>=thr and bin2[k]>=thr:
				c[thr]+=1
	return [c[thr] for thr in thrs]


def extract_feature_for_pair_users(uid1, uid2):
	# if random.randint(1, 2)==2:
	# 	tmp = uid1
	# 	uid1 = uid2
	# 	uid2 = uid1

	global user_features
	features = []
	f1 = user_features[uid1]
	f2 = user_features[uid2]
	
	features+=f1['click_count_day_time'].tolist()
	features+=f2['click_count_day_time'].tolist()
	features+=f1['click_count_time'].tolist()
	features+=f2['click_count_time'].tolist()

	features+=np.absolute((f1['click_count_day_time'] - f2['click_count_day_time'])).tolist()
	features+=np.absolute((f1['click_count_time'] - f2['click_count_time'])).tolist()

	#not really help, reduce recall abit
	features+=f1['click_count_time_normalized'].tolist()
	features+=f2['click_count_time_normalized'].tolist()

	#not really help, reduce recall abit
	features+=f1['click_count_day_time_normalized'].tolist()
	features+=f2['click_count_day_time_normalized'].tolist()
	
	# remove=> increase P but reduce R abit, f1 increaes abit
	# features.append(click_distribution_similarity(cc_dt_n1, cc_dt_n2))
	# features.append(click_distribution_similarity(cc_t_n1, cc_t_n2))

	# not really important
	# remove=> increase P but reduce R abit, f1 increaes abit
	# cc_t_n1 = f1['click_count_time_normalized']
	# cc_t_n2 = f2['click_count_time_normalized']
	# cc_dt_n1 = f1['click_count_day_time_normalized']
	# cc_dt_n2 = f2['click_count_day_time_normalized']
	# features.append(click_distribution_similarity_KL(cc_dt_n1, cc_dt_n2))
	# features.append(click_distribution_similarity_KL(cc_t_n1, cc_t_n2))

	for obj in order_objs:
		features+=obj.get_orders_infor(uid1,uid2)

	try:
		v1 = doc2vec_model.docvecs[uid1]
		v2 = doc2vec_model.docvecs[uid2]
		cos = 1 - spatial.distance.cosine(v1, v2)
	except:
		cos = -1
		pass
	# Try adding entire vectors
	# features+=v1.tolist()
	# features+=v2.tolist()
	features.append(cos)

	# features+=get_time_overalap(uid1,uid2,60,[1,5,10,30])
	return features

class OrderClass:
	def __init__(self, nn_pairs):
		self.orders = {}
		self.scores = {}
		self.extract_orders(nn_pairs)
		self.extract_scores(nn_pairs)

	def extract_orders(self, nn_pairs):
		self.orders = {}
		orders_keys = set()
		for p in nn_pairs:
			if p[0] not in orders_keys:
				orders_keys.add(p[0])
				self.orders[p[0]] = {}
			self.orders[p[0]][p[1]] = p[3]

	def extract_scores(self, nn_pairs):
		ps = set()
		score_keys = set()
		for p in nn_pairs:
			u,v = min(p[0],p[1]), max(p[0],p[1])
			if u not in score_keys:
				self.scores[u] =  {}
				score_keys.add(u)
			self.scores[u][v] =  p[2]


	def get_orders_infor(self,u,v):
		try:
			if u in self.orders[v].keys():
				order2 = self.orders[u][v]
			else:
				order2 = 101
		except:
			order2 = 101

		try:
			if v in self.orders[u].keys():
				order1 = self.orders[u][v]
			else:
				order1 = 101
		except:
			order1 = 101
		
		uu,vv = min(u,v), max(u,v)
		
		try:
			score = self.scores[uu][vv]
		except:
			score = 3

		return [order1,order2, score]#, int (order1!=101 and order2!=101), int (score>0), int (order1<=15 and order2<=15)]


def filter_nn_pairs(nn_pairs):
	'''
	Remove duplication of pair u,v and v,u
	'''
	candidates = []
	s = set()
	for p in nn_pairs:
		uid1, uid2 = min(p[0],p[1]), max(p[0],p[1])
		if (uid1,uid2) not in s:
			candidates.append((min(p[0],p[1]), max(p[0],p[1])))
			s.add((uid1, uid2))
	return candidates

def filter_order_list(pairs, topk):
	print ('filter top_{}'.format(topk))
	order={}
	order_ks = set()
	for p in pairs:
		u = p[0]
		if u in order_ks:
			order[u].append(p)
		else:
			order_ks.add(u)
			order[u] = [p]

	res=[]
	for u in order_ks:
		order[u] =  sorted(order[u], key=lambda x: x[2])
		order[u] = order[u][:topk]
		res += order[u]
	
	return sorted(res, key=lambda x: x[2])

# models = ['candidate_pairs.baseline.nn.15.train-100k.with-orders.strict.json.gz', 
# 'candidate_pairs.nn.15.train-100k.doc2vec.json.gz',
# 'candidate_pairs.baseline.nn.15.train-100k.with-orders.tf-scaled.json.gz',
# 'candidate_pairs.baseline.nn.15.train-100k.with-orders.tf-binary.strict.json.gz'
# ]

models=['candidates/candidate_pairs.baseline.nn.100.train-100k.with-orders.tf-scaled.full-hierarchy.3.json.gz'
]#'candidates/candidate_pairs.nn.100.train-100k.word2vec.json.gz']

nn_pairs_lst = [filter_order_list(dictFromFileUnicode(m),15) for m in models]
order_objs = [OrderClass(ps) for ps in nn_pairs_lst]


nn_pairs= []
for ps in nn_pairs_lst:
	nn_pairs += ps
nn_pairs = filter_nn_pairs(nn_pairs)
random.shuffle(nn_pairs)

golden_edges = set()
positive_samples = []
timer = ProgressBar(title="Reading golden edges")
with open('train_100k.csv','r') as f:
	for line in f:
		timer.tick()
		uid1, uid2 = line.strip().split(',')
		golden_edges.add((min(uid1, uid2),max(uid1, uid2)))

TRAIN_SIZE = int(len(golden_edges)) #!!!
TEST_SIZE = 80000

sample_pairs_test = nn_pairs[-TEST_SIZE:]
nn_pairs = nn_pairs[:-TEST_SIZE]

sample_pairs_train = []
i=0
positive_count = 0
while len(sample_pairs_train)<TRAIN_SIZE and i<len(nn_pairs):
	if (min(nn_pairs[i][0],nn_pairs[i][1]), max(nn_pairs[i][0],nn_pairs[i][1])) in golden_edges:
		sample_pairs_train.append(nn_pairs[i])
		positive_count += 1
	i+=1

i=0
while len(sample_pairs_train)<2.5*positive_count and i<len(nn_pairs): #3.5-5 for testing
	if (min(nn_pairs[i][0],nn_pairs[i][1]), max(nn_pairs[i][0],nn_pairs[i][1])) not in golden_edges:
		sample_pairs_train.append(nn_pairs[i])
	i+=1

random.shuffle(sample_pairs_train)

del nn_pairs


def get_features_for_samples(sample_pairs):
	samples = []
	for pair in sample_pairs:
		# !!! duplication
		uid1,uid2 = pair[0], pair[1]
		samples.append([uid1,uid2,int((min(uid1,uid2),max(uid1,uid2)) in golden_edges)] + extract_feature_for_pair_users(uid1,uid2))
	return samples

samples_train = get_features_for_samples(sample_pairs_train)
samples_test = get_features_for_samples(sample_pairs_test)
X = [e[3:] for e in samples_train]
Y = [e[2] for e in samples_train]
XX = [e[3:] for e in samples_test]
YY = [e[2] for e in samples_test]


X = np.array(X)
Y = np.array(Y)
XX = np.array(XX)
YY = np.array(YY)
'''
Run Keras MLP
'''
# timer = ProgressBar(title="5-fold on {} samples (Random Forest)".format(len(samples_train)))
print("Training MLP...(with GPU hopefully)..")
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

# TY:Change to None, Del in Python doesn't do anything I believe
samples_train = None

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(10, input_dim=X.shape[1], init='normal', activation='relu'))
	model.add(Dense(1, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset

# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)

# encoded_YY = encoder.transform(YY)

model = create_baseline()

estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=25, batch_size=64, verbose=1)
estimator.fit(X, Y)
YY_result = estimator.predict(XX)

# kfold = StratifiedKFold(y=Y, n_folds=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)

# # scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
# # print "5-Fold score = {}".format(np.mean(scores))

# clf.fit(X, Y)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# YY_result = clf.predict(XX)
print "\n"
print "P[class-1] = {}".format(precision_score(YY, YY_result, pos_label=1, average='binary'))
print "R[class-1] = {}".format(recall_score(YY, YY_result, pos_label=1, average='binary'))
print "F1[class-1] = {}".format(f1_score(YY, YY_result, pos_label=1, average='binary'))
print "Golden_count={}; Predict_count={}".format(sum(YY), sum(YY_result))

from sklearn.externals import joblib

import sys
sys.setrecursionlimit(10000)

print "Saving deep model to disk.."
joblib.dump(estimator,'deep_models/mlp.pkl')

