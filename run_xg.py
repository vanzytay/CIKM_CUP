from __future__ import division
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

# Set up args parser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str,
                    help="the main functionality")
args = parser.parse_args()

feature_index = None

def normalize_click(a):
	return np.true_divide(a,np.sum(a)) if np.sum(a)!=0 else 0

user_features = dictFromFileUnicodeNormal('features/user_features.json.gz')
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

# Load doc2vec ensemble!
doc2vec_model = Doc2Vec.load('models/user-url.d.400.w.8.minf.1.filtered-urls.5trains.doc2vec')
doc2vec_model_h1 = Doc2Vec.load('models/mdl_urls_300_w5_h1.d2v')
doc2vec_model_h2 = Doc2Vec.load('models/mdl_urls_300_w10_h2.d2v')
doc2vec_model_h3 = Doc2Vec.load('models/mdl_urls_300_w10_h3.d2v')
doc2vec_model_concat = Doc2Vec.load('models/mdl_urls_300_concat.d2v')
word_model = Doc2Vec.load('models/mdl_word.d2v')

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
	global feature_index 
	feature_columns = []

	features = []
	f1 = user_features[uid1]
	f2 = user_features[uid2]

	# -----------Click Count Day Time---------------------# 
	click_count_day_time = f1['click_count_day_time'].tolist() + f2['click_count_day_time'].tolist()
	features += click_count_day_time
	
	# -----------Click Count Time---------------------# 
	click_count_time = f1['click_count_time'].tolist() + f2['click_count_time'].tolist()
	features+=click_count_time

	# ------------DIFF click count day time-----------#
	diff_click_count_day_time = np.absolute((f1['click_count_day_time'] - f2['click_count_day_time'])).tolist()
	features += diff_click_count_day_time

	# ------------DIFF click count time-----------#
	diff_click_count_time = np.absolute((f1['click_count_time'] - f2['click_count_time'])).tolist()
	features+=diff_click_count_time

	#not really help, reduce recall abit
	#-------------------Click_Count_Time_Normalized---#
	click_count_time_normalized = f1['click_count_time_normalized'].tolist() + f2['click_count_time_normalized'].tolist()
	features += click_count_time_normalized

	#not really help, reduce recall abit
	click_count_day_time_normalized = f1['click_count_day_time_normalized'].tolist() + f2['click_count_day_time_normalized'].tolist()
	features+= click_count_day_time_normalized

	
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

	#-------------------TF-IDF features--------------------------#
	order_objs_lens = []
	for obj in order_objs:
		tmp_f = obj.get_orders_infor(uid1,uid2)
		features+= tmp_f
		order_objs_lens.append(len(tmp_f))

	#------------------Doc2Vec Features--------------------------#
	try:
		v1 = doc2vec_model.docvecs[uid1]
		v2 = doc2vec_model.docvecs[uid2]
		cos = 1 - spatial.distance.cosine(v1, v2)
	except:
		cos = -1
		pass

	features.append(cos)
	#------------Doc2vec Higher Level x 1------------------------------#
	try:
		# TY: My word_level model uses USER_ + ID as label id
		v1 = doc2vec_model_h1.docvecs['USER_'+str(uid1)]
		v2 = doc2vec_model_h1.docvecs['USER_'+str(uid2)]
		cos = 1 - spatial.distance.cosine(v1, v2)
	except:
		cos = -1
		pass

	features.append(cos)

	#------------Doc2vec Higher Level x 2------------------------------#
	try:
		# TY: My word_level model uses USER_ + ID as label id
		v1 = doc2vec_model_h2.docvecs['USER_'+str(uid1)]
		v2 = doc2vec_model_h2.docvecs['USER_'+str(uid2)]
		cos = 1 - spatial.distance.cosine(v1, v2)
	except:
		cos = -1
		pass

	features.append(cos)

	#------------Doc2vec Higher Level x 3------------------------------#
	try:
		# TY: My word_level model uses USER_ + ID as label id
		v1 = doc2vec_model_h3.docvecs['USER_'+str(uid1)]
		v2 = doc2vec_model_h3.docvecs['USER_'+str(uid2)]
		cos = 1 - spatial.distance.cosine(v1, v2)
	except:
		cos = -1
		pass

	features.append(cos)


	#------------Doc2vec Level 0 Concat ------------------------------#
	try:
		# TY: My word_level model uses USER_ + ID as label id
		v1 = doc2vec_model_concat.docvecs['USER_'+str(uid1)]
		v2 = doc2vec_model_concat.docvecs['USER_'+str(uid2)]
		cos = 1 - spatial.distance.cosine(v1, v2)
	except:
		cos = -1
		pass

	features.append(cos)


	#------------Word-Lvl Doc2vec------------------------------#
	try:
		# TY: My word_level model uses USER_ + ID as label id
		v1 = word_model.docvecs['USER_'+str(uid1)]
		v2 = word_model.docvecs['USER_'+str(uid2)]
		cos = 1 - spatial.distance.cosine(v1, v2)
	except:
		cos = -1
		pass

	features.append(cos)

	if(feature_index is None):
		# This should be done once only! 
		print("Creating Feature Index...")
		feature_columns+=['click_count_day_time_'+str(i) for i in range(len(click_count_day_time))]
		feature_columns+=['click_count_time_'+str(i) for i in range(len(click_count_time))]
		feature_columns+=['diff_click_count_day_time_'+str(i) for i in range(len(diff_click_count_day_time))]
		feature_columns+=['diff_click_count_time_'+str(i) for i in range(len(diff_click_count_time))]
		feature_columns+=['click_count_time_normalized_'+str(i) for i in range(len(click_count_time_normalized))]
		feature_columns+=['click_count_day_time_normalized_'+str(i) for i in range(len(click_count_day_time_normalized))]
		for oo in order_objs_lens:
			feature_columns+=['order_objs_'+str(i) for i in range(oo)]
		feature_columns+=['doc2vec_dist']
		feature_columns+=['doc2vec_dist_h1']
		feature_columns+=['doc2vec_dist_h2']
		feature_columns+=['doc2vec_dist_h3']
		feature_columns+=['doc2vec_dist_concat']
		feature_columns+=['word_model_dist']
		# This should be equal!
		print("Number of feature columns %d",len(feature_columns))
		print("Total features: %d",len(features))
		feature_index = feature_columns
	# features+=get_time_overalap(uid1,uid2,60,[1,5,10,30])
	return features 


models=['candidates/candidate_pairs.baseline.nn.100.train-100k.with-orders.tf-scaled.full-hierarchy.3.json.gz']
#'candidates/candidate_pairs.nn.100.train-100k.word2vec.json.gz']

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




def get_features_for_samples(sample_pairs):
	samples = []
	for pair in sample_pairs:
		# !!! duplication
		uid1,uid2 = pair[0], pair[1]
		samples.append([uid1,uid2,int((min(uid1,uid2),max(uid1,uid2)) in golden_edges)] + extract_feature_for_pair_users(uid1,uid2))
	return samples



samples_train = get_features_for_samples(sample_pairs_train)
samples_test = get_features_for_samples(sample_pairs_test)

_tmp = {'train':[e[:3] for e in samples_train],'test':[e[:3] for e in samples_test]}

import cPickle as pickle
with open('train_test.pkl','w+') as f:
	pickle.dump(_tmp,f)
print("Saved train/test to file")

del nn_pairs


X = [e[3:] for e in samples_train]
Y = [e[2] for e in samples_train]
XX = [e[3:] for e in samples_test]
YY = [e[2] for e in samples_test]

'''
Setting up XG Boost
'''
timer = ProgressBar(title="Running XG Boost")

samples_train = None

xgb1 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=2500,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=8,
 reg_alpha=0.005,
 scale_pos_weight=1,
 seed=27)


# modelfit(xgb1, np.array(X), np.array(Y), feature_index=feature_index)
print("Fitting XGB[1]")
xgb1.fit(np.array(X),np.array(Y), verbose=True)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("Predicting XGB[1] on hold out set")
YY_result = xgb1.predict(XX)
print "P[class-1] = {}".format(precision_score(YY, YY_result, pos_label=1, average='binary'))
print "R[class-1] = {}".format(recall_score(YY, YY_result, pos_label=1, average='binary'))
print "F1[class-1] = {}".format(f1_score(YY, YY_result, pos_label=1, average='binary'))
print "Golden_count={}; Predict_count={}".format(sum(YY), sum(YY_result))

# print("Fitting XGB[2]")
# xgb2.fit(np.array(XX),np.array(YY), verbose=True)

# print("Predicting XGB[2] on hold out set")
# Y_result = xgb2.predict(X)
# print "P[class-1] = {}".format(precision_score(Y, Y_result, pos_label=1, average='binary'))
# print "R[class-1] = {}".format(recall_score(Y, Y_result, pos_label=1, average='binary'))
# print "F1[class-1] = {}".format(f1_score(Y, Y_result, pos_label=1, average='binary'))
# print "Golden_count={}; Predict_count={}".format(sum(Y), sum(Y_result))

'''
Predicting the real test pairs
'''
# models = ['candidate_pairs.baseline.nn.15.test-100k.with-orders.strict.json.gz', 
# 'candidate_pairs.nn.15.test-100k.doc2vec.json.gz',
# 'candidate_pairs.baseline.nn.15.test-100k.with-orders.tf-scaled.json.gz',
# 'candidate_pairs.baseline.nn.15.train-100k.with-orders.tf-binary.strict.json.gz'
# ]
# models=['candidates/candidate_pairs.baseline.nn.100.test-100k.with-orders.tf-scaled.full-hierarchy.3.json.gz'
# ]#'candidates/candidate_pairs.nn.100.test-100k.word2vec.json.gz']

models=['candidates/candidate_pairs.baseline.nn.100.test.with-orders.tf-scaled.full-hierarchy.3.json.gz']

nn_pairs_lst = [filter_order_list(dictFromFileUnicode(m),15) for m in models]
order_objs = [OrderClass(ps) for ps in nn_pairs_lst]


nn_pairs= []
for ps in nn_pairs_lst:
	nn_pairs += ps

nn_pairs = filter_nn_pairs(nn_pairs)
random.shuffle(nn_pairs)


from multiprocessing.pool import ThreadPool
pool = ThreadPool(20)


def get_features_for_samples_test(sample_pairs):
	samples = []
	for pair in sample_pairs:
		uid1,uid2 = pair[0], pair[1]
		# uid2 may be the keys
		try:
			if uid1 in orders[uid2].keys():
				order = orders[udi2][uid1]
			else:
				order = 100
		except:
			order=100
		samples.append(extract_feature_for_pair_users(uid1,uid2))
	return samples


timer = ProgressBar(title="Predicting")
def predict(pairs):
	timer.tick()
	samples = get_features_for_samples_test(pairs)
	confidences = xgb1.predict_proba(samples).tolist()
	# confidences2 = xgb2.predict_proba(samples).tolist()
	return [(pair[0],pair[1],confidences[i][1]) for i,pair in enumerate(pairs)]
	# return [(pair[0],pair[1], (confidences[i][1] + confidences2[i][1]) / 2) for i,pair in enumerate(pairs)]

def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in range(0, len(l), n):
		yield l[i:i + n]

results_tmp = pool.map(predict, chunks(nn_pairs, 10000))
results_tmp = [y for x in results_tmp for y in x]

# Combine scores
cs = {}
cs_keys = set()
r_ps = set()
for r in results_tmp:
	u1,u2 = min(r[0],r[1]), max(r[0],r[1])
	if u1 not in cs_keys:
		cs[u1] = {}
		cs_keys.add(u1)
	try:
		cs[u1][u2] = (cs[u1][u2] + r[2])/2
	except:
		cs[u1][u2] = r[2]
		pass
	r_ps.add((u1,u2))


ordered_score ={}
for r in results_tmp:
	ordered_score[r[0]] = []

for r in results_tmp:
	u1,u2 = r[0], r[1]
	ordered_score[u1].append((u2, cs[min(u1,u2)][max(u1,u2)]))

for u in ordered_score.keys():
	ordered_score[u] = sorted(ordered_score[u], key=lambda x: x[-1],  reverse=True)	

#finalize the result by layer
output_set = set()
results = []
current_layer = 0
while len(output_set)<len(r_ps):
	# !!!
	break
	tmp_lst = []
	for u1 in ordered_score.keys():
		if current_layer < len(ordered_score[u1]):
			u2 = ordered_score[u1][current_layer][0]
			if (min(u1,u2),max(u1,u2)) not in output_set:
				output_set.add((min(u1,u2),max(u1,u2)))
				tmp_lst.append((min(u1,u2),max(u1,u2),ordered_score[u1][current_layer][1]))
	tmp_lst = sorted(tmp_lst, key=lambda x: x[-1],  reverse=True)
	results += tmp_lst
	current_layer += 1
	if current_layer==1:
		break

results_full = [(p[0],p[1],cs[p[0]][p[1]]) for p in r_ps]
results_full = sorted(results_full, key=lambda x: x[-1],  reverse=True)

for r in results_full:
	u1,u2 = r[0], r[1]
	if (u1,u2) not in output_set:
		results.append((u1,u2, cs[u1][u2]))


# results = [(p[0],p[1],cs[p[0]][p[1]]) for p in r_ps]
# results = sorted(results, key=lambda x: x[-1],  reverse=True)
timer.finish()

# dictToFile(results, 'candidate_pairs.result.100.with-orders.json.gz')

def write_to_file(results, filename):
	with open(filename,'w') as f:
		for r in results:
			f.write("{},{}\n".format(r[0],r[1]))


def evaluate(results):
	golden_edges = set()
	with open('test_100k.csv','r') as f:
		for line in f:
			uid1, uid2 = line.strip().split(',')
			golden_edges.add((min(uid1, uid2),max(uid1, uid2)))

	t_p = 0
	count = 0
	for pair in results:
		if (pair[0],pair[1]) in golden_edges:
			t_p+=1
		count += 1
	print "P@{} = {}".format(count, float(t_p)/count)
	print "R@{} = {}".format(count, float(t_p)/len(golden_edges))


# output the top predictions
results_top_prediction = []
for r in results[:215307]:  #192149 #215307
	results_top_prediction.append((r[0],r[1]))
write_to_file(results_top_prediction, 'result_ordered.submit.txt')
evaluate(results_top_prediction)


# output the positive predictions
results_strict = []
for r in results:
	if r[2]>=0.5:
		results_strict.append((r[0],r[1]))
write_to_file(results_strict,'result_strict.submit.txt')
evaluate(results_strict)
