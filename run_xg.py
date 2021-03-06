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

#load time of event
user_click_times = defaultdict(list)

from datetime import datetime
timer = ProgressBar(title="Reading facts.json")
with open('./datasets/facts.json','r') as f:
	for line in f:
		timer.tick()
		js = json.loads(line.strip())

		# for each userID
		uid = js['uid']
		for fact in js['facts']:
			ts = fact['ts']
			if ts>9999999999999:
				ts = ts/1000
			ts = ts/1000	
			user_click_times[uid].append(ts)

for uid in user_click_times.keys():
	user_click_times[uid] = sorted(user_click_times[uid])

timer.finish()

# Load doc2vec ensemble!
doc2vec_model = Doc2Vec.load('models/user-url.d.400.w.8.minf.1.filtered-urls.5trains.doc2vec')
doc2vec_model_h1 = Doc2Vec.load('models/mdl_urls_300_w5_h1.d2v')
doc2vec_model_h2 = Doc2Vec.load('models/mdl_urls_300_w10_h2.d2v')
doc2vec_model_h3 = Doc2Vec.load('models/mdl_urls_300_w10_h3.d2v')
doc2vec_model_concat = Doc2Vec.load('models/mdl_urls_300_concat.d2v')

def get_time_overalap(u1,u2,interval):
	t1 = user_click_times[u1]
	t2 = user_click_times[u2]
	i1 = 0
	i2 = 0
	count = 0
	while i1<len(t1):
		i22 = i2
		while i22>=0 and (t1[i1]-t2[i22])<=interval:
			i22-=1
			count+=1

		while i2+1<len(t2) and abs(t1[i1]-t2[i2+1])<=interval:
			i2+=1
			count+=1

		i1+=1
	return count

def getDoc2VecFeatures(u1, u2, mdl):
	'''
	Gets doc2vec Features 
	returns 
		:Cosine Similarity
		:L1 and L2 Norm Distances
		:Orders of U1 in U2 and U2 in U1
	'''
	try:
		cos = mdl.docvecs.similarity(u1, u2)
	except:
		cos = -1

	try:
		euc = euclidean_distances([mdl.docvecs[u1]],[mdl.docvecs[u2]])[0][0]
	except:
		euc = -1

	try:
		mhd = manhattan_distances([mdl.docvecs[u1]],[mdl.docvecs[u2]])[0][0]
	except:
		mhd = -1

	return [cos, euc, mhd]

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
	
	features += [np.sum(f1['click_count_day_time'])]
	features += [np.sum(f2['click_count_day_time'])] 
	features += [np.sum(f1['click_count_day_time'])+ np.sum(f2['click_count_day_time'])]
	features += [np.abs([np.sum(f1['click_count_day_time'])- np.sum(f2['click_count_day_time'])])]

	features += [np.sum(f1['click_count_time'])]
	features += [np.sum(f2['click_count_time'])] 
	features += [np.sum(f1['click_count_time'])+np.sum(f2['click_count_time'])] 
	features += [np.abs([np.sum(f1['click_count_time'])- np.sum(f2['click_count_time'])])]
	
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
	features += getDoc2VecFeatures('USER_'+str(uid1),'USER_'+str(uid2), doc2vec_model_h1)


	#------------Doc2vec Higher Level x 2------------------------------#
	features += getDoc2VecFeatures('USER_'+str(uid1),'USER_'+str(uid2), doc2vec_model_h2)


	#------------Doc2vec Higher Level x 3------------------------------#
	features +=getDoc2VecFeatures('USER_'+str(uid1),'USER_'+str(uid2), doc2vec_model_h3)

	
	#------------Doc2vec Level 0 Concat ------------------------------#
	features +=getDoc2VecFeatures('USER_'+str(uid1),'USER_'+str(uid2), doc2vec_model_concat)

		
	# if(feature_index is None):
	# 	# This should be done once only! 
	# 	print("Creating Feature Index...")
	# 	feature_columns+=['click_count_day_time_'+str(i) for i in range(len(click_count_day_time))]
	# 	feature_columns+=['click_count_time_'+str(i) for i in range(len(click_count_time))]
	# 	feature_columns+=['diff_click_count_day_time_'+str(i) for i in range(len(diff_click_count_day_time))]
	# 	feature_columns+=['diff_click_count_time_'+str(i) for i in range(len(diff_click_count_time))]
	# 	feature_columns+=['click_count_time_normalized_'+str(i) for i in range(len(click_count_time_normalized))]
	# 	feature_columns+=['click_count_day_time_normalized_'+str(i) for i in range(len(click_count_day_time_normalized))]
	# 	for oo in order_objs_lens:
	# 		feature_columns+=['order_objs_'+str(i) for i in range(oo)]
	# 	feature_columns+=['doc2vec_dist']
	# 	feature_columns+=['doc2vec_dist_h1']
	# 	feature_columns+=['doc2vec_dist_h2']
	# 	feature_columns+=['doc2vec_dist_h3']
	# 	feature_columns+=['doc2vec_dist_concat']
	# 	feature_columns+=['word_model_dist']
	# 	# This should be equal!
	# 	print("Number of feature columns %d",len(feature_columns))
	# 	print("Total features: %d",len(features))
	# 	feature_index = feature_columns
	# features+=get_time_overalap(uid1,uid2,60,[1,5,10,30])
	
	features+=[get_time_overalap(uid1,uid2,5)]
	features+=[get_time_overalap(uid1,uid2,10)]
	features+=[get_time_overalap(uid1,uid2,60)]
	
	return features 

def get_features_for_samples(sample_pairs, golden_edges):
	samples = []
	for pair in tqdm(sample_pairs):
		# !!! duplication
		uid1,uid2 = pair[0], pair[1]
		samples.append([uid1,uid2,int((min(uid1,uid2),max(uid1,uid2)) in golden_edges)] + extract_feature_for_pair_users(uid1,uid2))
	return samples

def get_features_for_samples_test(sample_pairs):
	samples = []
	for pair in sample_pairs:
		uid1,uid2 = pair[0], pair[1]
		samples.append(extract_feature_for_pair_users(uid1,uid2))
	return samples

def write_to_file(results, filename, with_scores=False):
	with open(filename,'w') as f:
		for _ in results:
			if not with_scores:
				f.write("{},{}\n".format(_[0],_[1]))
			else:
				f.write("{},{},{}\n".format(_[0],_[1],_[2]))

def evaluate_on_test_98k(results):
	'''
	Evaluate on test_98k.csv
	'''
	golden_edges = set()
	with open('test_98k.csv','r') as f:
		for line in f:
			uid1, uid2 = line.strip().split(',')
			golden_edges.add((min(uid1, uid2),max(uid1, uid2)))
	
	# print "Len golden_pairs = {}".format(golden_edges)
	golden_edges_lst = list(golden_edges)
	random.shuffle(golden_edges_lst)
	golden_edges_half = set(golden_edges_lst[:int(len(golden_edges)/2)])
	t_p = 0
	count = 0
	for pair in results:
		if (pair[0],pair[1]) in golden_edges_half:
			t_p+=1
		count += 1
	pre = float(t_p)/count
	re = float(t_p)/len(golden_edges_half)
	print
	print "50%-pairs"
	print "P@{} = {}".format(count, pre)
	print "R@{} = {}".format(count, re)
	print "F1@{} = {}".format(count, 2*pre*re/(pre+re))
	t_p = 0
	count = 0
	for pair in results:
		if (pair[0],pair[1]) in golden_edges:
			t_p+=1
		count += 1
	pre = float(t_p)/count
	re = float(t_p)/len(golden_edges)
	print
	print "Full-pairs"
	print "P@{} = {}".format(count, pre)
	print "R@{} = {}".format(count, re)
	print "F1@{} = {}".format(count, 2*pre*re/(pre+re))


def extend_pairs(pairs):
	'''
	Extend pairs by merging 2 cluster. In other words, making pair u,v if u,v is in a connected component
	'''
	all_vertexs = set([p[0] for p in pairs] + [p[1] for p in pairs])
	edges = set([(p[0],p[1]) for p in pairs])

	r = {}
	for v in all_vertexs:
		r[v] = -1

	def get_root(u,r):
		while r[u]!=-1:
			u = r[u]
		return u

	for pair in pairs:
		u,v = pair[0], pair[1]
		ru = get_root(u,r)
		rv = get_root(v,r)
		if ru!=rv:
			r[ru] = rv

	cluster = {}
	for v in all_vertexs:
		try:
			cluster[get_root(v,r)].add(v)
		except:
			cluster[get_root(v,r)] = set([v])

	new_pairs = []
	for cid in cluster.keys():
		for u in cluster[cid]:
			for v in cluster[cid]:
				if u<v: # and (u,v) not in edges:
					new_pairs.append((u,v))

	return new_pairs

def get_top_pair_for_submit(pairs):
	p_set = set()
	p_lst = []
	c = 0
	for p in pairs:
		if (p[0],p[1]) not in p_set:
			p_set.add((p[0],p[1]))
			p_lst.append((p[0],p[1]))
			c+=1
			if c==215307:
				break
	return p_lst

'''
Setting up XG Boost
First XG Boost is to predict top pairs from the knn candidates
'''

train_candidate_sets=[
	'candidates/candidate_pairs.baseline.nn.100.train-98k.with-orders.tf-scaled.full-hierarchy.3.json.gz'
	,'candidates/candidate_pairs.nn.100.train-98k.mdl_urls_300_w10_h3.d2v.json.gz'
	,'candidates/candidate_pairs.nn.100.train-98k.mdl_urls_300_w10_h2.d2v.json.gz'
	,'candidates/candidate_pairs.nn.100.train-98k.mdl_urls_300_w5_h1.d2v.json.gz'
	,'candidates/candidate_pairs.nn.100.train-98k.domain-only.no-duplicate.doc2vec.json.gz'
	,'candidates/candidate_pairs.nn.100.train-98k.user-url-title.word2vec.json.gz'
]

nn_pairs_lst = [filter_order_list(dictFromFileUnicode(train_candidate_sets[0]),18)] + [filter_order_list(dictFromFileUnicode(_),10) for _ in train_candidate_sets[1:]]
order_objs = [OrderClass(filter_order_list(dictFromFileUnicode(_),60)) for _ in train_candidate_sets[:1]]+[OrderClass(filter_order_list(dictFromFileUnicode(_),15)) for _ in train_candidate_sets[1:]]

# Build the train and test data xgb1	

nn_pairs= []
for ps in nn_pairs_lst: #!!! skip embedding candidates
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
while len(sample_pairs_train)<3*positive_count and i<len(nn_pairs): #3.5-5 for testing
	if (min(nn_pairs[i][0],nn_pairs[i][1]), max(nn_pairs[i][0],nn_pairs[i][1])) not in golden_edges:
		sample_pairs_train.append(nn_pairs[i])
	i+=1

random.shuffle(sample_pairs_train)

del nn_pairs

samples_train = get_features_for_samples(sample_pairs_train, golden_edges)
samples_test = get_features_for_samples(sample_pairs_test, golden_edges)
X = [e[3:] for e in samples_train]
Y = [e[2] for e in samples_train]
XX = [e[3:] for e in samples_test]
YY = [e[2] for e in samples_test]

# Train xgb1

# Clean up to save memory
# Load doc2vec ensemble!
print("Cleaning doc2vec ensemble to save memory! (Will load again later)")
doc2vec_model = None
doc2vec_model_h1 = None
doc2vec_model_h2 = None
doc2vec_model_h3 = None
doc2vec_model_concat = None
word_model = None

timer = ProgressBar(title="Running XG Boost")

samples_train = None
del order_objs

xgb1 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=4000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=20,
 reg_alpha=0.005,
 scale_pos_weight=1,
 seed=27)

# modelfit(xgb1, np.array(X), np.array(Y), feature_index=feature_index)
print("Fitting XGB[1]")
xgb1.fit(np.array(X),np.array(Y), verbose=True)

del X
del Y
# Evaluate xgb1

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("Predicting XGB[1] on hold out set")
YY_result = xgb1.predict(XX)
print "P[class-1] = {}".format(precision_score(YY, YY_result, pos_label=1, average='binary'))
print "R[class-1] = {}".format(recall_score(YY, YY_result, pos_label=1, average='binary'))
print "F1[class-1] = {}".format(f1_score(YY, YY_result, pos_label=1, average='binary'))
print "Golden_count={}; Predict_count={}".format(sum(YY), sum(YY_result))

# Load doc2vec ensemble!
print("Reloading Doc2Vec Ensemble!")
doc2vec_model = Doc2Vec.load('models/user-url.d.400.w.8.minf.1.filtered-urls.5trains.doc2vec')
doc2vec_model_h1 = Doc2Vec.load('models/mdl_urls_300_w5_h1.d2v')
doc2vec_model_h2 = Doc2Vec.load('models/mdl_urls_300_w10_h2.d2v')
doc2vec_model_h3 = Doc2Vec.load('models/mdl_urls_300_w10_h3.d2v')
doc2vec_model_concat = Doc2Vec.load('models/mdl_urls_300_concat.d2v')
word_model = Doc2Vec.load('models/mdl_word.d2v')


'''
Build second Random Forest
Training pairs will be the top 50000 pairs from the last step and the pairs extended from them
'''

TOP_PAIRS_NB = 40000

dev_candidates_sets=[
	'candidates/candidate_pairs.baseline.nn.100.test-98k.with-orders.tf-scaled.full-hierarchy.3.json.gz'
	,'candidates/candidate_pairs.nn.100.test-98k.mdl_urls_300_w10_h3.d2v.json.gz'
	,'candidates/candidate_pairs.nn.100.test-98k.mdl_urls_300_w10_h2.d2v.json.gz'
	,'candidates/candidate_pairs.nn.100.test-98k.mdl_urls_300_w5_h1.d2v.json.gz'
	,'candidates/candidate_pairs.nn.100.test-98k.domain-only.no-duplicate.doc2vec.json.gz'
	,'candidates/candidate_pairs.nn.100.test-98k.user-url-title.word2vec.json.gz'
]

def predict_by_rf(rf_model, candidates_sets, strict_mode, nn_pairs=None):
	'''
	Return top predictions from random forest classifier
		 - rf_model: trained rf model
		 - candidates_sets: list of knn file to fetch the pairs
		 - strict_mode: if True => output only positive pairs. otherwise output by likelihood order
		 - nn_pairs: explicit pairs for predicting. If not None => candidates_sets won't be use
	'''
	global nn_pairs_lst
	global order_objs

	if nn_pairs==None:
		#nn_pairs_lst = [filter_order_list(dictFromFileUnicode(m),15) for m in candidates_sets]
		#order_objs = [OrderClass(ps) for ps in nn_pairs_lst]
		
		nn_pairs_lst = [filter_order_list(dictFromFileUnicode(candidates_sets[0]),18)] + [filter_order_list(dictFromFileUnicode(_),10) for _ in candidates_sets[1:]]
		order_objs = [OrderClass(filter_order_list(dictFromFileUnicode(_),60)) for _ in candidates_sets[:1]]+[OrderClass(filter_order_list(dictFromFileUnicode(_),15)) for _ in candidates_sets[1:]]
		
		nn_pairs= []
		for ps in nn_pairs_lst: # !!! skip embedding candidates
			nn_pairs += ps
		
		

		nn_pairs = filter_nn_pairs(nn_pairs)
		random.shuffle(nn_pairs)

	from multiprocessing.pool import ThreadPool
	pool = ThreadPool(20)

	timer = ProgressBar(title="Predicting")
	def predict(pairs):
		timer.tick()
		samples = get_features_for_samples_test(pairs)
		confidences = rf_model.predict_proba(samples).tolist()
		return [(pair[0],pair[1], confidences[i][1]) for i,pair in enumerate(pairs)]

	def chunks(l, n):
		"""Yield successive n-sized chunks from l."""
		for i in range(0, len(l), n):
			yield l[i:i + n]

	results_tmp = pool.map(predict, chunks(nn_pairs, 10000))
	results_tmp = [y for x in results_tmp for y in x]

	# Combine scores
	# Take average if u,v and v,u in the candidates set
	cs = {}
	cs_keys = set()
	r_ps = set()
	for _ in results_tmp:
		u1,u2 = min(_[0],_[1]), max(_[0],_[1])
		if u1 not in cs_keys:
			cs[u1] = {}
			cs_keys.add(u1)
		try:
			cs[u1][u2] = (cs[u1][u2] + _[2])/2
		except:
			cs[u1][u2] = _[2]
			pass
		r_ps.add((u1,u2))

	# Sort by likelihood of prediction
	results = [(p[0],p[1],cs[p[0]][p[1]]) for p in r_ps]
	results = sorted(results, key=lambda x: x[-1],  reverse=True)
	timer.finish()

	if strict_mode:
		results = [_ for _ in results if _[2]>0.5]

	return results

results = predict_by_rf(xgb1, dev_candidates_sets, False, None)

# If you don't use the inference part. Can ouput the results here. Note that the current results is for test_98k candidates file. 

print("Evaluating on test 98k..")
evaluate_on_test_98k(results[:100000])
evaluate_on_test_98k(results[:110000])
evaluate_on_test_98k(results[:115000])
evaluate_on_test_98k(results[:120000])
evaluate_on_test_98k(results[:125000])
evaluate_on_test_98k(results[:130000])
evaluate_on_test_98k(results[:210000])

write_to_file(results[:215307], './dev_scores/final_dev_result_scores.txt', with_scores=True)


# Extend top 50k pairs
TOP_PAIRS_NB = 45000
ext_pairs = extend_pairs(results[:TOP_PAIRS_NB])

# Build XGB2 on the extended pairs

golden_edges = set()
with open('test_98k.csv','r') as f:
	for line in f:
		uid1, uid2 = line.strip().split(',')
		golden_edges.add((min(uid1, uid2),max(uid1, uid2)))

samples_train = get_features_for_samples(ext_pairs, golden_edges)
X = [e[3:] for e in samples_train]
Y = [e[2] for e in samples_train]
print "Given {} pairs, extend to {} pairs ({} positive pairs)".format(TOP_PAIRS_NB, len(ext_pairs), sum(Y))

xgb2 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=4000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=20,
 reg_alpha=0.005,
 scale_pos_weight=1,
 seed=27)

print("Fitting XGB[2]")
xgb2.fit(np.array(X),np.array(Y), verbose=True)

'''
PREDICT ON REAL TEST DATA
	1. Candidate pairs filtering by knn
	2. Use first RF Classifier to take the top predictions
	3. Extend pairs from the 50k top predictions. 
	Extending approach is hirachical merning (similar to hierarchical clustering) with 
	the help of second RF classifier to take vote where 2 cluster should be merged
'''


test_candidate_sets=[
	'candidates/candidate_pairs.baseline.nn.100.test.with-orders.tf-scaled.full-hierarchy.3.json.gz'
	,'candidates/candidate_pairs.nn.100.test.mdl_urls_300_w10_h3.d2v.json.gz'
	,'candidates/candidate_pairs.nn.100.test.mdl_urls_300_w10_h2.d2v.json.gz'
	,'candidates/candidate_pairs.nn.100.test.mdl_urls_300_w5_h1.d2v.json.gz'
	,'candidates/candidate_pairs.nn.100.test.domain-only.no-duplicate.doc2vec.json.gz'
	,'candidates/candidate_pairs.nn.100.test.user-url-title.word2vec.json.gz'
]

XGB1_results = predict_by_rf(xgb1, test_candidate_sets, False, None)

write_to_file(XGB1_results[:215307], 'result_ordered.submit.txt',False)
write_to_file(XGB1_results[:215307], 'result_ordered.score.txt',True)

ext_pairs = extend_pairs(XGB1_results[:TOP_PAIRS_NB])
print "Number of extended pairs {}".format(len(ext_pairs))


# Hierarchical merging
def get_root(u,r):
	while r[u]!=-1:
		u = r[u]
	return u

for TOP_PAIRS_NB in [35000,40000,45000,50000]:

	ext_pairs = extend_pairs(XGB1_results[:TOP_PAIRS_NB])
	print "Number of extended pairs {}".format(len(ext_pairs))

	# Caching score of XGB2 prediction
	results_ext_scores = defaultdict(float)
	for _ in predict_by_rf(xgb2, test_candidate_sets, False, nn_pairs=ext_pairs):
		results_ext_scores[(_[0],_[1])] = abs(_[2])


	print "Inferencing top {} pairs".format(TOP_PAIRS_NB)
	r = defaultdict(lambda: -1)
	cluster = defaultdict(set)
	for p in XGB1_results[:TOP_PAIRS_NB]:
		cluster[p[0]] = set([p[0]])
		cluster[p[1]] = set([p[1]])

	XGB2_ext_pairs = []
	timer = ProgressBar(title="Extending pairs")
	for p in XGB1_results[:TOP_PAIRS_NB]:
		timer.tick()
		ru = get_root(p[0],r)
		rv = get_root(p[1],r)
		if ru!=rv:
			clusters_pairs = []
			p_count = 0
			c_score = 0
			for u in cluster[ru]:
				for v in cluster[rv]:
					p_count +=1
					c_score += results_ext_scores[(min(u,v),max(u,v))]
					clusters_pairs.append((min(u,v),max(u,v)))
			confidences = float(c_score)/p_count
			if ((confidences >= 0.3) and (len(cluster[ru]) + len(cluster[rv])<50)) or (len(cluster[ru]) + len(cluster[rv])<=3):
				XGB2_ext_pairs += clusters_pairs
				for u in cluster[ru]:
					cluster[rv].add(u)
				r[ru] = rv

	print "Final results: given {} pairs, extend to {} pairs".format(TOP_PAIRS_NB, len(XGB2_ext_pairs))
	print "new_extended pairs {}".format(len(set(get_top_pair_for_submit(XGB2_ext_pairs+XGB1_results))-set([(p[0],p[1]) for p in XGB1_results[:215307]])))

	# Merge XGB2 ext_pairs on top of XGB1 predictions 
	write_to_file(get_top_pair_for_submit(XGB2_ext_pairs+XGB1_results), 'result_ordered.with-inference.{}.submit.txt'.format(TOP_PAIRS_NB), False)
