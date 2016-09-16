import json
import numpy as np
# this reference implementation solves the challenge in ~1h using 16 threads on an 8-core machine with 8Gb RAM
# it produces the submission.txt file that we share for demo purposes.

from tqdm import tqdm
from datetime import datetime
import pandas as pd
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing.dummy import Pool as ThreadPool

print datetime.now().strftime("%H:%M:%S start")

urls_tokens = {}
with open('./urls.csv') as f_in:
	for line in tqdm(f_in):
		key, tokens = line.strip().split(',')

		# domain = tokens.split('/')[0]
		# # !!!
		# if domain.find("?")>-1:
		# 	domain = domain.split('?')[0]

		# urls_tokens[key] = domain

		'''
		Take all url hierarchy
		'''
		urls=[]
		es = tokens.split('/')
		es = es[0].split('?')+ es[1:]
		path = ''
		for e in es[:3]:
			path=path+'_'+e
			urls.append(path)
		urls_tokens[key] = urls

facts = {}
with open('./facts.json') as f_in:
	for line in tqdm(f_in):
		j = json.loads(line.strip())
		lst=[]
		for x in j.get('facts'):
			for url in urls_tokens[str(x['fid'])]:
				lst.append(url)
		facts[j.get('uid')] = " ".join(lst) #[urls_tokens[str(x['fid'])] for x in j.get('facts')]
del urls_tokens 

users_in_train = set()
with open('./facts.txt','w') as f_out:
	with open('./test_100k.csv') as f_in:
		for line in tqdm(f_in):
			user1,user2 = line.strip().split(',')
			f_out.write("%s %s\n" % (facts[user1], facts[user2]))
			users_in_train.update([user1,user2])
users_for_predict = set(facts.keys()).difference(users_in_train)


users_for_predict_list = sorted(list(users_in_train)) #!!!users_for_predict
with open('./facts_test.tsv','w') as f_out:
	for x in users_for_predict_list:
		f_out.write("%s\t%s\n" % (x, facts[x]))
del users_for_predict_list
del users_in_train
del users_for_predict

print datetime.now().strftime("%H:%M:%S done with preprocessing")
		
tf = TfidfVectorizer(min_df=2, lowercase=False, preprocessor=None, binary=False, sublinear_tf =True).fit(map(lambda x: x.strip().split('\t')[-1], open('facts_test.tsv').readlines()))
print "Size Vobab = {}".format(len(tf.idf_))

#binary=True

del facts

users, row_text = [], []
with open('facts_test.tsv') as f_in:
	for line in tqdm(f_in):
		user, t = line.strip().split('\t')
		users.append(user)
		row_text.append(t)
		
tf_test = tf.transform(row_text)
del row_text

print datetime.now().strftime("%H:%M:%S done with tf-idf")

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(tf_test, range(1, tf_test.shape[0] + 1))
del tf_test
pool = ThreadPool(20)


'''
output with order
'''
from myScripts import *
timer = ProgressBar(title="Approximating nearest neighbor")
def get_predict(line):
	res = []
	timer.tick()
	user_id, tokens = line.strip().split('\t')
	tmp = knn.kneighbors(X=tf.transform([tokens]), n_neighbors=100, return_distance=True)
	for i in range(len(tmp[0][0])):
		if user_id!=users[tmp[1][0][i]]:
				res.append((user_id, users[tmp[1][0][i]], tmp[0][0][i], i+1))
	res = sorted(res, key=lambda x: x[2])
	return  [(r[0],r[1],r[2],i+1) for i,r in enumerate(res)]

lines = open('facts_test.tsv').readlines()
results = pool.map(get_predict, lines)
del knn
del tf
print datetime.now().strftime("%H:%M:%S done with predictions")

from myScripts import *
data = [y for x in results for y in x]
del results
data = sorted(data, key=lambda x: x[2])
dictToFile(data,'candidates/candidate_pairs.baseline.nn.100.test-100k.with-orders.tf-scaled.full-hierarchy.3.json.gz')


