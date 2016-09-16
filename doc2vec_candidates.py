from myScripts import *
import gensim, logging
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# get list of user in train
users_in_train = set()
with open('train_100k.csv') as f:
	for line in f:
		user1,user2 = line.strip().split(',')
		users_in_train.update([user1,user2])

# user_features = dictFromFileUnicodeNormal('user_features.json.gz')
# all_users = set(user_features.keys())
# users_in_train = all_users - users_in_train


# '''
# word2vec model
# '''

# # sentences = gensim.models.word2vec.LineSentence('my_sentences.user-url-title.word2vec.txt')
# # sentences = gensim.models.word2vec.LineSentence('my_sentences.user-url-title.user-embedding.word2vec.txt')
# # model = gensim.models.Word2Vec(sentences, size=400, window=5, min_count=1, workers=20, sg=1)
# # model.save("embedding.user-url-title.user-embedding.d.400.w.5.word2vec")


# model = gensim.models.Word2Vec.load("embedding.user-url-title.user-embedding.d.400.w.5.word2vec")
# # model = gensim.models.Word2Vec.load("embedding.user-url-title.d.400.w.5.word2vec")


# # fact2url = {}
# # timer = ProgressBar(title="Reading urls.csv")
# # with open('urls.csv','r') as f:
# # 	for line in f:
# # 		timer.tick()
# # 		fid = int(line.strip().split(',')[0].strip())
# # 		url = line.strip().split(',')[-1].split('/')[0]
# # 		url = url.split('?')[0]
# # 		fact2url[fid] = 'w' + url


# # def filter_duplicate_urls(urls):
# # 	if len(urls)<2:
# # 		return urls

# # 	res = [urls[0]]
# # 	for url in urls[1:]:
# # 		if len(res)>0 and url==res[-1]:
# # 			continue
# # 		res.append(url)

# # 	return res


# # user_vector = []
# # user_lst = []
# # timer = ProgressBar(title="Reading facts.json")
# # with open('facts.json','r') as f:
# # 	ec = 0
# # 	for line in f:
# # 		timer.tick()
# # 		js = json.loads(line.strip())

# # 		# for each userID
# # 		uid = js['uid']

# # 		# !!!
# # 		if uid not in users_in_train:
# # 			continue

# # 		words = [fact2url[fact['fid']] for fact in js['facts']]
# # 		words = filter_duplicate_urls(words)

# # 		vectors = []

# # 		for w in words:
# # 			try:
# # 				# vectors.append(model[w])
# # 				v = model[w]
# # 				vectors.append(v/np.linalg.norm(v))
# # 			except:
# # 				ec+=1
# # 				pass

# # 		uv = np.mean(vectors, axis=0)
# # 		uv = uv/np.linalg.norm(uv)

# # 		user_vector.append(uv)
# # 		user_lst.append(uid)

# # 	print "Key error: {}".format(ec)


# user_vector = []
# user_lst = []
# ec = 0
# for uid in users_in_train:
# 	try:
# 		v = model[uid]
# 	except:
# 		ec+=1
# 		continue
# 	user_vector.append(v/np.linalg.norm(v))
# 	user_lst.append(uid)

# print "Key error: {}".format(ec)


'''
Doc2vec model
'''
from gensim.models.doc2vec import *
from gensim.models import Doc2Vec

class LabeledLineSentence(object):
	def __init__(self, filename):
		self.filename = filename
	def __iter__(self):
		with open(self.filename,'r') as f:
			for line in f:
				es = line.strip().split(',')
				try:
					uids, sen = es[0].strip().split(), es[1].strip()
				except:
					print line
					raise
				yield TaggedDocument(sen.split(), uids)

model_name = 'hierarchy.3'
sentences = LabeledLineSentence('my_sentences.user-url.{}.doc2vec.txt'.format(model_name))


# model = Doc2Vec(sentences, size=300, window=8, min_count=2, workers=20)
# model.save('user-url.d.300.w.15.minf.2.{}.doc2vec'.format(model_name))
# model = Doc2Vec.load('user-url.d.300.w.15.minf.2.{}.doc2vec'.format(model_name))

model = Doc2Vec(alpha=0.025, min_alpha=0.025, size=300, window=15, min_count=2, workers=20)  # use fixed learning rate
model.build_vocab(sentences)
for epoch in range(5):
	 model.train(sentences)
	 model.alpha -= 0.002  # decrease the learning rate
	 model.min_alpha = model.alpha  # fix the learning rate, no decay
model.save('user-url.d.300.w.15.minf.2.{}.5trains.doc2vec')
model = Doc2Vec.load('user-url.d.300.w.15.minf.2.{}.5trains.doc2vec')

 
user_vector = []
user_lst = []
for uid in users_in_train:
	v = model.docvecs[uid]
	user_vector.append(v/np.linalg.norm(v))
	user_lst.append(uid)


'''
KNN based on user_vector and user_lst
'''

from multiprocessing.pool import ThreadPool
pool = ThreadPool(20)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, n_jobs = -1)
knn.fit(X=user_vector, y=range(1, len(user_lst)+1))

timer = ProgressBar(title="Nearest neighbor approximate")
def find_nearest(i):
	uid = user_lst[i]
	timer.tick()
	cur_vect = user_vector[i]
	tmp = knn.kneighbors(X=[cur_vect], n_neighbors=100, return_distance=True)
	res = []
	for j in range(len(tmp[0].tolist()[0])):
		if uid!=user_lst[tmp[1].tolist()[0][j]]:
			uid2= user_lst[tmp[1].tolist()[0][j]]
			res.append((uid, uid2, tmp[0][0][j], j+1))
	res = sorted(res, key=lambda x: x[2])
	return  [(r[0],r[1],r[2],j+1) for j,r in enumerate(res)]


nn_pairs = pool.map(find_nearest, range(len(user_lst)))
nn_pairs = set([y for x in nn_pairs for y in x])
nn_pairs = sorted(list(nn_pairs), key=lambda x: x[2])
timer.finish()

'''
SAVE TO FILE
'''
# dictToFile(nn_pairs,'candidate_pairs.split-url.doc2vec.json.gz')
# dictToFile(nn_pairs,'candidates/candidate_pairs.nn.100.train-100k.word2vec.json.gz')
dictToFile(nn_pairs,'candidate_pairs.nn.100.train-5k.{}.doc2vec.json.gz'.format(model_name))

