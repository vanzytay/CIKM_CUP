from myScripts import *
import string
import json


'''
Convert mh5 title to word_id
'''
# word2id = {}
# word_set = set()
# timer = ProgressBar(title="Reading titles.csv")
# with open('titles.csv','r') as f:
# 	for line in f:
# 		timer.tick()
# 		fid = int(line.strip().split(',')[0].strip())
# 		words = line.strip().split(',')[-1].strip().split()
# 		for word in words:
# 			word_set.add(word)


# wid = 0
# for word in word_set:
# 	wid += 1
# 	word2id[word] = "w{}".format(wid)
# dictToFile(word2id,'word2id.json.gz')

# word2id = dictFromFileUnicode('word2id.json.gz')

# fact_dict_title = {}
# timer = ProgressBar(title="Reading titles.csv")
# with open('titles.csv','r') as f:
# 	for line in f:
# 		timer.tick()
# 		fid = int(line.strip().split(',')[0].strip())
# 		words = line.strip().split(',')[-1].strip().split()
# 		fact_dict_title[fid] = []
# 		for word in words:
# 			fact_dict_title[fid].append(word2id[word])

'''
Convert mh5 url to word_id
'''
url2id = {}
url_set = set()
timer = ProgressBar(title="Reading urls.csv")
ec = 0
with open('urls.csv','r') as f:
	for line in f:
		timer.tick()
		fid, tokens = line.strip().split(',')
		es = tokens.split('/')
		es = es[0].split('?')+ es[1:]
		url_set.update(es)

urlid = 0
for url in url_set:
	urlid += 1
	url2id[url] = "{}".format(urlid)
# dictToFile(url2id,'url2idb.json.gz')

# url2id = dictFromFileUnicode('url2idb.json.gz')

fact_dict_url = {}
timer = ProgressBar(title="Reading urls.csv")
with open('urls.csv','r') as f:
	for line in f:
		timer.tick()
		fid, tokens = line.strip().split(',')


		'''
		Take all url hierarchy
		'''
		urls=[]
		es = tokens.split('/')
		es = es[0].split('?')+ es[1:]
		path = ''
		for e in es[:3]:
			path=path+'M'+url2id[e]
			urls.append(path)

		fact_dict_url[int(fid)] = urls


		# '''
		# Take domain only
		# '''
		# fact_dict_url[int(fid)] = [tokens.split('/')[0].split('?')[0]]

# '''
# Build sentence for url-title
# '''
# timer = ProgressBar(title="Building sentence for url-title")
# for fid in fact_dict_title.keys():
# 	timer.tick()
# 	sentence = ' '.join(fact_dict_title[fid])
# 	if len(sentence)>0:
# 		sentence = fact_dict_url[fid] + " " + sentence + " " + fact_dict_url[fid] 
# 	my_sentences.append(sentence)


'''
Build sentence for url-url
'''

def filter_duplicate_urls(urls):
	if len(urls)<2:
		return urls

	res = [urls[0]]
	for url in urls[1:]:
		if len(res)>0 and url==res[-1]:
			continue
		res.append(url)

	return res

# def filter_duplicate_urls(urls):

# 	def categorize_count(c):
# 		if 1<c<=5:
# 			return '5_'
# 		elif 5<c<=10:
# 			return '10_'
# 		elif c>10:
# 			return 'b_'
# 		return ''

# 	if len(urls)<2:
# 		return urls

# 	res = [urls[0]]
# 	c = 0
# 	for url in urls[1:]:
# 		if len(res)>1 and url==res[-1]:
# 			c+=1
# 		else:
# 			res.append(categorize_count(c)+url)
# 			c=1
# 	res.append(categorize_count(c)+url)

# 	return res



'''
user-user
'''

# get list of user in train
users_in_train = set()
with open('./train_5k.csv') as f:
	for line in f:
		user1,user2 = line.strip().split(',')
		users_in_train.update([user1,user2])

user_features = dictFromFileUnicodeNormal('user_features.json.gz')
all_users = set(user_features.keys())
users_in_train = all_users - users_in_train


pairs = set()
timer = ProgressBar(title="Reading golden edges")
with open('train.csv','r') as f:
	for line in f:
		timer.tick()
		uid1, uid2 = line.strip().split(',')
		pairs.add((min(uid1, uid2),max(uid1,uid2))) 

all_vertexs = set([p[0] for p in pairs] + [p[1] for p in pairs])
edges = set([(p[0],p[1]) for p in pairs])

r = {}
sv = {}
for v in all_vertexs:
	r[v] = -1
	sv[v] = 1

def get_root(u):
	while r[u]!=-1:
		u = r[u]
	return u

timer = ProgressBar("merging")
for pair in pairs:
	timer.tick()
	u,v = pair[0], pair[1]
	ru = get_root(u)
	rv = get_root(v)
	if ru!=rv:
		if sv[ru]<sv[rv]:
			r[ru] = rv
			sv[rv] += sv[ru]
		else:
			r[rv] =ru
			sv[ru] += sv[rv]

timer = ProgressBar("grouping")
cluster = {}
r_set = set()
for v in all_vertexs:
	timer.tick()
	if get_root(v) in r_set:
		cluster[get_root(v)].add(v)
	else:
		cluster[get_root(v)] = set([v])
		r_set.add(get_root(v))

user_has_group = set(r.keys())


from datetime import datetime
timer = ProgressBar(title="Reading facts.json")
with open('my_sentences.user-url.hierarchy.3.doc2vec.txt','w') as fo:
	with open('facts.json','r') as f:
		for line in f:
			timer.tick()
			js = json.loads(line.strip())

			# for each userID
			uid = js['uid']
			facts = []
			for fact in js['facts']:
				fid = fact['fid']
				ts = fact['ts']
				if ts>9999999999999:
					ts = ts/1000
				ts = ts/1000	
				facts.append((fid, ts))

			if len(facts)==0:
				continue

			facts = sorted(facts, key=lambda x: x[1])
			urls = fact_dict_url[facts[0][0]][:]
			last_time = datetime.fromtimestamp(facts[0][1])
			for i in range(1,len(facts)):
				if True or (datetime.fromtimestamp(facts[i][1]) - last_time).total_seconds()<3600*6:
					urls+=fact_dict_url[facts[i][0]]
					last_time = datetime.fromtimestamp(facts[i][1])
				else:
					urls = filter_duplicate_urls(urls)
					sentence = ' '.join(urls)
					sentence = uid + ' ' + sentence
					fo.write(sentence + "\n")

					urls = fact_dict_url[facts[i][0]][:]
					last_time = datetime.fromtimestamp(facts[i][1])

			#!!!

			uids = [uid]
			# if uid in users_in_train and uid in user_has_group:
			# 	root = get_root(uid)
			# 	if len(cluster[root])>1:
			# 		uids = cluster[root]

			urls = filter_duplicate_urls(urls)
			sentence = ' '.join(uids)+","+' '.join(urls)
			fo.write(sentence + "\n")


