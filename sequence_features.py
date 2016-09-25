from collections import defaultdict
from myScripts import *
import string
import json
import tqdm


'''
Reading facts 
'''
from datetime import datetime
MAX_GRP = 24*7
def categorize_time(ts):
	if ts>9999999999999:
		ts = ts/1000
	dt = datetime.fromtimestamp(ts/1000)
	# print dt
	week_day = dt.weekday()
	time_slot = dt.hour
	return 24*week_day + time_slot


def get_semantic_urls(fids):
	url_f = []
	for fid in fids:
		url_f += []
	return url_f

def dict_to_order_list(d):
	lst = []
	x = min(d.keys())
	y = max(d.keys())
	for i in range(x,y+1):
		lst.append(d[i])
	return lst
'''
Convert mh5 url to word_id
'''

url2id = {}
url_set = set()
timer = ProgressBar(title="Reading urls.csv")
ec = 0
with open('./datasets/urls.csv','r') as f:
	for line in f:
		timer.tick()
		fid, tokens = line.strip().split(',')
		es = tokens.split('/')
		es = es[0].split('?')+ es[1:]
		url_set.update(es)

url_set.update('TIME_PAD')
url2id = dict((c, i) for i, c in enumerate(url_set))

dictToFile(url2id,'index/url2idb.json.gz')


# url2id = dictFromFileUnicode('index/url2idb.json.gz')

# timer = ProgressBar()
# user_features = {}

# fact_dict_url = {}
# timer = ProgressBar(title="Reading urls.csv")
# with open('./datasets/urls.csv','r') as f:
# 	for line in f:
# 		timer.tick()
# 		fid, tokens = line.strip().split(',')
# 		'''
# 		Take all url hierarchy
# 		'''
# 		urls=[]
# 		es = tokens.split('/')
# 		es = es[0].split('?')+ es[1:]
# 		path = ''
# 		for e in es[:3]:
# 			path=path+'M'+str(url2id[e])
# 			urls.append(path)

# 		fact_dict_url[int(fid)] = urls

# with open('./datasets/facts.json','r') as f:
# 	for line in f:
# 		js = json.loads(line.strip())
# 		# for each userID
# 		uid = js['uid']
# 		click_count_day_time = {}
# 		fact_count_day_time = {}
# 		for i in range(MAX_GRP):
# 			click_count_day_time[i] = 0
# 			fact_count_day_time[i] = []

# 		click_count_time = {}
# 		fact_count_time = {}
# 		for i in range(24):
# 			click_count_time[i] = 0
# 			fact_count_time[i] = []

# 		url_sequence = []
# 		old_url = ""
# 		for fact in js['facts']:
# 			fid = fact['fid']
# 			ts = fact['ts']
# 			dt = datetime.fromtimestamp(ts/1000)
# 			# categorize ts
# 			ts_grp = categorize_time(ts)

# 			click_count_day_time[ts_grp] += 1
# 			fact_count_day_time[ts_grp].append(fid)

# 			click_count_time[ts_grp%24] += 1
# 			fact_count_time[ts_grp%24].append(fid)
# 			url = fact_dict_url[fid]
# 			if(url[0]==old_url):
# 				last_time = ts
# 				continue
# 			# diff_time = ts.now()-last_time.now()
# 			# h = divmod(diff_time, 3600)  # hours
# 			# m = divmod(h[1], 60)  # minutes
# 			# if(h>1):
# 			# 	for i in range(h):
# 			# 		url_sequence.append(('TIME_PAD'))
# 			if(len(url)>0):
# 				url_sequence.append(url[0])
# 				old_url = url[0]
# 				last_time = datetime.fromtimestamp(ts/1000)

# 		user_features[uid] = {
# 			'click_count_day_time':dict_to_order_list(click_count_day_time),
# 			'click_count_time':dict_to_order_list(click_count_time),
# 			'url_sequence':url_sequence
# 			# 'facts_day_time':fact_count_day_time,
# 			# 'facts_time':fact_count_time
# 		}
# 		timer.tick()


# dictToFileNormal(user_features, 'user_features_ty.json.gz')