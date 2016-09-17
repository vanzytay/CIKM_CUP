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


timer = ProgressBar()
user_features = {}

with open('./datasets/facts.json','r') as f:
	for line in f:
		js = json.loads(line.strip())
		print(js)
		# for each userID
		uid = js['uid']
		click_count_day_time = {}
		fact_count_day_time = {}
		for i in range(MAX_GRP):
			click_count_day_time[i] = 0
			fact_count_day_time[i] = []

		click_count_time = {}
		fact_count_time = {}
		for i in range(24):
			click_count_time[i] = 0
			fact_count_time[i] = []

		for fact in js['facts']:
			fid = fact['fid']
			ts = fact['ts']

			# categorize ts
			ts_grp = categorize_time(ts)

			click_count_day_time[ts_grp] += 1
			fact_count_day_time[ts_grp].append(fid)

			click_count_time[ts_grp%24] += 1
			fact_count_time[ts_grp%24].append(fid)

		user_features[uid] = {
			'click_count_day_time':dict_to_order_list(click_count_day_time),
			'click_count_time':dict_to_order_list(click_count_time)
			# 'facts_day_time':fact_count_day_time,
			# 'facts_time':fact_count_time
		}
		timer.tick()


# dictToFileNormal(user_features, 'user_features.json.gz')