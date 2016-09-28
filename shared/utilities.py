import json
import gzip
import re
import timeit
import urllib2
import urllib
import json
import random
from bs4 import BeautifulStoneSoup
from bs4 import BeautifulSoup
from bs4 import *
from operator import itemgetter
from collections import defaultdict
import gzip
import copy

def splitByDelirmiters(myStr):
	#myStr = unicode(myStr,'utf8')
	myStr  = re.split(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~ ]+',myStr)
	return myStr
	#return myStr.encode('utf8')

def getCleanText(text):
	# remove html tags and symbols
	return BeautifulSoup(HTMLEntitiesToUnicode(urllib.unquote(text))).text

def normalizeLines(text):
	# add '.' to the end and every line
	lines = text.split('\n')
	for l in lines:
		if len (l)==0:
			continue
		if l[-1]!='.':
			l = l + '.'
	return '\n'.join(lines)


def HTMLEntitiesToUnicode(text):
	"""Converts HTML entities to unicode.  For example '&amp;' becomes '&'."""
	# text = unicode(BeautifulStoneSoup(text, convertEntities=BeautifulStoneSoup.ALL_ENTITIES))
	return text

import time

def getData(url,data):
	for i in range(5):
		try:
			response = urllib2.urlopen(url, data, timeout=5)
			return response.read()
		except:
			time.sleep(1)
			pass
	return None

def getDataJson(url,data):
	return json.loads(getData(url,data))
	
class ProgressBar:
	def __init__(self, totalCount=-1, title=""):
		self.totalCount = totalCount
		self.title = title
		self.currentCount = 0
		self.start_time = timeit.default_timer()
		self.previous_time = self.start_time
		self.MIN_INTERVAL = 5
		if len(title)>0:
			print "Starting " + title

	def tick(self):
		self.currentCount += 1
		current_time = timeit.default_timer()
		if (current_time - self.previous_time)>self.MIN_INTERVAL:
			print "	Progress: {}/{} {}(s)".format(self.currentCount, self.totalCount, int(current_time - self.start_time))
			self.previous_time  = current_time

	def getCount(self):
		return self.currentCount
	
	def finish(self):
		current_time = timeit.default_timer()
		print "Finished: {} {}/{} {}(s)\n".format(self.title, self.currentCount, self.totalCount, int(current_time - self.start_time))


def jaccardSimilarityString(s1, s2):
	w1 = set(s1.split())
	w2 = set(s2.split())
	if len(w1|w2)==0:
		return 0
	else:
		return float(len(w1&w2))/float(len(w1|w2))

def jaccardSimilaritySet(s1, s2):
	if len(s1|s2)==0:
		return 0
	else:
		return float(len(s1&s2))/float(len(s1|s2))

def dictToFile(dict,path):
	print "Writing to {}".format(path)
	with gzip.open(path, 'w') as f:
		f.write(json.dumps(dict))

def dictToFileNormal(dict,path):
	print "Writing to {}".format(path)
	with open(path, 'w') as f:
		f.write(json.dumps(dict))

def dictFromFileUnicodeNormal(path):
	'''
	Read js file:
	key ->  unicode keys 
	string values -> unicode value
	'''
	print "Loading {}".format(path)
	with open(path, 'r') as f:
		return json.loads(f.read())

def dictFromFileUnicode(path):
	'''
	Read js file:
	key ->  unicode keys 
	string values -> unicode value
	'''
	print "Loading {}".format(path)
	with gzip.open(path, 'r') as f:
		return json.loads(f.read())


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