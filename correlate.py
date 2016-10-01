#!/usr/bin/env python
''' Script to calculate correlations and analyze submissions
'''
from __future__ import division
import os
import csv
from collections import Counter
import numpy as np

SUBMIT_FULL = 215307
# SUBMIT_FULL = 115000

TOPK = SUBMIT_FULL

def readSubmissions(directory,limitFiles=None):
	"""
	Read all CSV files from partials directory
	"""
	print("Reading submission file from %s", directory)
	fileList = []
	for file in os.listdir(str(directory)):
		if file.endswith(".txt"):
			if(limitFiles!=None):
				# Allow passing no limits
				if(len(fileList)>=limitFiles):
					print("Limit reached,returning filelist with len %d",len(fileList))
					return fileList
			if('scores' in file):
				fileList.append(file)
	print("Returning full filelist")
	return fileList

file_list = readSubmissions('./ensemble')
print(file_list)

pairs = []
users = []

score_dict = {}

import itertools
file_pairs = list(itertools.combinations(file_list, 2))

print(file_pairs)

sub_dict = {}

for f in file_list:
	sub_dict[f] = []
	with open('./ensemble/'+f,'r') as fin:
		for index, line in enumerate(fin):
			if(index>=TOPK):
				break
			pair = line.strip().split(',')
			if(len(pair)<2):
				continue
			sub_dict[f].append(tuple([pair[0],pair[1]]))

for fp in file_pairs:
	_f1 = set(sub_dict[fp[0]])
	_f2 = set(sub_dict[fp[1]])
	overlap = _f1.intersection(_f2)
	overlap = set(overlap)
	print("{} vs {} | {}".format(fp[0],fp[1],str(len(overlap)/TOPK)))



			# users.append(pair[0])
			# users.append(pair[1])
			# user_pair = tuple([pair[0],pair[1]])
			# if(user_pair in score_dict):
			# 	score_dict[user_pair]+=[pair[2]]
			# else:
			# 	score_dict[user_pair]=[pair[2]]
			# pairs.append(tuple(pair))
		
# for key, value in score_dict.items():
# 	score_dict[key] = np.mean(np.array(value).astype(np.float))
# # count_pairs = Counter(pairs)
# # print(count_pairs)
# sorted_pairs = sorted(score_dict.items(), key=lambda i: i[1], reverse=True)

# selected_pairs = []

# Only approve if >half agrees on pair
# threshold = int(len(file_list) / 2) + 4

# threshold = len(file_list) - 1

# selected_pairs = [x for x in sorted_pairs if x[1]>=threshold][:SUBMIT_FULL]

# print("{}".format(len(sorted_pairs)))

# selected_pairs = sorted_pairs[:SUBMIT_FULL]

# print(len(selected_pairs))

# with open('./ensemble/final/submission.txt','w+') as f_out:
# 	for r in selected_pairs:
# 		f_out.write("{},{}\n".format(r[0][0],r[0][1]))

# with open('./ensemble/final/submission_scores.txt','w+') as f_out:
# 	for r in selected_pairs:
# 		f_out.write("{},{},{}\n".format(r[0][0],r[0][1],r[1]))



