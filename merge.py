#!/usr/bin/env python
''' Script to ensemble submission 
'''
from __future__ import division
import os
import csv
from collections import Counter

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
			fileList.append(file)
	print("Returning full filelist")
	return fileList

file_list = readSubmissions('./history')
print(file_list)

pairs = []
users = []

for f in file_list:
	with open('./history/'+f,'r') as fin:
		for index, line in enumerate(fin):
			pair = line.strip().split(',')
			users.append(pair[0])
			users.append(pair[1])
			pairs.append(tuple(pair))
		

count_pairs = Counter(pairs)
# print(count_pairs)
sorted_pairs = sorted(count_pairs.items(), key=lambda i: i[1], reverse=True)

selected_pairs = []

# Only approve if >half agrees on pair
# threshold = int(len(file_list) / 2) + 4

threshold = len(file_list) - 1

selected_pairs = [x for x in sorted_pairs if x[1]>=threshold]

print(len(selected_pairs))

with open('./history/ensemble/submission.txt','w+') as f_out:
	for r in selected_pairs:
		f_out.write("{},{}\n".format(r[0][0],r[0][1]))



