#!/usr/bin/env python
'''
Script to do inference
'''

SUBMIT_FULL = 215307
TOP_50 = int(0.5 * SUBMIT_FULL)
TOP_25 = int(0.25 * SUBMIT_FULL)
TOP_33 = int(0.33 * SUBMIT_FULL)
TOP_40 = int(0.4 * SUBMIT_FULL)
TOP_60 = int(0.6 * SUBMIT_FULL)
TOP_10 = int(0.1 * SUBMIT_FULL)

pairs = []
map1 = {}
map2 = {}
users = []
full_pairs = []

with open('result_ordered.submit.txt','r') as f:
	for index, line in enumerate(f):
		pair = line.strip().split(',')
		full_pairs.append(tuple(pair))

with open('result_ordered.submit.txt','r') as f:
	for index, line in enumerate(f):
		if(index>TOP_10):
			break
		pair = line.strip().split(',')
		pairs.append(tuple(pair))
		if(pair[0] not in map1):
			map1[pair[0]] = [pair[1]]
		else:
			map1[pair[0]].append(pair[1])
		if(pair[1] not in map2):
			map2[pair[1]] = [pair[0]]
		else:
			map2[pair[1]].append(pair[0])
		users.append(pair[0])
		users.append(pair[1])

users = list(set(users))

# If a->b and b->c, a->c
print("Number of initial pairs:%d", len(pairs))

infer_pairs = []

for u in users:
	if(u in map1):
		matches = map1[u]
		# print("%d matches",len(matches))
		for m in matches:
			_new_pairs = []
			if(m in map1):
				_new_pairs += map1[m]
			elif(m in map2):
				_new_pairs += map2[m]
			_new_pairs = list(set(_new_pairs))
			for n in _new_pairs:
				if(u<n):
					if(u in map1):
						if(n in map1[u]):
							continue
					_pair = tuple([u,n])
				else:
					if(n in map2):
						if(u in map2[n]):
							continue
					_pair = tuple([n,u])
				infer_pairs.append(_pair)
	

print(len(infer_pairs))
infer_pairs = list(set(infer_pairs))
print(len(infer_pairs))


all_pairs = pairs + infer_pairs
print(len(all_pairs))
all_pairs = list(set(all_pairs))
print(len(all_pairs))

overlap = set(pairs).union(infer_pairs)
infer_pairs = [x for x in infer_pairs if x not in overlap]
print("Removing overlap")
print(len(infer_pairs))
num_infer = len(infer_pairs)
num_to_take = SUBMIT_FULL - num_infer

for p in all_pairs:
	if(p[0]>p[1]):
		logging.warn("Pairs not in order")

all_pairs = full_pairs[:num_to_take] + infer_pairs
print("Num all pairs %d",len(all_pairs))

with open('results_inferred.txt','w+') as f_out:
	for r in all_pairs:
		f_out.write("{},{}\n".format(r[0],r[1]))




# # Remove duplicates
# infer_pairs = [x for x in infer_pairs if x not in pairs]
# print(len(infer_pairs))







