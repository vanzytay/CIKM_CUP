'''
Extend pairs by doing a->b, b->c then a->c
'''

from myScripts import *

results = []
with open('result_ordered.submit.txt','r') as f:
	for line in f:
		results.append(line.strip().split(','))


def write_to_file(results, filename):
	with open(filename,'w') as f:
		for r in filter_duplicate(results):
			f.write("{},{}\n".format(r[0],r[1]))


def evaluate(results):
	golden_edges = set()
	with open('test_100k.csv','r') as f:
		for line in f.readlines():
			es =  line.strip().split(',')
			uid1 = es[0]
			uid2 = es[1]
			golden_edges.add((min(uid1, uid2),max(uid1, uid2)))

	t_p = 0
	count = 0
	for pair in filter_duplicate(results):
		if (pair[0],pair[1]) in golden_edges:
			t_p+=1
		count += 1
	print "P@{} = {}".format(count, float(t_p)/count)
	print "R@{} = {}".format(count, float(t_p)/len(golden_edges))


def filter_duplicate(pairs):
	res = []
	pset = set()
	for p in pairs:
		if (min(p[0],p[1]), max(p[0],p[1])) not in pset:
			pset.add((min(p[0],p[1]), max(p[0],p[1])))
			res.append((min(p[0],p[1]), max(p[0],p[1])))
	return res


timer = ProgressBar(title="Filtering triple")
def extend_by_triple(pairs):
	adj = {}
	adj_set =set()
	for p in pairs:
		if p[0] not in adj_set:
			adj_set.add(p[0])
			adj[p[0]] = set([p[1]])
		else:
			adj[p[0]].add(p[1])

		if p[1] not in adj_set:
			adj_set.add(p[1])
			adj[p[1]] = set([p[0]])
		else:
			adj[p[1]].add(p[0])

	new_ps = []
	for u in adj_set:
		timer.tick()
		for v in adj[u]:
			if u<v:
				for k in adj[u]:
					if u<k and v<k:
						new_ps.append((u,v))
						new_ps.append((u,k))
						new_ps.append((v,k))
	new_ps = set(new_ps)

	res= [p for p in pairs if (min(p[0],p[1]),max(p[0],p[1])) in new_ps]
	return res #sorted(res, key=lambda x: x[2])


def extend_by_merging_cluster(pairs):
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
				if u<v and (u,v) not in edges:
					new_pairs.append((u,v))

	return new_pairs

# size = [len(s) for s in cluster.values()]
# import matplotlib.pyplot as plt
# plt.hist(size, bins=max(size))
# plt.show()


# '''
# split train/test 100k 100k
# '''
# r_set_lst = list(r_set)
# random.shuffle(r_set_lst)
# size = int(len(r_set_lst)/2.7)
# train_user_root_set = set(r_set_lst[:size])
# test_user_root_set = set(r_set_lst[-size:])

# #write train file
# with open('train_100k.csv','w') as f:
# 	for p in golden_edges:
# 		u1 = get_root(p[0])
# 		u2 = get_root(p[1])
# 		if u1!=u2:
# 			raise

# 		if u1 in train_user_root_set:
# 			f.write("{},{}\n".format(p[0],p[1]))

# #write test file
# with open('test_100k.csv','w') as f:
# 	for p in golden_edges:
# 		u1 = get_root(p[0])
# 		u2 = get_root(p[1])
# 		if u1!=u2:
# 			raise

# 		if u1 in test_user_root_set:
# 			f.write("{},{}\n".format(p[0],p[1]))

# size = [len(cluster[v])  for v in train_user_root_set]
# import matplotlib.pyplot as plt
# plt.hist(size, bins=max(size))
# plt.show()

# raise
# print max(size)

# new_pairs = []
# for cid in cluster.keys():
# 	if len(cluster[cid])<=4:
# 		for u in cluster[cid]:
# 			for v in cluster[cid]:
# 				if u<v and (u,v) not in edges:
# 					new_pairs.append((u,v))

# print len(pairs), len(new_pairs)

# with open("result_extend.submit.txt",'w') as f:
# 	for r in pairs:
# 		f.write("{},{}\n".format(r[0],r[1]))
# 	for r in new_pairs:
# 		f.write("{},{}\n".format(r[0],r[1]))
