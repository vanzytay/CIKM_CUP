'''
Ordered Object Class and Some Utilities
'''

class OrderClass:
	def __init__(self, nn_pairs):
		self.orders = {}
		self.scores = {}
		self.extract_orders(nn_pairs)
		self.extract_scores(nn_pairs)

	def extract_orders(self, nn_pairs):
		self.orders = {}
		orders_keys = set()
		for p in nn_pairs:
			if p[0] not in orders_keys:
				orders_keys.add(p[0])
				self.orders[p[0]] = {}
			self.orders[p[0]][p[1]] = p[3]

	def extract_scores(self, nn_pairs):
		ps = set()
		score_keys = set()
		for p in nn_pairs:
			u,v = min(p[0],p[1]), max(p[0],p[1])
			if u not in score_keys:
				self.scores[u] =  {}
				score_keys.add(u)
			self.scores[u][v] =  p[2]

	def get_orders_infor(self,u,v):
		try:
			#if u in self.orders[v].keys():
				order2 = self.orders[v][u]
			#else:
			#	order2 = 101
		except:
			order2 = 101

		try:
			#if v in self.orders[u].keys():
				order1 = self.orders[u][v]
			#else:
			#	order1 = 101
		except:
			order1 = 101
		
		uu,vv = min(u,v), max(u,v)
		
		try:
			score = self.scores[uu][vv]
		except:
			score = 3

		return [order1,order2, score]#, int (order1!=101 and order2!=101), int (score>0), int (order1<=15 and order2<=15)]

def filter_nn_pairs(nn_pairs):
	'''
	Remove duplication of pair u,v and v,u
	'''
	candidates = []
	s = set()
	for p in nn_pairs:
		uid1, uid2 = min(p[0],p[1]), max(p[0],p[1])
		if (uid1,uid2) not in s:
			candidates.append((uid2,uid2))
			s.add((uid1, uid2))
	return candidates

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
