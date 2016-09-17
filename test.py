
models=['candidates/candidate_pairs.baseline.nn.100.train-100k.with-orders.tf-scaled.full-hierarchy.3.json.gz']

nn_pairs_lst = [filter_order_list(dictFromFileUnicode(m),15) for m in models]

print(len(nn_pairs_lst))