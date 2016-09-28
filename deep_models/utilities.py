import json
import gzip

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