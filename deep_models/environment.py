#!/usr/bin/env python
'''
Holds Environment Class (and iterator) 
'''
import numpy as np
import random
import sys
import logging
import string 
import os
from collections import Counter,OrderedDict
import json
from tqdm import tqdm
import math
import gensim
import multiprocessing
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import csv
from itertools import groupby


class Environment(object):
    '''
    Main Class for storing index, events and all. 
    '''
    def __init__(self, loader=['events','title','train','urls'], output_level='url', \
                    url_depth=2, sort_events=1):
        self.user_logs = {}
        self.user_map1 = {}
        self.user_map2 = {}
        self.events = []
        self.events2url = {}
        self.user_event_lengths = []
        self.limit = None
        self.user_logs = {}
        self.user_map1 = {}
        self.user_map2 = {}
        self.events2words = {}
        self.url_depth = url_depth
        self.loader = loader
        self.output_level = output_level
        self.sort_events = sort_events

        if(len(self.loader)==0):
            logging.warn("Please supply sets to load!")
            raise Exception

        # Loads a variety of sets 
        if('events' in self.loader):
            self.loadEvents()
        if('train' in self.loader):
            self.loadTrainSet()
        if('urls' in self.loader):
            self.loadUrls()
        if('title' in self.loader):
            self.loadWords()
        if(sort_events==1):
            self.sortEvents()

        logging.info("Building user index..")
        self.user_indices = dict((c,i) for i,c in enumerate(self.user_logs.keys()))

        logging.info("Finished init!")

    def sortEvents(self):
        for user,evt in tqdm(self.user_logs.items()):
            event_seq = []
            #ordered_events = dict(evt.items())
            ordered_events = {}
            if(self.sort_events==1):
                ordered_events = OrderedDict(sorted(evt.items()))
            else:
                ordered_events = dict(evt.items())
            if(self.output_level=='url'):
                # Convert to domain names
                evts = [self.events2url[x] for x in ordered_events.values() if x in self.events2url]
                evts = [x[0] for x in groupby(evts)]
            elif(self.output_level=='words'):
                evts = ordered_events.values()
            self.user_logs[user] = evts
            self.user_event_lengths.append(len(ordered_events))
            # Make a global event index
            self.events += ordered_events.values()

    def loadEvents(self):
        """
        Load Events (fids and their facts)
        """
        with open('./Dataset/facts.json') as f_in:
            i = 0
            for line in tqdm(f_in):
                if(self.limit is not None):
                    if(i>self.limit):
                        break
                j = json.loads(line.strip())
                #logging.info(j.get('facts'))
                uid = j.get('uid')
                # if(uid not in self.user_map1 and uid not in self.user_map2):
                #     continue
                if(uid not in self.user_logs):
                    self.user_logs[uid] = {}
                facts = j.get('facts')
                for x in facts:
                    timestamp = str(x['ts'])
                    # Add E to distinguish between timestamp
                    fid = 'E'+str(x['fid'])
                    self.user_logs[uid][timestamp] = fid
                i+=1

    def loadTrainSet(self):
        """
        Loads Training Set
        """
        logging.info("Loading Training Set")
        with open('./Dataset/train.csv') as f_in:
            for line in tqdm(f_in):
                user1,user2 = line.strip().split(',')
                self.user_map1[user1] = user2
                self.user_map2[user2] = user1
        logging.info("Got %d training pairs",len(self.user_map1))

    def loadUrls(self):
        """
        Load Urls 
        """
        logging.info("Loading URLS with url_depth=%d",self.url_depth)
        with open('./Dataset/urls.csv') as f_in:
            for line in tqdm(f_in):
                key, tokens = line.strip().split(',')
                key = 'E' + str(key)
                all_tokens = tokens.split('/')
                self.events2url[key] = all_tokens[0]
                if(self.url_depth>0):
                    for i in range(self.url_depth):
                        try:
                            next_token = '/' + str(all_tokens[1])
                            self.events2url[key] += next_token
                        except:
                            pass
    def loadWords(self):
        """
        Load titles and construct event2word dict
        """
        logging.info("Loading Titles...")
        with open('Dataset/titles.csv','r') as f:
            reader = csv.reader(f,delimiter='\t')
            for index,r in enumerate(reader):
                r = r[0]
                r = r.split(',')
                if(len(r)==0):
                    continue
                fid = 'E' + str(r[0])
                words = r[1]
                words = words.split()
                self.events2words[fid] = words

    def __iter__(self):
        user_items = self.user_logs.items()
        random.shuffle(user_items)
        for user,events in tqdm(user_items):
            # evts = events.values()
            # words = [self.events2url[x] for x in evts if x in self.events2url]
            if(self.output_level=='words'):
                _words = []
                for e in events:
                    if(e in self.events2words):
                        _tmp = self.events2words[e]
                        _words += _tmp
                events = _words
            if(len(events)==0):
                continue
            yield LabeledSentence(words=events, tags=['USER_%s' % user])

