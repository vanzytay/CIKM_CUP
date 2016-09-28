#!/usr/bin/env python
'''
Script for CIKM Cup 2016 
Cross-Device Entity Linking Challenge
Event-based RNN to predict user given a sequence of events
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Merge
from keras.layers import LSTM, Bidirectional, GRU
from keras.optimizers import RMSprop
from keras.layers.embeddings import Embedding
from keras.utils.data_utils import get_file
from keras.preprocessing import sequence
from keras.utils import np_utils
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
from utilities import *
from itertools import groupby
from keras.models import load_model


# Setup logger
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

class EventRNN:


    def saveEnvironment(self, mode='load'):
        limit = None
        # Load Events from File

        user_map1 = {}
        user_map2 = {}
        users_in_train = []
        events2url = {}

        # Get training users
        with open('./Dataset/train.csv') as f_in:
            for line in tqdm(f_in):
                user1,user2 = line.strip().split(',')
                user_map1[user1] = user2
                user_map2[user2] = user1
                users_in_train += [user2,user1]

        user_logs = {}
        with open('./Dataset/facts.json') as f_in:
            i = 0
            for line in tqdm(f_in):
                if(limit is not None):
                    if(i>limit):
                        break
                j = json.loads(line.strip())
                #logging.info(j.get('facts'))
                uid = j.get('uid')
                # if(uid not in user_map1 or uid not in user_map2):
                #     continue
                if(uid not in user_logs):
                    user_logs[uid] = {}
                facts = j.get('facts')
                for x in facts:
                    timestamp = str(x['ts'])
                    # Add E to distinguish between timestamp
                    fid = 'E'+str(x['fid'])
                    user_logs[uid][timestamp] = fid
                    # logging.info(fid)
                i+=1

        j = None

        url_index = []
        logging.info("Loading events2url")
        with open('./Dataset/urls.csv') as f_in:
                for line in tqdm(f_in):
                    key, tokens = line.strip().split(',')
                    key = 'E' + str(key)
                    url = tokens.split('/')[0]
                    events2url[key] = url
                    url_index.append(url)

        url_index = list(set(url_index))
        logging.info("Building Index..")
        user_indices = dict((c,i) for i,c in enumerate(user_logs.keys()))
        url_indices = dict((c,i) for i,c in enumerate(url_index))
        logging.info("Number of URLS:%d",len(url_index))
        logging.info("Ordering Events...")

        # Form Sentences
        events = []
        user_event_lengths = []
        # I'm not sure if the events are already ordered
        # Doing this just in-case
        i = 0
        for user,evt in tqdm(user_logs.items()):
            #ordered_events = dict(evt.items())
            ordered_events = OrderedDict(sorted(evt.items()))
            evts = [url_indices[events2url[x]] for x in ordered_events.values() if x in events2url]
            evts = [x[0] for x in groupby(evts)]
            user_logs[user] = evts
            # logging.info(user_logs[user])
            user_event_lengths.append(len(user_logs[user]))
            # Make a global event index
            # events += ordered_events.values()
            # Remove duplicate lists
        # events_counter = dict(Counter(events))
        # events = list(set(events))
        # logging.info("Raw Events:%d",len(events))
        # events = [key for key,value in events_counter.items() if value>25]
        # logging.info("Filtered Events:%d",len(events))
        # logging.info(events[0])

        self.env = {'user_log':user_logs,'url_index':url_index,'maxlen':np.max(user_event_lengths),'user_indices':user_indices}
        if(mode=='save'):
            dictToFile(self.env,'seq_env_sorted_urls_pruned_h0.json.gz')

    def setup():
        logging.info("Loading Training Set..")
        # Get training users
        self.users_in_train = []
        self.user_map1 = {}
        self.user_map2 = {}
        self.train_set = []
        with open('./Dataset/train.csv') as f_in:
            for line in tqdm(f_in):
                user1, user2 = line.strip().split(',')
                self.user_map1[user1] = user2
                self.user_map2[user2] = user1
                self.users_in_train += [user1,user2]
                self.train_set.append((user1,user2))

        # Loading Saved Dictionaries
        logging.info("Loading saved environment...")
        self.env = dictFromFileUnicode('seq_env_sorted_urls_h0.json.gz')
        self.user_logs = env['user_log']
        self.url_index = env['url_index']
        self.maxlen = env['maxlen']
        self.user_indices = env['user_indices']
        self.env = None
        self._create_model()

    def _create_model(self):
        EMBED_HIDDEN_SIZE = 32

        logging.info("Maxlen:%d",maxlen)
        logging.info("Creating LSTM model")
        #-----------------Create LSTM------------------------------#

        u1_rnn = Sequential()
        u1_rnn.add(Embedding(len(self.url_index), EMBED_HIDDEN_SIZE,
                              input_length=self.maxlen))
        u1_rnn.add(GRU(128))
        u1_rnn.add(Dropout(0.1))

        u2_rnn = Sequential()
        u2_rnn.add(Embedding(len(self.url_index), EMBED_HIDDEN_SIZE,
                              input_length=self.maxlen))
        u2_rnn.add(GRU(128))
        u2_rnn.add(Dropout(0.1))

        model = Sequential()
        model.add(Merge([u1_rnn,u2_rnn], mode='concat'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        return model


    def create_train_sets(self, verbose=1):

        # Build index and reverse index
        self.url_indices = dict((c, i) for i, c in enumerate(url_index))
        self.indices_url = dict((i, c) for i, c in enumerate(url_index))

        # Initialize
        X1,X2 = [],[]
        Y = [] 
        X1_test, X2_test = [],[]
        Y_test = []

        # Form corrupt set
        corrupt_set = []

        for pair in self.train_set:
            # Get random pair in train set and mix
            other_pair = self.train_set[random.randint(0,len(self.train_set)-1)]
            if(pair[0]<other_pair[1]):
                if(pair[0] in self.user_map1):
                    if(self.user_map1[pair[0]]==other_pair[1]):
                        continue
                new_pair = (pair[0],other_pair[1])
                corrupt_set.append(new_pair)
            else:
                if(pair[0] in user_map2):
                    if(self.user_map2[other_pair[1]]==pair[0]):
                        continue
                new_pair = (pair[0],other_pair[1])
                corrupt_set.append(new_pair)

        logging.info("Form corrupt set of %d",len(corrupt_set))

        random.shuffle(self.train_set)
        test_set = self.train_set[:25000]
        train_set = self.train_set[25000:50000]
        random.shuffle(corrupt_set)

        logging.info('Train Set:%d | Test Set:%d',len(train_set),len(test_set))

        for index, pair in enumerate(train_set):
            # Already converted to indices
            user1_seq = self.user_logs[pair[0]]
            user2_seq = self.user_logs[pair[1]]
            X1.append(user1_seq)
            X2.append(user2_seq)
            Y.append(1)
            # Sample corrupt
            corrupt_pair = corrupt_set[:25000][index]
            user1_seq = self.user_logs[corrupt_pair[0]]
            user2_seq = self.user_logs[corrupt_pair[1]]
            X1.append(user1_seq)
            X2.append(user2_seq)
            Y.append(0)

        for index,pair in enumerate(test_set):
            user1_seq = self.user_logs[pair[0]]
            user2_seq = self.user_logs[pair[1]]
            X1_test.append(user1_seq)
            X2_test.append(user2_seq)
            Y_test.append(1)
            # Sample corrupt
            corrupt_pair = corrupt_set[25000:][index]
            user1_seq = self.user_logs[corrupt_pair[0]]
            user2_seq = self.user_logs[corrupt_pair[1]]
            X1_test.append(user1_seq)
            X2_test.append(user2_seq)
            Y_test.append(0)

        self.X1 = np.array(X1)
        self.X2 = np.array(X2)
        self.X1_test = np.array(X1_test)
        self.X2_test = np.array(X2_test)
        self.Y = np.array(Y)
        self.Y_test = np.array(Y_test)

        if(verbose==1):
            logging.info(X1.shape)
            logging.info(X2.shape)
            logging.info(Y.shape)
            
        self.X1 = sequence.pad_sequences(X1, maxlen=maxlen)
        self.X2 = sequence.pad_sequences(X2, maxlen=maxlen)
        self.X1_test = sequence.pad_sequences(X1_test, maxlen=maxlen)
        self.X2_test = sequence.pad_sequences(X2_test, maxlen=maxlen)


    def fit():
        logging.info("Starting training of Event RNN")
        for i in range(0,100):
            self.model.fit([X1,X2],Y, nb_epoch=1, batch_size=32, verbose=1)
            scores = self.model.evaluate([X1_test,X2_test],Y_test, verbose=1)
            model.save_weights('Deep_Models/GRU1.h5', overwrite=True)


if __name__ == '__main__':
    e = EventRNN()
    e.setup()



