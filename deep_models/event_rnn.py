#!/usr/bin/env python
''' Script for CIKM Cup 2016 
Cross-Device Entity Linking Challenge
Event-based RNN to predict user given a sequence of events
'''
from __future__ import print_function
import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
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
from itertools import groupby
from keras.models import load_model
sys.path.append('../shared')
from ordered_object import *
from utilities import *
import cPickle as pickle

# Setup logger
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

SUBMIT_FULL = 215307

def write_to_file(results, filename):
    with open(filename,'w') as f:
        for r in results:
            f.write("{},{}\n".format(r[0],r[1]))

def write_score_to_file(results, filename):
    # Save scores
    with open(filename,'w') as f:
        for r in results:
            f.write("{},{},{}\n".format(r[0],r[1],r[2]))

class EventRNN:
    """ Event Recurrent Neural Network 
    Implements 2 GRU units using Keras
    Performs binary classification with two GRUs
    Trains user pair -> 1 if same user and =0 if otherwise (negative sampling)
    """

    def __init__(self):
        self.model = None

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

    def setup(self):
        logging.info("Loading Training Set..")
        # Get training users
        self.users_in_train = []
        self.user_map1 = {}
        self.user_map2 = {}
       
        with open('../datasets/train.csv') as f_in:
            for line in tqdm(f_in):
                user1, user2 = line.strip().split(',')
                self.user_map1[user1] = user2
                self.user_map2[user2] = user1
                self.users_in_train += [user1,user2]
                # self.train_set.append((user1,user2))

        with open('train_test.pkl','r') as f:
            _tmp = pickle.load(f)

        self.train_set = _tmp['train']
        self.test_set = _tmp['test']

        # Loading Saved Dictionaries
        logging.info("Loading saved environment...")
        self.env = dictFromFileUnicode('seq_env_sorted_urls_h0.json.gz')
        self.user_logs = self.env['user_log']
        self.url_index = self.env['url_index']
        self.maxlen = self.env['maxlen']
        self.user_indices = self.env['user_indices']
        self.env = None
           # Build index and reverse index
        self.url_indices = dict((c, i) for i, c in enumerate(self.url_index))
        self.indices_url = dict((i, c) for i, c in enumerate(self.url_index))
        

    def _create_model(self, weight_path=None):
        """ Creates a dual-GRU
        """
        EMBED_HIDDEN_SIZE = 128

        logging.info("Maxlen:%d", self.maxlen)
        logging.info("Creating LSTM model")
        #-----------------Create LSTM------------------------------#

        u1_rnn = Sequential()
        u1_rnn.add(Embedding(len(self.url_index), EMBED_HIDDEN_SIZE,
                              input_length=self.maxlen))
        u1_rnn.add(Bidirectional(GRU(256)))
        u1_rnn.add(Dropout(0.1))
        

        u2_rnn = Sequential()
        u2_rnn.add(Embedding(len(self.url_index), EMBED_HIDDEN_SIZE,
                              input_length=self.maxlen))
        u2_rnn.add(Bidirectional(GRU(256)))
        u2_rnn.add(Dropout(0.1))

        model = Sequential()
        model.add(Merge([u1_rnn,u2_rnn], mode='concat'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        if(weight_path is not None):
            self._load_weights(weight_path)
        return model


    def create_train_sets(self, limit=None, verbose=1):

        # Initialize
        X1,X2 = [],[]
        Y = [] 
        X1_test, X2_test = [],[]
        Y_test = []

        # Form corrupt set
        # for pair in self.train_set:
        #     # Get random pair in train set and mix
        #     other_pair = self.train_set[random.randint(0,len(self.train_set)-1)]
        #     if(pair[0]<other_pair[1]):
        #         if(pair[0] in self.user_map1):
        #             if(self.user_map1[pair[0]]==other_pair[1]):
        #                 continue
        #         new_pair = (pair[0],other_pair[1])
        #         corrupt_set.append(new_pair)
        #     else:
        #         if(pair[0] in user_map2):
        #             if(self.user_map2[other_pair[1]]==pair[0]):
        #                 continue
        #         new_pair = (pair[0],other_pair[1])
        #         corrupt_set.append(new_pair)

        if(limit is not None):
            random.shuffle(self.train_set)
            # random.shuffle(self.test_set)
            self.train_set = self.train_set[:limit]
            # self.test_set = self.test_set[:limit]
        logging.info('Train Set:%d | Test Set:%d',len(self.train_set),len(self.test_set))
        seq_lens = []
        for index, pair in enumerate(self.train_set):
            # Already converted to indices
            user1_seq = self.user_logs[pair[0]]
            user2_seq = self.user_logs[pair[1]]
            seq_lens += [len(user1_seq),len(user2_seq)]
            X1.append(user1_seq)
            X2.append(user2_seq)
            Y.append(pair[2])
            # Sample corrupt
            # corrupt_pair = corrupt_set[:25000][index]
            # user1_seq = self.user_logs[corrupt_pair[0]]
            # user2_seq = self.user_logs[corrupt_pair[1]]
            # X1.append(user1_seq)
            # X2.append(user2_seq)
            # Y.append(0)

        for index,pair in enumerate(self.test_set):
            user1_seq = self.user_logs[pair[0]]
            user2_seq = self.user_logs[pair[1]]
            seq_lens += [len(user1_seq),len(user2_seq)]
            X1_test.append(user1_seq)
            X2_test.append(user2_seq)
            Y_test.append(pair[2])
            # Sample corrupt
            # corrupt_pair = corrupt_set[25000:][index]
            # user1_seq = self.user_logs[corrupt_pair[0]]
            # user2_seq = self.user_logs[corrupt_pair[1]]
            # X1_test.append(user1_seq)
            # X2_test.append(user2_seq)
            # Y_test.append(0)
        # from collections import Counter

        # count_lens = Counter(seq_lens)
        # logging.info(count_lens)

        self.X1 = np.array(X1)
        self.X2 = np.array(X2)
        self.X1_test = np.array(X1_test)
        self.X2_test = np.array(X2_test)
        self.Y = np.array(Y)
        self.Y_test = np.array(Y_test)

        if(verbose==1):
            logging.info(self.X1.shape)
            logging.info(self.X2.shape)
            logging.info(self.Y.shape)

        logging.info("Clipping max seq..")
        self.maxlen = 200  
        logging.info("Max Sequence Length:%d",self.maxlen)

        self.X1 = sequence.pad_sequences(X1, maxlen=self.maxlen)
        self.X2 = sequence.pad_sequences(X2, maxlen=self.maxlen)
        self.X1_test = sequence.pad_sequences(X1_test, maxlen=self.maxlen)
        self.X2_test = sequence.pad_sequences(X2_test, maxlen=self.maxlen)

        self._create_model()

    def fit(self, compact=True):
        logging.info("Starting training of Event RNN!")
        if(compact):
            self._compact()
        for i in range(0,100):
            self.model.fit([self.X1,self.X2],self.Y, nb_epoch=1, batch_size=16, verbose=1)
            if(i % 10==0):
                scores = self.model.evaluate([self.X1_test,self.X2_test], self.Y_test, verbose=1)
                logging.info(scores)
            self.model.save_weights('models/GRU_10k.h5', overwrite=True)

    def _compact(self):
        ''' save memory while training RNN
        '''
        logging.info("Removing index to save space while training..")
        self.env = None
        self.user_logs = None
        self.url_index = None
        self.user_indices = None
        self.url_indices = None
        self.indices_url = None
        
    def loadCandidates(self):
        """ Calculate scores for candidate pairs
        """
        models=['../candidates/candidate_pairs.baseline.nn.100.test.with-orders.tf-scaled.full-hierarchy.3.json.gz']
        nn_pairs_lst = [filter_order_list(dictFromFileUnicode(m),5) for m in models]
        order_objs = [OrderClass(ps) for ps in nn_pairs_lst]
        nn_pairs= []
        for ps in nn_pairs_lst:
            nn_pairs += ps
        nn_pairs = filter_nn_pairs(nn_pairs)
        self.candidates = nn_pairs
        logging.info("Loaded %d candidates",len(self.candidates))

    def _load_weights(self, weight_path):
        self.model.load_weights(weight_path)
        logging.info("Loaded weights!")

    def computeCandidateScores(self):
        """ Computes scores for candidate pairs
        """
        self.candidate_scores = {}
        X1, X2 = [], []
        for pair in tqdm(self.candidates):
            user1_seq = self.user_logs[pair[0]]
            user2_seq = self.user_logs[pair[1]]
            X1.append(user1_seq)
            X2.append(user2_seq)
        X1 = np.array(X1)
        X2 = np.array(X2)
        _x1 = sequence.pad_sequences(X1, maxlen=self.maxlen)
        _x2 = sequence.pad_sequences(X2, maxlen=self.maxlen)
        logging.info("Predicting probability now...")
        proba = self.model.predict_proba([_x1,_x2], batch_size=512)
        for index,p in tqdm(enumerate(proba)):
            logging.info(p)
            self.candidate_scores[tuple(self.candidates[index])] = p[0]

    def generatePredictions(self, k):
        """ Generates top-k predictions from candidates
        """
        import operator
        with open('candidate_scores_10k.pkl','r') as f:
            self.candidate_scores = pickle.load(f)
        logging.info("Loaded scores!")
        sorted_x = sorted(self.candidate_scores.items(), key=operator.itemgetter(1), reverse=True)
        logging.info(sorted_x[k])
        logging.info(len(sorted_x))
        scores = [x[1] for x in sorted_x]
        below_one = [x for x in scores if x<0.5]
        logging.info(len(below_one))
        logging.info(np.max(scores))
        logging.info(np.min(scores))
        results = []
        results_score = []
        for data in sorted_x[:k]:
            pair = data[0]
            results.append((pair[0],pair[1]))
            results_score.append((pair[0],pair[1],data[1]))
        logging.info("Number of results %d",len(results))
        write_to_file(results,'result_rnn_10k.txt')
        write_score_to_file(results_score,'results_rnn_10k_scores.txt')


if __name__ == '__main__':
    # Set up args parser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str,
                    help="the main functionality")
    args = parser.parse_args()
    e = EventRNN()
    e.setup()
    e.maxlen = 200
    if(args.mode=='predict'):
        # Prediction Mode using CPU
        # e._create_model(weight_path = 'models/GRU_max.h5')
        # e.loadCandidates()
        # e.computeCandidateScores()
        # with open('candidate_scores_max.pkl','w+') as f:
        #     pickle.dump(e.candidate_scores, f)
        e.generatePredictions(SUBMIT_FULL)
    elif(args.mode=='train'):
        e.create_train_sets(limit=10000)
        e.fit()
  
    # dictToFileNormal(e.candidate_scores, 'rnn_scores.json.gz')
    # logging.info("Finished!")

    # e.generatePredictions(SUBMIT_FULL/2)

