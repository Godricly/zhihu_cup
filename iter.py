import os,sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../mxnet/python"))
import numpy as np
import mxnet as mx
from read_embed import read_embed


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label, bucket_key):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names
        self.bucket_key = bucket_key

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class zhihu_iter(mx.io.DataIter):
    def __init__(self, question_set_path,
            question_topic,
            topic_info='topic_info.txt',
            char_embed_path='./char_embedding.txt',
            word_embed_path='./word_embedding.txt',
            embed_mode = 0,
            batch_size=32):
        with open(question_set_path) as f:
            self.raw_questions = f.readlines()
        with open(question_topic) as f:
            self.question_topic = f.readlines()
        with open(topic_info) as f:
            self.topic_info = f.readlines()
            self.topic_encode = {v.split('\t', 1)[0] : i+1 for i,v in enumerate(self.topic_info)}
        
        self.batch_size = batch_size
        self.embed_mode = embed_mode

        if os.path.exists('bucket_cache.txt'):
            buckets = open('bucket_cache.txt').readlines()
            buckets = [v.strip() for v in buckets]
            self.create_buckets(buckets)
        else:
            self.create_buckets()
            with open('bucket_cache.txt','w') as f:
                for v in self.buckets:
                    f.write(v+'\n')
        self.reset()
        keys = self.max_bucket_key.split(',')
        self.provide_data = []

        if embed_mode %2 == 0:
            self.char_dict, self.char_dict_size, self.char_dict_dim = read_embed(char_embed_path)
            self.provide_data +=[('tc_array',(self.batch_size,int(keys[0]),self.char_dict_dim)),\
                                 ('cc_array',(self.batch_size,int(keys[1]),self.char_dict_dim))] 
        if embed_mode > 0:
            self.word_dict, self.word_dict_size, self.word_dict_dim = read_embed(word_embed_path)
            if embed_mode %2 == 0:
                self.provide_data +=[('tw_array',(self.batch_size,int(keys[2]),self.word_dict_dim)),\
                                     ('cw_array',(self.batch_size,int(keys[3]),self.word_dict_dim))] 
            else:
                self.provide_data +=[('tw_array',(self.batch_size,int(keys[0]),self.word_dict_dim)),\
                                     ('cw_array',(self.batch_size,int(keys[1]),self.word_dict_dim))] 

        self.provide_label = [('label', (self.batch_size, len(self.topic_info) + 1))]



    def create_buckets(self, buckets=None):
        if buckets is None:
            self.buckets = self.default_buckets()
        else:
            self.buckets = buckets
            self.buckets_num = len(buckets)

        bucket_filter = np.asarray([[int(v) for v in k.split(',')] for k in self.buckets])
        self.bucket_samples_inds = {k:[] for k in self.buckets}

        for idx, v in enumerate(self.raw_questions):
            values = v.split()
            if len(values) < 3:
                continue
            tc = len(values[1].split(','))
            tw = len(values[2].split(','))
            if len(values) > 3:
                if values[3][0]=='c':
                    cc = len(values[3].split(','))
                else:
                    cc = tc
                    cw = len(values[3].split(','))
                if len(values) == 5:
                    cw = len(values[4].split(','))
            else:
                cc = tc
                cw = tw
            
            if self.embed_mode % 2 == 0:
                tc_bucket = min(np.where(bucket_filter[:,0] >= tc)[0])
                cc_bucket = min(np.where(bucket_filter[:,1] >= cc)[0])
                if self.embed_mode > 0:
                    tw_bucket = min(np.where(bucket_filter[:,2] >= tw)[0])
                    cw_bucket = min(np.where(bucket_filter[:,3] >=cw )[0])
                    bucket_id = max([tc_bucket, cc_bucket, tw_bucket, cw_bucket])
                else:
                    bucket_id = max([tc_bucket, cc_bucket])
            else:
                tw_bucket = min(np.where(bucket_filter[:,0] >= tw)[0])
                cw_bucket = min(np.where(bucket_filter[:,1] >= cw)[0])
                bucket_id = max([tw_bucket, cw_bucket])
            self.bucket_samples_inds[self.buckets[bucket_id]].append(idx)
        batch_dist = np.asarray([len(v)/self.batch_size for v in self.bucket_samples_inds.values()])
        self.max_bucket_key = self.buckets[max(np.where(batch_dist > 0)[0])]
        self.bucket_order = np.hstack((np.ones(v,dtype=int)*i for i,v in enumerate(batch_dist)))
    
    def default_buckets(self):
        self.buckets_num = 10
        tc_length = {}
        cc_length = {}
        tw_length = {}
        cw_length = {}

        for v in self.raw_questions:
            values = v.split()
            if len(values) < 3:
                continue
            tc = values[1].split(',')
            tw = values[2].split(',')
            if tc_length.has_key(len(tc)):
                tc_length[len(tc)] += 1
            else:
                tc_length[len(tc)] = 1
            if tw_length.has_key(len(tw)):
                tw_length[len(tw)] += 1
            else:
                tw_length[len(tw)] = 1
            if len(values) > 3:
                if values[3][0]=='c':
                    cc = values[3].split(',')
                    if cc_length.has_key(len(cc)):
                        cc_length[len(cc)] += 1
                    else:
                        cc_length[len(cc)] = 1
                else:
                    cw = values[3].split(',')
                    if cw_length.has_key(len(cw)):
                        cw_length[len(cw)] += 1
                    else:
                        cw_length[len(cw)] = 1
                if len(values) == 5:
                    cw = values[4].split(',')
                    if cw_length.has_key(len(cw)):
                        cw_length[len(cw)] += 1
                    else:
                        cw_length[len(cw)] = 1
        max_tc = max(tc_length.keys())
        max_cc = max(cc_length.keys())
        max_tw = max(tw_length.keys())
        max_cw = max(cw_length.keys())

        if self.embed_mode % 2 == 0:
            if self.embed_mode > 0:
                buckets = [','.join([str(max_tc*(i+1)/self.buckets_num),
                            str(max_cc*(i+1)/self.buckets_num),
                            str(max_tw*(i+1)/self.buckets_num),
                            str(max_cw*(i+1)/self.buckets_num)]) 
                            for i in range(self.buckets_num)]
            else:
                buckets = [','.join([str(max_tc*(i+1)/self.buckets_num),
                            str(max_cc*(i+1)/self.buckets_num)])
                            for i in range(self.buckets_num)]
        else:
            buckets = [','.join([str(max_tw*(i+1)/self.buckets_num),
                        str(max_cw*(i+1)/self.buckets_num)]) 
                        for i in range(self.buckets_num)]
        return buckets

    def reset(self):
        self.bucket_order = np.random.permutation(self.bucket_order)
        self.bucket_offset = [0 for _ in self.buckets]
        for k,v in self.bucket_samples_inds.items():
            self.bucket_samples_inds[k] = np.random.permutation(v)
        


    def __iter__(self):
        for idx in self.bucket_order:
            bucket_key = self.buckets[idx]
            inds= self.bucket_samples_inds[bucket_key] \
                    [self.bucket_offset[idx]:self.bucket_offset[idx]+self.batch_size]
            shapes= [(self.batch_size, int(v)) for v in bucket_key.split(',')]
            if len(shapes) == 4:
                tc_array = np.zeros(shapes[0]+(self.char_dict_dim,))
                cc_array = np.zeros(shapes[1]+(self.char_dict_dim,))
                tw_array = np.zeros(shapes[2]+(self.word_dict_dim,))
                cw_array = np.zeros(shapes[3]+(self.word_dict_dim,))
            elif self.embed_mode  %2 == 0:
                tc_array = np.zeros(shapes[0]+(self.char_dict_dim,))
                cc_array = np.zeros(shapes[1]+(self.char_dict_dim,))
            else:
                tw_array = np.zeros(shapes[0]+(self.word_dict_dim,))
                cw_array = np.zeros(shapes[1]+(self.word_dict_dim,))
            #print '*'*20
            #print shapes,bucket_key,tw_array.shape, cw_array.shape
            
            label = np.zeros((self.batch_size, len(self.topic_encode)+1))
            for i,ind in enumerate(inds):
                values = self.raw_questions[ind].split()
                tc = values[1]
                tw = values[2]
                if len(values) > 3:
                    if values[3][0]=='c':
                        cc = values[3]
                    else:
                        cc = tc
                        cw = values[3]
                    if len(values) == 5:
                        cw = values[4]
                else:
                    cc = tc
                    cw = tw

                if self.embed_mode  %2 == 0:
                    for j, v in enumerate(tc.split(',')):
                        tc_array[i,j] = self.char_dict[v]
                    for j, v in enumerate(cc.split(',')):
                        cc_array[i,j] = self.char_dict[v]
                if self.embed_mode > 0:
                    for j, v in enumerate(tw.split(',')):
                        tw_array[i,j] = self.word_dict[v]
                    for j, v in enumerate(cw.split(',')):
                        cw_array[i,j] = self.word_dict[v]

                top = self.question_topic[ind].split()[1].split(',')
                for t in top:
                    label[i,self.topic_encode[t]] = 1
            data_name = []
            data = []
            if self.embed_mode  %2 == 0:
                data_name += ['tc_array', 'cc_array']
                data += [mx.nd.array(tc_array), mx.nd.array(cc_array)]
            if self.embed_mode > 0:
                data_name += ['tw_array', 'cw_array']
                data += [mx.nd.array(tw_array), mx.nd.array(cw_array)]
            label = [mx.nd.array(label)]
            label_name = ['label']
            #print bucket_key, data
            yield SimpleBatch(data_name, data, label_name, label, bucket_key)
        raise StopIteration

if __name__ == '__main__':
    ziter = zhihu_iter('tidy_question_train_set.txt','tidy_question_topic_train_set.txt',embed_mode=1)
    #ziter.reset()
    for i in ziter:
        print i.provide_data
