import numpy as np
import mxnet as mx
from read_embed import read_embed
class zhihu_iter(mx.io.DataIter):
    def __init__(self, question_set_path,
            question_topic,
            topic_info='topic_info.txt',
            char_embed_path='./char_embedding.txt',
            word_embed_path='./word_embedding.txt',
            embed_mode = 0,
            batch_size=1):
        with open(question_set_path) as f:
            self.raw_questions = f.readlines()
        with open(question_topic) as f:
            self.question_topic = f.readlines()
        with open(topic_info) as f:
            self.topic_info = f.readlines()
            self.topic_encode = {v.split('\t', 1)[0] : i+1 for i,v in enumerate(self.topic_info)}
        
        self.batch_size = batch_size
        self.embed_mode = embed_mode
        self.create_buckets()
        if embed_mode %2 == 0:
            self.char_dict, self.char_dict_size, self.char_dict_dim = read_embed(char_embed_path)
        if embed_mode > 0:
            self.word_dict, self.word_dict_size, self.word_dict_dim = read_embed(word_embed_path)
        self.provide_data = [('data', (self.batch_size, ))]
        self.provide_label = [('label', (self.batch_size, len(self.topic_info) + 1))]


    def create_buckets(self, buckets=None):
        if buckets is None:
            self.buckets = self.default_buckets()
        else:
            self.buckets = buckets
            self.buckets_num = len(buckets)
        self.bucket_samples_inds = [[] for _ in self.buckets]

        for idx, v in enumerate(self.raw_questions):
            values = v.split()
            if len(values) < 3:
                continue
            tc = values[1].split(',')
            tw = values[2].split(',')
            if len(values) > 3:
                if values[3][0]=='c':
                    cc = values[3].split(',')
                else:
                    cc = tc
                    cw = values[3].split(',')
                if len(values) == 5:
                    cw = values[4].split(',')
            else:
                cc = tc
                cw = tw

                    
            


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
                self.max_bucket_key = ','.join([str(max_tc), str(max_cc), str(max_tw), str(max_cw)])
                buckets = [','.join([str(max_tc*(i+1)/self.buckets_num),
                            str(max_cc*(i+1)/self.buckets_num),
                            str(max_tw*(i+1)/self.buckets_num),
                            str(max_cw*(i+1)/self.buckets_num)]) 
                            for i in range(self.buckets_num)]
            else:
                self.max_bucket_key = ','.join([str(max_tc), str(max_cc)])
                buckets = [','.join([str(max_tc*(i+1)/self.buckets_num),
                            str(max_cc*(i+1)/self.buckets_num)])
                            for i in range(self.buckets_num)]
        else:
            self.max_bucket_key = ','.join([str(max_tw), str(max_cw)])
            buckets = [','.join([str(max_tw*(i+1)/self.buckets_num),
                        str(max_cw*(i+1)/self.buckets_num)]) 
                        for i in range(self.buckets_num)]
        return buckets

    def reset(self):
        self.order = np.random.permutation(len(self.raw_questions))

    def __iter__(self):
        max_tc = 0
        max_cc = 0
        max_tw = 0
        max_cw = 0
        for idx in self.order:
            raw_value = self.raw_questions[idx]
            raw_label = self.question_topic[idx]
            values = raw_value.split()
            labels = raw_label.split()[1].split(',')
            label = np.zeros((self.batch_size, len(self.topic_info) + 1))
            for l in labels:
                label[0, self.topic_encode[l]]=1
            label_name =['topic']
            label = [mx.nd.array(label)]
            # print len(values)

            title_char = values[1]
            title_word = values[2]
            if len(values) > 3:
                if values[3][0]=='c':
                    content_char = values[3]
                else:
                    content_word = values[3]
                    content_char = title_char
                if len(values) > 4:
                    content_word = values[4]
                else:
                    content_word = title_word
            else:
                # duplicating title as content
                content_char = title_char
                content_word = title_word

            title_char = title_char.split(',')
            title_word = title_word.split(',')
            content_char = content_char.split(',')
            content_word = content_word.split(',')
            data_name = []
            data = []
            bucket_key = ''


            if self.embed_mode  %2 == 0:
                tc_array = np.zeros((len(title_char), self.char_dict_dim))
                cc_array = np.zeros((len(content_char), self.char_dict_dim))
                max_tc = max(max_tc, len(title_char))
                max_cc = max(max_cc, len(content_char))
                for i, c in enumerate(title_char):
                    tc_array[i,:] = self.char_dict[c]
                for i, c in enumerate(content_char):
                    cc_array[i,:] = self.char_dict[c]
                data_name += ['tc_array', 'cc_array']
                data += [mx.nd.array(tc_array), mx.nd.array(cc_array)]
                bucket_key += str(tc_array.shape[0])+','+str(cc_array.shape[0])
                # print tc_array.shape, cc_array.shape

            if self.embed_mode > 0:
                tw_array = np.zeros((len(title_word), self.word_dict_dim))
                cw_array = np.zeros((len(content_word), self.word_dict_dim))
                max_tw = max(max_tw,len(title_word))
                max_cw = max(max_cw,len(content_word))
                for i, w in enumerate(title_word):
                    tw_array[i,:] = self.word_dict[w]
                for i, w in enumerate(content_word):
                    cw_array[i,:] = self.word_dict[w]
                data_name += ['tw_array', 'cw_array']
                data += [mx.nd.array(tw_array), mx.nd.array(cw_array)]
                if len(bucket_key) > 0:
                    bucket_key += ','
                bucket_key += str(tw_array.shape[0])+','+str(cw_array.shape[0])
                # print tw_array.shape, cw_array.shape
        print max_tc,max_cc,max_tw,max_cw




if __name__ == '__main__':
    ziter = zhihu_iter('tidy_question_train_set.txt','tidy_question_topic_train_set.txt',embed_mode=1)
    ziter.reset()
