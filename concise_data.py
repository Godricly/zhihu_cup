import mxnet as mx
from read_embed import read_embed
char_raw = open('sorted_char_count.txt').readlines()
word_raw = open('sorted_word_count.txt').readlines()
char_raw = [v.strip().split(',') for v in char_raw]
word_raw = [v.strip().split(',') for v in word_raw]
char = [v[0] for v in char_raw]
char = {k:0 for k in char}
word = [v[0] for v in word_raw]
word = {k:0 for k in word}

char_embed_path='./char_embedding.txt'
word_embed_path='./word_embedding.txt'
word_dict,_,_ = read_embed(word_embed_path)
char_dict,_,_ = read_embed(char_embed_path)

word = {k:0 for k in word if word_dict.has_key(k)}
char = {k:0 for k in char if char_dict.has_key(k)}


f = open('question_topic_train_set.txt')
question_topic = f.readlines()
f = open('question_train_set.txt')
raw_questions = f.readlines()
f_tidy_question = open('tidy_question_train_set.txt','w')
f_tidy_topic = open('tidy_question_topic_train_set.txt','w')

tc_length = {i:0 for i in range(10000)}
cc_length = {i:0 for i in range(30000)}
tw_length = {i:0 for i in range(1000)}
cw_length = {i:0 for i in range(4000)}




for raw_value, raw_label in zip(raw_questions, question_topic):
    value = raw_value.split()
    if len(value) < 3:
        continue
    #f_tidy_question.write(value[0])
    tc = value[1].split(',')
    tc = [v for v in tc if char.has_key(v)]
    tc_length[len(tc)] +=1
    tc = ','.join(tc)
    #f_tidy_question.write('\t'+tc)
    tw = value[2].split(',')
    tw = [v for v in tw if word.has_key(v)]
    tw_length[len(tw)] +=1
    tw = ','.join(tw)
    #f_tidy_question.write('\t'+tw)

    if len(tc)==0 or len(tw) ==0:
        continue
    write_line = '\t'.join([value[0], tc, tw])

    if len(value)>3:
        cc = value[3].split(',')
        cc = [v for v in cc if char.has_key(v)]
        cc_length[len(cc)] +=1
        cc = ','.join(cc)
        write_line += '\t'+cc
    if len(value)>4:
        cw = value[4].split(',')
        cw = [v for v in cw if word.has_key(v)]
        cw_length[len(cw)] +=1
        cw = ','.join(cw)
        write_line += '\t'+cw
    write_line += '\n'
    f_tidy_question.write(write_line)
    f_tidy_topic.write(raw_label)

f_tidy_question.close()
f_tidy_topic.close()

with open('tc_length.txt','w') as f:
    for k,v in tc_length.items():
        f.write(str(k)+','+str(v)+'\n')
with open('cc_length.txt','w') as f:
    for k,v in cc_length.items():
        f.write(str(k)+','+str(v)+'\n')
with open('tw_length.txt','w') as f:
    for k,v in tw_length.items():
        f.write(str(k)+','+str(v)+'\n')
with open('cw_length.txt','w') as f:
    for k,v in cw_length.items():
        f.write(str(k)+','+str(v)+'\n')
