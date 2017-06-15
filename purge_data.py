from read_embed import read_embed
char_embed_path='./char_embedding.txt'
word_embed_path='./word_embedding.txt'
word_dict,_,_ = read_embed(word_embed_path)
char_dict,_,_ = read_embed(char_embed_path)
char_count={k:0 for k in char_dict}
word_count={k:0 for k in word_dict}

f = open('question_topic_train_set.txt')
question_topic = f.readlines()
f = open('question_train_set.txt')
raw_questions = f.readlines()
f_tidy_question = 'tidy_question_topic_train_set.txt'
f_tidy_topic = 'tidy_question_train_set.txt'
for raw_value, raw_label in zip(raw_questions, question_topic):
    value = raw_value.split()
    if len(value) < 3:
        print value
    if len(value)>1:
        tc = value[1].split(',')
    if len(value)>2:
        tw = value[2].split(',')
    if len(value)>3:
        cc = value[3].split(',')
    if len(value)>4:
        cw = value[4].split(',')

    for k in tc:
        if char_count.has_key(k):
            char_count[k] += 1
        else:
            char_count[k] = 1
    for k in cc:
        if char_count.has_key(k):
            char_count[k] += 1
        else:
            char_count[k] = 1
    for k in tw:
        if word_count.has_key(k):
            word_count[k] += 1
        else:
            word_count[k] = 1
    for k in cw:
        if word_count.has_key(k):
            word_count[k] += 1
        else:
            word_count[k] = 1

with open('char_count.txt','w') as f:
    for k,v in char_count.items():
        f.write(k+','+str(v)+'\n')

with open('word_count.txt','w') as f:
    for k,v in word_count.items():
        f.write(k+','+str(v)+'\n')
