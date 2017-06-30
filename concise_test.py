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


f = open('question_eval_set.txt')
raw_questions = f.readlines()
f_tidy_question = open('tidy_question_eval_set.txt','w')

for raw_value in raw_questions:
    value = raw_value.split()
    if len(value) < 3:
        continue
    #f_tidy_question.write(value[0])
    tc = value[1].split(',')
    tc = [v for v in tc if char.has_key(v)]
    tc = ','.join(tc)
    #f_tidy_question.write('\t'+tc)
    tw = value[2].split(',')
    tw = [v for v in tw if word.has_key(v)]
    tw = ','.join(tw)
    #f_tidy_question.write('\t'+tw)

    if len(tc)==0 or len(tw) ==0:
        continue
    write_line = '\t'.join([value[0], tc, tw])

    if len(value)>3:
        cc = value[3].split(',')
        cc = [v for v in cc if char.has_key(v)]
        cc = ','.join(cc)
        write_line += '\t'+cc
    if len(value)>4:
        cw = value[4].split(',')
        cw = [v for v in cw if word.has_key(v)]
        cw = ','.join(cw)
        write_line += '\t'+cw
    write_line += '\n'
    f_tidy_question.write(write_line)

f_tidy_question.close()

