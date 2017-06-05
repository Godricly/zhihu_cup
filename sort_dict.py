import numpy as np
char_raw = open('char_count.txt').readlines()
word_raw = open('word_count.txt').readlines()
char_raw = [v.strip().split(',') for v in char_raw]
word_raw = [v.strip().split(',') for v in word_raw]
char = [v[0] for v in char_raw]
char_freq = [int(v[1]) for v in char_raw]
char_idx = np.argsort(char_freq)

zeros = sum(np.asarray(char_freq) ==0)
char_idx = char_idx[zeros:-5]

with open('sorted_char_count.txt','w') as f:
    for i in char_idx:
        if char_freq[i]<=5:
            continue
        f.write(char[i]+','+str(char_freq[i])+'\n')

word = [v[0] for v in word_raw]
word_freq = [int(v[1]) for v in word_raw]
word_idx = np.argsort(word_freq)
zeros = sum(np.asarray(word_freq) == 0)
word_idx = word_idx[zeros:-5]
with open('sorted_word_count.txt','w') as f:
    for i in word_idx:
        if(word_freq[i]<=5):
            continue
        f.write(word[i]+','+str(word_freq[i])+'\n')
