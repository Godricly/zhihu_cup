import numpy as np
def read_embed(fpath):
    with open(fpath) as f:
        head = f.readline()
        num, dim = head.split()
        num, dim = int(num), int(dim)
        remains = f.readlines()
        remains = [v.split(' ',1) for v in remains]
        embed_dict= {v[0]:np.asarray([float(i) for i in v[1].split()]) for v in remains}
        return embed_dict, num, dim

if __name__=='__main__':
    #read_embed('char_embedding.txt')
    print read_embed('word_embedding.txt')
