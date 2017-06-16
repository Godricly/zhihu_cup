with open('topic_info.txt') as f:
    pairs={}
    for l in f:
        l = l.split()[:2]
        node = l[0]
        parent = l[1] if l[1][0] !='c' and l[1][0] !='w' else None
        pairs[node]=parent


    with open('edge_list.csv','w') as f_edge:
        #f_edge.write('Node,Target,Source\n')
#        f_edge.write('Target,Source\n')
        for n,edges in pairs.items():
            if edges is not None:
                edges = edges.split(',')
                for e in edges:
                    f_edge.write(e+','+n+'\n')
#                    f_edge.write(n+','+n+','+e+'\n')
#            
#            else:
#                    f_edge.write(','+n+'\n')

