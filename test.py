import os,sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../mxnet/python"))
import numpy as np
import mxnet as mx
from iter import zhihu_iter
batch_size =1
ziter = zhihu_iter('tidy_question_eval_set.txt',batch_size=batch_size,embed_mode=1)


num_layers = 1 
num_class = 2000
num_hidden = 512 
tw_cell = mx.rnn.FusedRNNCell(num_hidden, num_layers=num_layers, bidirectional=True, mode='lstm', prefix ='tw_')
cw_cell = mx.rnn.FusedRNNCell(num_hidden, num_layers=num_layers, bidirectional=True, mode='lstm', prefix ='cw_')

def fc_module(data, prefix, num_hidden=256):
    with mx.name.Prefix(prefix):
        fc1  = mx.sym.FullyConnected(data=data, num_hidden=num_hidden, name='fc1')
        relu_fc1 = mx.sym.Activation(data=fc1, act_type='relu', name='relu_fc1')
        return relu_fc1

data_name = [i[0] for i in ziter.provide_data]
label_name = [i[0] for i in ziter.provide_label]
def sym_gen_word(bucket_key):
    key = bucket_key.split(',')
    tw_length = int(key[0])
    cw_length = int(key[1])
    tw_data = mx.sym.Variable('tw_array')
    cw_data = mx.sym.Variable('cw_array')
    label   = mx.sym.Variable('label')
    tw_slices = list(mx.symbol.SliceChannel(data=tw_data, axis=1, num_outputs=tw_length, squeeze_axis=True, name='tw_slice'))
    cw_slices = list(mx.symbol.SliceChannel(data=cw_data, axis=1, num_outputs=cw_length, squeeze_axis=True, name='cw_slice'))
    tw_concat, _ = tw_cell.unroll(tw_length, inputs = tw_slices, merge_outputs=True, layout='TNC')
    cw_concat, _ = cw_cell.unroll(cw_length, inputs = cw_slices, merge_outputs=True, layout='TNC')
    tw_concat = mx.sym.transpose(tw_concat, (1, 2, 0)) 
    cw_concat = mx.sym.transpose(cw_concat, (1, 2, 0)) 
    tw_concat = mx.sym.Pooling(tw_concat, kernel=(1,), global_pool = True, pool_type='max')
    cw_concat = mx.sym.Pooling(cw_concat, kernel=(1,), global_pool = True, pool_type='max')
    feature = mx.sym.Concat(*[tw_concat, cw_concat], name= 'concat')
    feature = fc_module(feature, 'fc1', num_hidden=1024)
    feature = fc_module(feature, 'fc2', num_hidden=1024)
    feature = mx.sym.Dropout(feature, p=0.5)
    feature = fc_module(feature, 'feature', num_hidden=2000)
    loss = mx.sym.LogisticRegressionOutput(feature, label=label, name='regression')
    return loss, data_name, label_name

#mod = mx.module.BucketingModule(sym_gen_word, default_bucket_key=ziter.max_bucket_key,context=mx.gpu(1),data_names=data_name, label_names=label_name)
print 'wow'
mod = mx.module.BucketingModule(sym_gen_word, default_bucket_key=ziter.max_bucket_key,context=mx.context.gpu(7))
print 'wow'
_, arg_param, aux_param = mx.rnn.load_rnn_checkpoint([tw_cell,cw_cell],'model/zhihu', 9)
mod.bind(ziter.provide_data, ziter.provide_label,for_training=False)
mod.set_params(arg_params=arg_param, aux_params=aux_param)

for db in ziter:
    mod.forward(db, is_train=False)
    mod.get_outputs()[0].wait_to_read()
    print np.max(mod.get_outputs()[0].asnumpy())


