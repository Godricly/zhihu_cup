import os,sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../mxnet/python"))
import numpy as np
import mxnet as mx
from iter import zhihu_iter
batch_size =4
ziter = zhihu_iter('tiny_train.txt','tiny_topic.txt',batch_size=batch_size,embed_mode=1)


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
    feature = mx.sym.Dropout(feature, p=0.5)
    feature = fc_module(feature, 'feature', num_hidden=2000)
    loss = mx.sym.LogisticRegressionOutput(feature, label=label, name='regression')
    return loss, data_name, label_name

#mod = mx.module.BucketingModule(sym_gen_word, default_bucket_key=ziter.max_bucket_key,context=mx.gpu(1),data_names=data_name, label_names=label_name)
mod = mx.module.BucketingModule(sym_gen_word, default_bucket_key=ziter.max_bucket_key,context=mx.context.gpu(1))
import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
prefix='model/textline'
learning_rate = 0.01
optimizer_params={'learning_rate': learning_rate,
                'clip_gradient': 10 }
monitor=mx.mon.Monitor(200, pattern='.*')



num_epoch = 10
print 'fit begin'
mod.fit(train_data=ziter, eval_data=ziter,
          optimizer='adadelta',
          optimizer_params = optimizer_params,
          eval_metric = mx.metric.MSE(),
          num_epoch=num_epoch,
          initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
          batch_end_callback=mx.callback.Speedometer(batch_size, 50),
          epoch_end_callback = mx.rnn.do_rnn_checkpoint([tw_cell, cw_cell], prefix, 1))

