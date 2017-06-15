import os,sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../mxnet/python"))
import mxnet as mx
import numpy as np


class Debug(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        f = open('s.txt','w')
        value = in_data[0].asnumpy()
        for i in range(value.size):
            f.write('%1.6f\n'%(value.flat[i]))
        f.close()
        #x = in_data[0].asnumpy()
        #for i in range(144):
        #    print(x[i])
        self.assign(out_data[0],req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0],req[0], out_grad[0])

@mx.operator.register('debug')
class DebugProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(DebugProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        print in_shape[0]
        return [in_shape[0]],[in_shape[0]],[]

    def create_operator(self,ctx, shapes, dtypes):
        return Debug()

def fc_module(data, prefix, num_hidden=256):
    with mx.name.Prefix(prefix):
        fc1  = mx.sym.FullyConnected(data=data, num_hidden=num_hidden, name='fc1')
        relu_fc1 = mx.sym.Activation(data=fc1, act_type='relu', name='relu_fc1')
        return relu_fc1

def sym_gen_char(bucket_key):
    num_layers = 1
    num_class = 2000
    num_hidden = 512
    key = bucket_key.split(',')
    tc_length = int(key[0])
    cc_length = int(key[1])
    tc_data = mx.sym.Variable('tc_array')
    cc_data = mx.sym.Variable('cc_array')
    label   = mx.sym.Variable('label')
    tc_cell = mx.rnn.FusedRNNCell(num_hidden, num_layers=num_layers, bidirectional=True, mode='lstm', prefix ='tc_')
    cc_cell = mx.rnn.FusedRNNCell(num_hidden, num_layers=num_layers, bidirectional=True, mode='lstm', prefix ='cc_')
    tc_slices = list(mx.symbol.SliceChannel(data=tc_data, axis=1, num_outputs=tc_length, squeeze_axis=True, name='tc_slice'))
    cc_slices = list(mx.symbol.SliceChannel(data=cc_data, axis=1, num_outputs=cc_length, squeeze_axis=True, name='cc_slice'))
    tc_concat, _ = tc_cell.unroll(tc_length, inputs = tc_slices, merge_outputs=True, layout='TNC')
    cc_concat, _ = cc_cell.unroll(cc_length, inputs = cc_slices, merge_outputs=True, layout='TNC')
    tc_concat = mx.sym.transpose(tc_concat, (1, 2, 0))
    cc_concat = mx.sym.transpose(cc_concat, (1, 2, 0))
    tc_concat = mx.sym.Pooling(tc_concat, kernel=(1,), global_pool = True, pool_type='max')
    cc_concat = mx.sym.Pooling(cc_concat, kernel=(1,), global_pool = True, pool_type='max')
    feature = mx.sym.Concat(*[tc_concat, cc_concat], name= 'concat')
    feature = mx.sym.Dropout(feature, p=0.5)
    feature = fc_module(feature, 'feature', num_hidden=2000)
    loss = mx.sym.LogisticRegressionOutput(feature, label=label, name='regression')
    return loss


def sym_gen_word(bucket_key):
    num_layers = 1
    num_class = 2000
    num_hidden = 512
    key = bucket_key.split(',')
    tw_length = int(key[0])
    cw_length = int(key[1])
    tw_data = mx.sym.Variable('tw_array')
    cw_data = mx.sym.Variable('cw_array')
    label   = mx.sym.Variable('label')
    tw_cell = mx.rnn.FusedRNNCell(num_hidden, num_layers=num_layers, bidirectional=True, mode='lstm', prefix ='tw_')
    cw_cell = mx.rnn.FusedRNNCell(num_hidden, num_layers=num_layers, bidirectional=True, mode='lstm', prefix ='cw_')
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
    data_name = ['tw_array', 'cw_array']
    label_name = ['label']
    return loss, data_name, label_name


def sym_gen_both(bucket_key):
    num_layers = 1
    num_class = 2000
    num_hidden = 512
    key = bucket_key.split(',')
    tc_length = int(key[0])
    cc_length = int(key[1])
    tw_length = int(key[2])
    cw_length = int(key[3])
    tc_data = mx.sym.Variable('tc_array')
    cc_data = mx.sym.Variable('cc_array')
    tw_data = mx.sym.Variable('tw_array')
    cw_data = mx.sym.Variable('cw_array')
    label   = mx.sym.Variable('label')
    tc_cell = mx.rnn.FusedRNNCell(num_hidden, num_layers=num_layers, bidirectional=True, mode='lstm', prefix ='tc_')
    cc_cell = mx.rnn.FusedRNNCell(num_hidden, num_layers=num_layers, bidirectional=True, mode='lstm', prefix ='cc_')
    tw_cell = mx.rnn.FusedRNNCell(num_hidden, num_layers=num_layers, bidirectional=True, mode='lstm', prefix ='tw_')
    cw_cell = mx.rnn.FusedRNNCell(num_hidden, num_layers=num_layers, bidirectional=True, mode='lstm', prefix ='cw_')
    tc_slices = list(mx.symbol.SliceChannel(data=tc_data, axis=1, num_outputs=tc_length, squeeze_axis=True, name='tc_slice'))
    cc_slices = list(mx.symbol.SliceChannel(data=cc_data, axis=1, num_outputs=cc_length, squeeze_axis=True, name='cc_slice'))
    tw_slices = list(mx.symbol.SliceChannel(data=tw_data, axis=1, num_outputs=tw_length, squeeze_axis=True, name='tw_slice'))
    cw_slices = list(mx.symbol.SliceChannel(data=cw_data, axis=1, num_outputs=cw_length, squeeze_axis=True, name='cw_slice'))
    tc_concat, _ = tc_cell.unroll(tc_length, inputs = tc_slices, merge_outputs=True, layout='TNC')
    cc_concat, _ = cc_cell.unroll(cc_length, inputs = cc_slices, merge_outputs=True, layout='TNC')
    tw_concat, _ = tw_cell.unroll(tw_length, inputs = tw_slices, merge_outputs=True, layout='TNC')
    cw_concat, _ = cw_cell.unroll(cw_length, inputs = cw_slices, merge_outputs=True, layout='TNC')
    tc_concat = mx.sym.transpose(tc_concat, (1, 2, 0))
    cc_concat = mx.sym.transpose(cc_concat, (1, 2, 0))
    tw_concat = mx.sym.transpose(tw_concat, (1, 2, 0))
    cw_concat = mx.sym.transpose(cw_concat, (1, 2, 0))
    tc_concat = mx.sym.Pooling(tc_concat, kernel=(1,), global_pool = True, pool_type='max')
    cc_concat = mx.sym.Pooling(cc_concat, kernel=(1,), global_pool = True, pool_type='max')
    tw_concat = mx.sym.Pooling(tw_concat, kernel=(1,), global_pool = True, pool_type='max')
    cw_concat = mx.sym.Pooling(cw_concat, kernel=(1,), global_pool = True, pool_type='max')
    feature = mx.sym.Concat(*[tc_concat, cc_concat, tw_concat, cw_concat], name= 'concat')
    feature = mx.sym.Dropout(feature, p=0.5)
    feature = fc_module(feature, 'feature', num_hidden=2000)
    loss = mx.sym.LogisticRegressionOutput(feature, label=label, name='regression')
    return loss

if __name__ == '__main__':
    sym = sym_gen_both('100,33,11,21')
    batch_size = 32
    dim = 256
    length = 100
    shapes = sym.infer_shape_partial(tc_array=(batch_size,100,dim),
                                     cc_array=(batch_size,33,dim),
                                     tw_array=(batch_size,11,dim),
                                     cw_array=(batch_size,21,dim),
                                     label=(batch_size,2000))
    names = sym.list_arguments()
    for name, shape in zip(names, shapes[0]):
        print name, shape
