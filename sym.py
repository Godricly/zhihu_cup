
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
    tc_concat = mx.sym.swapaxes(tc_concat, 0, 1)
    cc_concat = mx.sym.swapaxes(cc_concat, 0, 1)


def sym_gen_word(bucket_key):


def sym_gen_both(bucket_key):


    key = bucket_key.split(',')
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
    tc_concat = mx.sym.swapaxes(tc_concat, 0, 1)
    cc_concat = mx.sym.swapaxes(cc_concat, 0, 1)
    tw_concat = mx.sym.swapaxes(tw_concat, 0, 1)
    cw_concat = mx.sym.swapaxes(cw_concat, 0, 1)
    #ch_outputs = mx.sym.Concat(*[tc_concat, cc_concat])
    #wd_outputs = mx.sym.Concat(*[tw_concat, cw_concat])
    #title_outputs= mx.sym.Concat(*[tc_concat, tw_concat])
    #content_outputs= mx.sym.Concat(*[cc_concat, cw_concat])
    #ch_outputs = fc_module(ch_outputs, 'ch_', num_hidden = 2000)
    #wd_outputs = fc_module(wd_outputs, 'wd_', num_hidden = 2000)
    #title_outputs = fc_module(title_outputs, 'title_', num_hidden = 2000)
    #content_outputs = fc_module(content_outputs, 'content_', num_hidden = 2000)
    #feature = mx.sym.Concat(*[ch_outputs, wd_outputs, title_outputs, content_outputs])
    feature = mx.sym.Concat(*[tc_concat, cc_concat, tw_concat, cw_concat])
    feature = fc_module(feature, 'feature', num_hidden=4000)
    feature = mx.sym.FullyConnected(data=feature, num_hidden=num_class, name='fc1')
    loss = mx.sym.LogisticRegressionOutput(feature, label, name='regression')
    return loss

if __name__ == '__main__':
    sym = sym_gen(100,100, 100, 100)
    batch_size = 32
    dim = 256
    length = 100
    shapes = sym.infer_shape_partial(tc_array=(batch_size,length,dim),
                                     cc_array=(batch_size,length,dim),
                                     tw_array=(batch_size,length,dim),
                                     cw_array=(batch_size,length,dim),
                                     label=(batch_size,2000))
    names = sym.list_arguments()
    for name, shape in zip(names, shapes[0]):
        print name, shape
