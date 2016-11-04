import mxnet as mx

def symbol(num_res):
    data = mx.sym.Variable('data')
    weight_var = {}
    bias_var = {}
    convs = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1']
    for k in convs:
        weight_var[k] = mx.sym.Variable(k+'_weight')
    for k in convs:
        bias_var[k] = mx.sym.Variable(k+'_bias')
    out = vgg_symbol(data, weight_var, bias_var)
    for i in range(num_res-1):
        data = mx.sym.Pooling(data=data, kernel=(2,2), stride=(2,2), pad=(0,0), pool_type='avg')
        out = mx.sym.Group([out, vgg_symbol(data, weight_var, bias_var)])
    return out

def vgg_symbol(data, weight, bias):
    def conv(name, data, num_filter, pad, kernel, stride, no_bias, workspace):
        return mx.sym.Convolution(name=name, data=data, weight=weight[name], bias=bias[name], num_filter=num_filter, pad=pad, kernel=kernel, stride=stride, no_bias=no_bias, workspace=workspace)
    conv1_1 = conv(name='conv1_1', data=data , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1 , act_type='relu')
    conv1_2 = conv(name='conv1_2', data=relu1_1 , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2 , act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu1_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv2_1 = conv(name='conv2_1', data=pool1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1 , act_type='relu')
    conv2_2 = conv(name='conv2_2', data=relu2_1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2 , act_type='relu')
    pool2 = mx.symbol.Pooling(name='pool2', data=relu2_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv3_1 = conv(name='conv3_1', data=pool2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1 , act_type='relu')
    return relu3_1

class AssignPatch(mx.operator.NDArrayOp):
    def __init__(self):
        super(AssignPatch, self).__init__(False)
        self.fwd_kernel = None
        self.bwd_kernel = None

    def list_arguments(self):
        return ['nn', 'source']

    def list_outputs(self):
        return ['target']

    def infer_shape(self, in_shape):
        nn_shape = in_shape[0]
        source_shape = in_shape[1]
        return in_shape, [[1,source_shape[1], nn_shape[0]+source_shape[2]-1, nn_shape[1]+source_shape[3]-1]]

    def forward(self, in_data, out_data):
        nn, source = in_data
        target = out_data[0]
        if self.fwd_kernel is None:
            self.fwd_kernel = mx.rtc('assignpatch', [('nn', nn), ('source', source)], [('target', target)], """
                int target_idx = threadIdx.x*target_dims[3]*target_dims[2]+blockIdx.x*target_dims[3]+blockIdx.y;
                int source_idx = nn[blockIdx.x*nn_dims[1]+blockIdx.y]*source_dims[1]*source_dims[2]*source_dims[3]
                                    + threadIdx.x*source_dims[2]*source_dims[3];
                for (int i = 0; i < source_dims[2]; i++){
                    for (int j = 0; j < source_dims[3]; j++){
                        atomicAdd(target+target_idx, source[source_idx]);
                        target_idx++;
                        source_idx++;
                    }
                    target_idx += target_dims[3]-source_dims[3];
                }
            """)
        self.fwd_kernel.push([nn, source], [target], (target.shape[2]-source.shape[2]+1, target.shape[3]-source.shape[3]+1, 1), (source.shape[1],1,1))


def assign_symbol():
    v_source = mx.sym.Variable('source')
    v_nn = mx.sym.Variable('nn')
    assign = AssignPatch()
    assign_symbo = assign(source=v_source, nn=v_nn)
    return assign_symbo
