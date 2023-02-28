# References:
# https://cs.github.com/apache/tvm/blob/274d8fa964489e03ad97e684902063d935bf192b/python/tvm/relay/op/_tensor_grad.py
# https://captum.ai/api/_modules/captum/attr/_core/deep_lift.html

import tvm.relay as r

from utils import *
from cgutils import *

dl_eps = r.const(1e-10)

grad_override_fns = {}

def register_grad_override(op_name, grad_fn):
    grad_override_fns[op_name] = grad_fn

def get_grad_fn(call):
    op_name = get_op_name(call.op)
    if op_name in grad_override_fns:
        return grad_override_fns[op_name]
    ret = call.op.get_attr('FPrimalGradient')
    if not ret:
        raise NotImplementedError(f'No grad_fn for {desc_expr(call)}')
    return ret

def dl_backward(lish, delta_y, delta_xs, mp_out, verbose=False):
    """Calculates the DeepLIFT multipliers to the target value w.r.t. the layerish's inputs.
    :param lish: The layerish.
    :param delta_y: The change in this layer's output.
    :param delta_xs: The change in the inputs to this layer, in the order of
    the layer's parents (arguments).
    :param mp_out: The multiplier to the target value w.r.t. the output of this
    layerish, similar to "grad_out".
    Returns the calculated multipliers (similar to "grad_in") in accordance with delta_xs.
    """

    backward_fns = {
        (r.Var,): dl_backward_var,
        ('add', 'nn.bias_add', 'tanh'): dl_backward_simple,
        ('nn.dropout',): dl_backward_passthrough,
        ('nn.relu',): dl_backward_nonlinear,
        ('nn.dense',): dl_backward_dense,
        ('nn.conv2d',): dl_backward_conv2d,
        ('reshape',): dl_backward_reshape,
        ('nn.max_pool2d', 'nn.avg_pool2d', 'nn.global_avg_pool2d_grad'): dl_backward_basic_pool,
        ('nn.adaptive_avg_pool2d',): dl_backward_adaptive_avg_pool2d,
        ('concatenate',): dl_backward_concatenate,
        ('nn.batch_norm',): dl_backward_batch_norm,
    }

    if verbose:
        print(f'Layerish: {desc_expr(lish)}')

    core = layerish_core(lish)
    for types, fn in backward_fns.items():
        if is_any_type(core, *types):
            ret = fn(lish, delta_y, delta_xs, mp_out)
            if verbose:
                print(f'Backward: {desc_exprs(ret)}\n')
            return ret

    raise NotImplementedError(f'{desc_expr(lish)} is not supported by backward()')

def dl_backward_var(lish, delta_y, delta_xs, mp_out):
    return []

def dl_backward_simple(lish, delta_y, delta_xs, mp_out):
    return get_grad_fn(lish)(lish, mp_out)

def dl_backward_passthrough(lish, delta_y, delta_xs, mp_out):
    return [mp_out for _ in delta_xs]

def dl_backward_nonlinear(lish, delta_y, delta_xs, mp_out):
    delta_x = delta_xs[0]
    orig_grad = get_grad_fn(lish)(lish, mp_out)[0]
    mp_in = mp_out * delta_y / (delta_x + dl_eps)
    mp_in = r.where(r.abs(delta_x) < dl_eps, orig_grad, mp_in)
    return [mp_in]

def dl_backward_dense(lish, delta_y, delta_xs, mp_out):
    weight_t = r.transpose(lish.args[1])
    mp_in = r.nn.dense(mp_out, weight_t)
    return [mp_in]

def dl_backward_conv2d(lish, delta_y, delta_xs, mp_out):
    backward_data, _backward_weight = get_grad_fn(lish)(lish, mp_out)
    return [backward_data]

def dl_backward_reshape(lish, delta_y, delta_xs, mp_out):
    return [r.reshape_like(mp_out, delta_xs[0])]

def dl_backward_basic_pool(lish, delta_y, delta_xs, mp_out):
    delta_x = delta_xs[0]
    grad_fn = get_grad_fn(lish)
    pool_grad = grad_fn(lish, mp_out)[0]
    mp_in = grad_fn(
        lish, mp_out * delta_y
    )[0] / (delta_x + dl_eps)
    mp_in = r.where(r.abs(delta_x) < dl_eps, pool_grad, mp_in)
    return [mp_in]

def dl_backward_adaptive_avg_pool2d(lish, delta_y, delta_xs, mp_out):
    delta_x = delta_xs[0]
    ish, isw = delta_x.checked_type.concrete_shape[2:]
    osh, osw = lish.checked_type.concrete_shape[2:]
    kernel_size = ((ish + osh - 1) // osh, (isw + osw - 1) // osw)
    strides = kernel_size
    padding = (kernel_size[0] * osh - ish, kernel_size[1] * osw - isw)
    attrs = {
        'pool_size': kernel_size,
        'strides': strides,
        'padding': padding,
        'count_include_pad': False,
    }
    pool_grad = r.nn.avg_pool2d_grad(
        mp_out, delta_x, **attrs
    )
    mp_in = r.nn.avg_pool2d_grad(
        mp_out * delta_y, delta_x, **attrs
    ) / (delta_x + dl_eps)
    mp_in = r.where(r.abs(delta_x) < dl_eps, pool_grad, mp_in)
    return [mp_in]

def dl_backward_concatenate(lish, delta_y, delta_xs, mp_out):
    in_mps = get_grad_fn(lish)(lish, mp_out)[0]
    return [in_mps]

def dl_backward_batch_norm(lish, delta_y, delta_xs, mp_out):
    bn = layerish_core(lish)
    gamma, var, epsilon, scale = bn.args[1], bn.args[4], bn.attrs['epsilon'], bn.attrs['scale']
    epsilon = r.const(epsilon)
    denominator = r.sqrt(var + epsilon)
    numerator = gamma if scale else 1
    ndims_remaining = len(delta_xs[0].checked_type.concrete_shape[2:])
    # E.g. if input is (1, 123, 2, 2), grad is reshaped to (1, 123, 1, 1)
    local_grad = r.reshape(numerator / denominator, (1, -1) + (1,) * ndims_remaining)
    mp_in = local_grad * mp_out
    return [mp_in]

# We only implement this for the input data
@r.op.register_gradient('nn.batch_norm')
def batch_norm_grad(orig, grad):
    gamma, var, epsilon, scale = orig.args[1], orig.args[4], orig.attrs['epsilon'], orig.attrs['scale']
    epsilon = r.const(epsilon)
    denominator = r.sqrt(var + epsilon)
    numerator = gamma if scale else 1
    ndims_remaining = len(get_shape(orig.args[0])[2:])
    # E.g. if input is (1, 123, 2, 2), grad is reshaped to (1, 123, 1, 1)
    local_grad = r.reshape(numerator / denominator, (1, -1) + (1,) * ndims_remaining)
    return [local_grad * grad]

@r.op.register_gradient('nn.adaptive_avg_pool2d')
def adaptive_avg_pool2d_grad(orig, grad):
    data = orig.args[0]
    ish, isw = get_shape(data)[2:]
    osh, osw = get_shape(orig)[2:]
    kernel_size = ((ish + osh - 1) // osh, (isw + osw - 1) // osw)
    strides = kernel_size
    padding = (kernel_size[0] * osh - ish, kernel_size[1] * osw - isw)
    attrs = {
        'pool_size': kernel_size,
        'strides': strides,
        'padding': padding,
        'count_include_pad': False,
    }
    pool_grad = r.nn.avg_pool2d_grad(
        grad, data, **attrs
    )
    return [pool_grad]

# Override and make compatible with QNN
def bias_add_grad(orig, grad):
    data = orig.args[0]
    return [
        r.reshape_like(grad, data),
        r.sum(grad, orig.attrs.axis, keepdims=False, exclude=True),
    ]
register_grad_override('nn.bias_add', bias_add_grad)

@r.op.register_gradient('nn.pad')
def pad_grad(orig, grad):
    # For now we just handle the simplest case
    input_shape = get_shape(orig.args[0])
    output_shape = get_shape(orig)
    if input_shape != output_shape:
        raise NotImplementedError('Real padding is not yet implemented')
    return [grad]

def register_pass_through_grad(op_name):
    @r.op.register_gradient(op_name)
    def pass_through_grad(orig, grad):
        return [grad]

register_pass_through_grad('nn.dropout')
