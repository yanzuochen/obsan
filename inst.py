# References:
# https://tvm.apache.org/docs/reference/api/python/relay/index.html
# https://tvm.apache.org/docs/reference/api/python/relay/nn.html
# https://tvm.apache.org/docs/reference/langref/relay_op.html
# https://tvm.apache.org/docs/reference/api/python/topi.html

from math import prod
import numpy as np
import tvm.relay as r
from tvm.ir import IRModule
from tvm.ir.transform import PassContext
import tvm
from functools import reduce
from typing import Dict, List, NamedTuple
import time

from utils import *
from cgutils import *
import backward as b

gn_modes = {'gn1', 'gn2', 'gninf'}

# Delegated to their respective functions
delegated_modes = {'npc', 'npcb', 'npcv', 'gn1', 'gn2', 'gninf', 'odin'}

def lcov_basic_diff(lcov, base):
    """Preserves only the newly covered neurons in lcov in terms of base."""
    return thread_first(lcov,
                        (r.logical_xor, base),
                        (r.logical_and, lcov))

def extra_params_basic(exprs, nclasses):
    """Enables each call to receive a base coverage tensor for comparison."""
    shapef = lambda e: (get_shape(e)[1],)
    if nclasses:
        shapef = lambda e: (nclasses, get_shape(e)[1])
    return [r.var(f'__ep_{i}', shape=shapef(e), dtype='bool') for i, e in enumerate(exprs)]

def extra_params_dual_bounds(exprs, nclasses):
    """Returns the extra params for the upper and lower bounds for the neurons
    in each call."""
    shapef = lambda e: (2, get_shape(e)[1])
    if nclasses:
        shapef = lambda e: (nclasses, 2, get_shape(e)[1])
    return [r.var(f'__ep_{i}', shape=shapef(e), dtype=get_dtype(e))
            for i, e in enumerate(exprs)]

def lcov_type_xf_basic(t: r.TensorType):
    """For a layer with n neurons, returns the type of a bool 1-dim array of
    shape (n,)."""
    return r.TensorType((t.concrete_shape[1],), 'bool')

def lcovs2ocov_basic(lcovs, types, per_neuron_max=1):
    """This function handles cases where the layer coverage output is a vector
    indicating whether each neuron is covered.
    In some coverage criteria (e.g. NBC) the coverage value of a neuron may be
    larger than 1, and that's when you set the per_neuron_max param."""
    if not len(lcovs):
        return r.const(np.array([0.])), r.TensorType((1,))
    ncovered = reduce(r.add, [thread_first(lcov,
                                           (r.cast, 'float32'),
                                           r.sum,
                                           (r.expand_dims, 0))
                              for lcov in lcovs])
    # Here lcovs are one-dimensional
    nneurons = sum([t.concrete_shape[0] for t in types])
    return ncovered / r.const(nneurons * per_neuron_max, dtype='float32'), r.TensorType((1,))

class CovModeConfig:
    """An object storing the configuration of a coverage criterion.

    lcov_type_xf: Layer coverage type transformer. A fn that, given the return
    type (a TensorType) of a layer (expr), lcov_type_xf: returns the type of
    the coverage data for the layer.

    lout2lcov: Layer output to layer coverage function. A fn that, given the
    output of a layer, the shape (ninputs, nneurons), and the corresponding
    extra param, returns an expr of the coverage of the layer which should
    match the type above.

    extra_params_fn: A fn that, given a list of interesting exprs and the
    number of classes, returns a list of extra params to be appended to the
    model.

    lcovs2ocov: Layer coverages to overall covearge function. A fn that, given
    a list of layer coverage exprs and their types, returns lcovs2ocov: an expr
    for the overall coverage for the model and its type.

    epb_mode: Extra params builder mode. The name of the mode that should be
    used for building the extra params for this coverage criterion.
    """
    def __init__(self, mode, lout2lcov,
                 lcov_type_xf=lcov_type_xf_basic,
                 extra_params_fn=extra_params_basic,
                 lcovs2ocov=lcovs2ocov_basic,
                 epb_mode=None):
        self.mode = mode
        self.lcov_type_xf = lcov_type_xf
        self.lout2lcov = lout2lcov
        self.extra_params_fn = extra_params_fn
        self.lcovs2ocov = lcovs2ocov
        self.epb_mode = epb_mode

# TODO: Some lout2lcov functions may not be compatible with non-class-cond
# cases any more

# NC {{{

def lout2lcov_nc(lout: r.Expr, shape, extra_param):
    threshold = r.const(1.)  # TODO: ad hoc
    return thread_first(lout,
                        (r.greater, threshold),
                        (lcov_basic_diff, extra_param),
                        (r.any, 0))

modecfg_nc = CovModeConfig('NC', lout2lcov_nc, epb_mode='NC')

# }}}

# NCS {{{

def ncs_scale(lout: r.Expr):
    rmin, rmax = [r.const(x) for x in [0., 1.]]
    lout_max = r.max(lout, axis=-1, keepdims=True)
    lout_min = r.min(lout, axis=-1, keepdims=True)
    lout_std = r.cast(lout - lout_min, 'float32') / r.cast(lout_max - lout_min, 'float32')
    lout_scaled = lout_std * (rmax - rmin) + rmin
    return lout_scaled

def lout2lcov_ncs(lout: r.Expr, shape, extra_param):
    threshold = r.const(.9)  # TODO: ad hoc
    return thread_first(lout,
                        ncs_scale,
                        (r.greater, threshold),
                        (lcov_basic_diff, extra_param),
                        (r.any, 0))

modecfg_ncs = CovModeConfig('NCS', lout2lcov_ncs, epb_mode='NCS')

# }}}

# TopK {{{

def lout2lcov_topk(lout: r.Expr, shape, extra_param):
    nneurons = shape[1]
    k = 10  # TODO: ad hoc
    k = min(nneurons, k)  # TVM doesn't support dynamic shape... yet
    kidxes = r.topk(lout, k=k, axis=-1, ret_type='indices')
    return thread_first(lout,
                        r.shape_of,
                        (r.zeros, 'int32'),
                        (r.scatter, kidxes, r.ones_like(kidxes), -1),
                        (r.cast, 'bool'),
                        (lcov_basic_diff, extra_param),
                        (r.any, 0))

modecfg_topk = CovModeConfig('TopK', lout2lcov_topk, epb_mode='TopK')

# }}}

# RB {{{

def lcov_type_xf_rb(t: r.TensorType):
    # 2 x nneurons tensor for lower and upper bounds for each neuron
    return r.TensorType((2, t.concrete_shape[1]), dtype=t.dtype)

def lout2lcov_rb(lout: r.Expr, shape, extra_param):
    lows, highs = [f(lout, 0) for f in [r.min, r.max]]
    return r.stack([lows, highs], 0)

modecfg_rb = CovModeConfig('rb', lout2lcov_rb, lcov_type_xf=lcov_type_xf_rb)

# }}}

# Identity {{{

def lcov_type_xf_identity(t: r.TensorType):
    return t

def lout2lcov_identity(lout: r.Call, shape, extra_param):
    return lout

modecfg_id = CovModeConfig('id', lout2lcov_identity, lcov_type_xf=lcov_type_xf_identity)

# }}}

# NBC {{{

def lcov_type_xf_nbc(t: r.TensorType):
    return r.TensorType((t.concrete_shape[1],), 'int8')

def lout2lcov_nbc(lout: r.Expr, shape, extra_param):
    lows, highs = [r.take(extra_param, r.const(i), axis=-2) for i in range(2)]
    # TODO: Check dtype compatibility
    lows, highs = [r.cast_like(d, lout) for d in [lows, highs]]
    low_covered = thread_first(lout,
                               (r.less, lows),
                               (r.any, 0),
                               (r.cast, 'int8'))
    high_covered = thread_first(lout,
                                (r.greater, highs),
                                (r.any, 0),
                                (r.cast, 'int8'),
                                (r.left_shift, r.const(1, dtype='int8')))
    return low_covered + high_covered

def lcovs2ocov_nbc(lcovs, types):
    lcovs = [thread_first(lcov,
                          (r.right_shift, r.const(1, dtype='int8')),
                          (r.add, r.bitwise_and(lcov, r.const(1, dtype='int8'))))
             for lcov in lcovs]
    return lcovs2ocov_basic(lcovs, types, per_neuron_max=2)

modecfg_nbc = CovModeConfig('NBC', lout2lcov_nbc, lcov_type_xf=lcov_type_xf_nbc,
                            extra_params_fn=extra_params_dual_bounds,
                            lcovs2ocov=lcovs2ocov_nbc,
                            epb_mode='rb')

# }}}

# WNBC {{{

def lcov_type_xf_wnbc(t: r.TensorType):
    return r.TensorType((t.concrete_shape[1],), 'float32')

def lout2lcov_wnbc(lout: r.Expr, shape, extra_param):
    lows, highs = [r.take(extra_param, r.const(i), axis=-2) for i in range(2)]
    range_sizes = r.cast_like(highs - lows, lout)
    max_size = r.maximum(r.max(range_sizes), r.const(1e-6))
    weights = r.const(1.) - range_sizes / max_size
    low_covered = thread_first(lout,
                               (r.less, lows),
                               (r.any, 0),
                               (r.cast, 'float32'))
    high_covered = thread_first(lout,
                                (r.greater, highs),
                                (r.any, 0),
                                (r.cast, 'float32'))
    scores = thread_first(low_covered + high_covered,
                          (r.multiply, weights))
    return scores

modecfg_wnbc = CovModeConfig('WNBC', lout2lcov_wnbc, lcov_type_xf=lcov_type_xf_wnbc,
                             extra_params_fn=extra_params_dual_bounds,
                             epb_mode='rb')

# }}}

# KMN {{{

def lcov_type_xf_kmn(t: r.TensorType):
    # nneurons x (k+1)
    k = 10
    return r.TensorType((t.concrete_shape[1], k+1), dtype='bool')

def lout2lcov_kmn(lout: r.Expr, shape, extra_param):
    k = 10
    nneurons = shape[1]
    lows, highs = [r.take(extra_param, r.const(i), axis=-2) for i in range(2)]
    bounds_valid = highs > lows
    neurons_valid = thread_first(lout >= lows,
                                 (r.bitwise_and, lout <= highs),
                                 (r.bitwise_and, bounds_valid))
    div = thread_first(bounds_valid,
                       r.bitwise_not,
                       (r.cast, 'float32'),
                       (r.multiply, r.const(1e-6)),
                       (r.add, thread_first(bounds_valid,
                                            (r.cast, 'float32'),
                                            (r.multiply, highs - lows))))
    msec_idxes = thread_first((lout - lows) / div * r.const(k, 'float32'),
                              r.ceil,
                              (r.multiply, r.cast(neurons_valid, 'float32')),
                              (r.cast, 'int32'),
                              (r.clip, 0, k),  # TODO: Should we add this?
                              r.transpose)
    return thread_first(r.zeros((nneurons, k+1), 'int32'),
                        (r.scatter, msec_idxes, r.ones_like(msec_idxes), -1),
                        (r.cast, 'bool'))

modecfg_kmn = CovModeConfig('KMN', lout2lcov_kmn, lcov_type_xf=lcov_type_xf_kmn,
                            extra_params_fn=extra_params_dual_bounds,
                            epb_mode='rb')

# }}}

# Covar {{{

def lcov_type_xf_covar(t: r.TensorType):
    return r.TensorType((1,))

def extra_params_covar(exprs, nclasses):
    # For each expr, we have an (A+2)xA matrix, where A is the number of
    # neurons in the layer.
    # The first AxA matrix is the covariance matrix
    # Than we have a 1xA vector of means
    # And finally a 1x1 scalar of the number of samples processed
    assert not nclasses
    ret = []
    for i, e in enumerate(exprs):
        A = e.checked_type.concrete_shape[1]
        ret.append(r.var(f'__ep_{i}',
                         shape=(A + 2, A),
                         dtype=e.checked_type.dtype))
    return ret

def lcov_type_xf_cb(t: r.TensorType):
    # cb stands for covar builder, so each cov will have the same type
    # as the extra param above.
    A = t.concrete_shape[1]
    return r.TensorType((A + 2, A), dtype='float32')

def lout2lcov_cb(lout: r.Expr, shape, extra_param):
    # TODO: Deal with NaN's by replacing 0 with 1e-6?

    N, A = shape
    cN = r.const(N, dtype='float32')
    c1 = r.const(1.)

    ep_split = r.split(extra_param, (A, A+1), axis=0)
    ocovar = ep_split[0]
    omeans = ep_split[1]
    ocount = r.take(ep_split[2], r.const(0))

    means = r.mean(lout, axis=0, keepdims=True)  # 1xA
    tmp = lout - means
    tmp = thread_first(tmp,
                       r.transpose,
                       (r.nn.matmul, tmp),
                       (r.divide, cN))
    weight_V = cN / (cN + ocount)
    additional_V = thread_first(omeans - means,
                                r.transpose,
                                (r.nn.matmul, omeans - means),
                                (r.multiply, weight_V * (c1 - weight_V)))
    new_covar = (r.multiply(ocovar, c1 - weight_V) +
                 r.multiply(tmp, weight_V)) + additional_V
    new_means = (r.multiply(omeans, c1 - weight_V) +
                 r.multiply(means, weight_V))
    new_count = r.full(ocount + cN, (1, A))

    return r.concatenate([new_covar, new_means, new_count], axis=0)

def lout2lcov_covar(lout: r.Expr, shape, extra_param):
    A = shape[1]
    ep_split = r.split(extra_param, (A, A+1), axis=0)
    ocovar = ep_split[0]
    covar = thread_first(lout,
                         (lout2lcov_cb, shape, extra_param),
                         (r.split, (A, A+1), 0))[0]
    return thread_first(ocovar - covar,
                        r.abs,
                        r.sum,
                        (r.expand_dims, 0))

def lcovs2ocov_covar(lcovs, types):
    return reduce(r.add, lcovs), r.TensorType((1,))

modecfg_cb = CovModeConfig('cb', lout2lcov_cb, lcov_type_xf=lcov_type_xf_cb,
                           extra_params_fn=extra_params_covar)

modecfg_covar = CovModeConfig('covar', lout2lcov_covar, lcov_type_xf=lcov_type_xf_covar,
                              extra_params_fn=extra_params_covar,
                              lcovs2ocov=lcovs2ocov_covar,
                              epb_mode='cb')

# }}}

# Some modes will be treated specially by instrument_module()

# NPC actually has two rounds during recording: npcb (to get the ref-y's for
# each layer) and npc (to get the neurons ever activated by the training set).
modecfg_npc = CovModeConfig('npc', None, epb_mode='npc')
modecfg_npcb = CovModeConfig('npcb', None)
modecfg_npcv = CovModeConfig('npcv', None)

modecfg_gn1 = CovModeConfig('gn1', None)
modecfg_gn2 = CovModeConfig('gn2', None)
modecfg_gninf = CovModeConfig('gninf', None)

modecfg_odin = CovModeConfig('odin', None)

cov_mode_configs = {
    'NC': modecfg_nc,
    'NCS': modecfg_ncs,
    'TopK': modecfg_topk,
    'rb': modecfg_rb,
    'id': modecfg_id,
    'NBC': modecfg_nbc,
    'WNBC': modecfg_wnbc,
    'KMN': modecfg_kmn,
    'cb': modecfg_cb,
    'covar': modecfg_covar,
    'npc': modecfg_npc,
    'npcb': modecfg_npcb,
    'npcv': modecfg_npcv,
    'gn1': modecfg_gn1,
    'gn2': modecfg_gn2,
    'gninf': modecfg_gninf,
    'odin': modecfg_odin,
}

class InstModPack(NamedTuple):
    irmod: IRModule
    extra_param_vars: List[r.Var]
    output_defs: List[Dict]

class HybridInstContainer(NamedTuple):
    extra_param_vars: List[r.Var]
    cov_exprs: List[r.Expr]

def make_empty_hic():
    return HybridInstContainer(extra_param_vars=[], cov_exprs=[])

def instrument_module(mod, modecfg_or_name, overall_cov=False, class_cond=None,
                      hic: HybridInstContainer = None,
                      skipped_weights=None, skipped_neurons=None, skip_as_eps=False,
                      verbose=False, **kwargs):
    """Returns (instrumented_mod, extra_params, output_defs).
    If hic is not None, hybrid instrumentation will be enabled and hic will be
    used and updated. Note that we currently only support NBC+gn* (or something
    similar).
    If overall_cov is True, overall coverage is returned instead of per-layer
    results.
    If an op's weight is in skipped_weights, it won't be used for coverage calculation.
    For each op, its neuron outputs with indices in skipped_neurons[op_weight] won't
    be used for coverage calculation.
    If as_eps was True in get_ignored_components, enable skip_as_eps here."""

    if isinstance(modecfg_or_name, str):
        modecfg = cov_mode_configs[modecfg_or_name]
    else:
        modecfg = modecfg_or_name

    mode = modecfg.mode.lower()
    lcov_type_xf = modecfg.lcov_type_xf
    lout2lcov = modecfg.lout2lcov
    extra_params_fn = modecfg.extra_params_fn
    lcovs2ocov = modecfg.lcovs2ocov

    start_time = time.time()

    if class_cond is not None and mode in delegated_modes + {'covar'}:
        warn(f'class_cond option will be ignored for {mode} mode.')
        class_cond = None

    if mode in delegated_modes:
        if skipped_weights or skipped_neurons or skip_as_eps:
            warn(f'{mode} mode does not support skipped_weights, skipped_neurons, or skip_as_eps')
        if mode.startswith('npc'):
            ret = instrument_module_npc(mod, modecfg, overall_cov=overall_cov, verbose=verbose)
        elif mode.startswith('gn'):
            ret = instrument_module_gn(
                mod, modecfg, overall_cov=overall_cov, verbose=verbose, hic=hic, **kwargs
            )
        elif mode == 'odin':
            if not overall_cov:
                warn('odin mode does not support per-layer coverage')
            ret = instrument_module_odin(mod, verbose=verbose)
        else:
            assert False
        if verbose:
            print(ret.irmod)
        print(f'Instrumentation finished in {time.time() - start_time:.3f} seconds.')
        assert isinstance(ret, InstModPack)
        return ret

    if class_cond is None and mode != 'covar':
        class_cond = True  # Default to True

    if hic:
        assert not hic.extra_param_vars, 'For NBC+gn* hybrid instrumentation, run NBC first.'

    extra_params = None
    rettype = None
    @r.transform.function_pass(opt_level=1)
    def ftransform(func: r.Function, mod: IRModule, ctx: PassContext):
        nonlocal extra_params, rettype, overall_cov, skipped_weights, skipped_neurons

        if not func.same_as(mod["main"]):
            return func

        model_out = func.body
        nclasses, pred_labels = 0, None
        if class_cond:
            nclasses = get_shape(model_out)[1]
            pred_labels = r.argmax(model_out, axis=1, keepdims=True)

        interesting_exprs = []

        class InterestingExprCollector(r.ExprVisitor):
            def __init__(self):
                super().__init__()
                self.skipped_exprs = set()

            def visit_call(self, call):
                # Goal: Instrument conv2d and dense layers
                op_name = get_op_name(call.op)
                # Do a postorder traversal
                # If we're a bias_add, don't let subsequent visits instrument the conv2d calls wrapped by us
                if op_name == 'nn.bias_add':
                    [self.skipped_exprs.add(x) for x in call.args]
                ret = super().visit_call(call)
                # Do a check to see if this is a torch.Linear equivalent. If so, instrument.
                if op_name in ['add', 'qnn.add']:
                    dequants = [x for x in call.args if isinstance(x, r.Call) and get_op_name(x.op) == 'qnn.dequantize']
                    requants = [dq.args[0] for dq in dequants if isinstance(dq.args[0], r.Call) and get_op_name(dq.args[0].op) == 'qnn.requantize']
                    qdenses = [rq.args[0] for rq in requants if isinstance(rq.args[0], r.Call) and get_op_name(rq.args[0].op) == 'qnn.dense']
                    if (any(isinstance(a, r.Call) and get_op_name(a.op) in ['nn.dense', 'qnn.dense'] for a in call.args)) or len(qdenses):
                        interesting_exprs.append(call)
                # Instrument conv2d operations with or without bias.
                elif op_name in ['nn.conv2d', 'qnn.conv2d', 'nn.bias_add']:
                    if call not in self.skipped_exprs:
                        interesting_exprs.append(call)
                return ret

            # def visit_tuple_getitem(self, t):
                # if t.index == 0 and \
                    # isinstance(t.tuple_value, r.Call) and get_op_name(t.tuple_value.op) == 'nn.batch_norm' and \
                    # isinstance(t.tuple_value.args[0], r.Call) and get_op_name(t.tuple_value.args[0].op) in ['nn.conv2d', 'qnn.conv2d', 'nn.dense']:
                    # self.skipped_exprs.add(t.tuple_value)
                    # self.skipped_exprs.add(t.tuple_value.args[0])
                # ret = super().visit_tuple_getitem(t)
                # interesting_exprs.append(t)
                # return ret

        InterestingExprCollector().visit(func.body)
        print(f'Identified {len(interesting_exprs)} instrumentable exprs.')

        def extract_weight_name(expr):
            if not (isinstance(expr, r.Call) and len(expr.args) > 1 and isinstance(expr.args[1], r.Var)):
                return None
            return expr.args[1].name_hint

        new_params = [x for x in func.params]
        if extra_params_fn:
            extra_params = extra_params_fn(interesting_exprs, nclasses)
            print(f'Appended {len(extra_params)} extra params.')

            if skipped_neurons:
                # If we have skipped neurons we must modify here so extra params
                # that are preserved in the function have the correct shapes.
                # FIXME: This assumes each neuron only has a column of
                # corresponding elements in the extra params.
                new_extra_params = extra_params.copy()
                for i, (expr, extra_param) in enumerate(zip(interesting_exprs, extra_params)):
                    weight_name = extra_param.name_hint if skip_as_eps else extract_weight_name(expr)
                    if weight_name not in skipped_neurons:
                        continue
                    ep_type = get_type(extra_param)
                    shape = ep_type.shape
                    # To be compatible with class_cond
                    assert 2 <= len(shape) <= 3
                    new_shape = shape[:-2] + [shape[-2], shape[-1] - len(skipped_neurons[weight_name])]
                    extra_param = r.var(extra_param.name_hint, shape=new_shape, dtype=ep_type.dtype)
                    new_extra_params[i] = extra_param
                extra_params = new_extra_params

            if verbose:
                print(f'{extra_params=}')
            assert len(extra_params) == len(interesting_exprs)
            new_params += extra_params
            if hic:
                hic.extra_param_vars.extend(extra_params)

        if skipped_weights:
            # We remove entries in interesting_exprs and extra_params at the same
            # time so they are still one-to-one. We don't need to remove them
            # from new_params because they aren't used anyway.
            skipped_weights = set(skipped_weights)
            new_interesting_exprs = []
            new_extra_params = []
            for i, expr in enumerate(interesting_exprs):
                if skip_as_eps:
                    if f'__ep_{i}' in skipped_weights:
                        continue
                else:
                    if extract_weight_name(expr) in skipped_weights:
                        continue
                new_interesting_exprs.append(expr)
                # Note that extra_params may be disabled
                if extra_params_fn:
                    new_extra_params.append(extra_params[i])
            print(f'Skipped {len(interesting_exprs) - len(new_interesting_exprs)} exprs.')
            interesting_exprs = new_interesting_exprs
            extra_params = new_extra_params

        cov_exprs = []
        layer_ret_types = [func.ret_type]
        class Mutator(r.ExprMutator):
            def handle_layer(self, e: r.Expr, extra_param):
                if verbose:
                    print(f'Handling {e.op if isinstance(e, r.Call) else type(e)} (orig type {e.checked_type})')
                shape = e.checked_type.concrete_shape
                dtype = e.checked_type.dtype
                weight_name = extract_weight_name(e)
                lcs = len(shape)
                assert lcs in [2, 4]
                if lcs == 4:
                    e = r.mean(e, (2, 3))
                    shape = (shape[0], shape[1])
                # Ignored whole layers have been removed from interesting_exprs
                # earlier, so we only need to care about neuron-wise selected
                # components here.
                if skipped_neurons:
                    key = extra_param.name_hint if skip_as_eps else weight_name
                    skipped_idxes = skipped_neurons.get(key, None)
                    if skipped_idxes:
                        preserved_idxes = [i for i in range(shape[1]) if i not in skipped_idxes]
                        e = r.take(e, r.const(np.array(preserved_idxes), dtype='int32'), axis=1)
                        shape = (shape[0], len(preserved_idxes))
                layer_ret_types.append(lcov_type_xf(r.TensorType(shape, dtype=dtype)))
                if class_cond:
                    extra_param = thread_first(extra_param,
                                               (r.take, pred_labels, 0),
                                               (r.squeeze, r.const(np.array([1]))))
                # Shape passed is (ninputs, nneurons)
                return lout2lcov(e, shape, extra_param)

            def visit(self, expr: r.Expr):
                # Only visit the first (last in graph) expr
                nonlocal cov_exprs
                cov_exprs = [self.handle_layer(e, extra_params[i] if extra_params_fn else None)
                             for i, e in enumerate(interesting_exprs)]
                out_tuple = r.Tuple([expr] + cov_exprs)
                return out_tuple

        new_body = Mutator().visit(func.body)
        rettype = r.TupleType(layer_ret_types)
        if verbose:
            print(f'{rettype=}')
        print(f'Function body mutated.')

        if not len(new_body.fields[1:]):
            warn(f'Did you skip all weights and neurons?')

        if overall_cov and mode in ['rb', 'cb']:
            warn(f'Automatically disabling overall cov for {mode} mode.')
            overall_cov = False

        if overall_cov:
            cov_expr, cov_type = lcovs2ocov(new_body.fields[1:], rettype.fields[1:])
            cov_exprs = [cov_expr]
            new_body = r.Tuple([func.body, cov_expr])
            rettype = r.TupleType([func.ret_type, cov_type])

        if hic:
            hic.cov_exprs.extend(cov_exprs)

        if verbose:
            print(f'{rettype=}')
        return r.Function(new_params,
                new_body,
                rettype,
                func.type_params, func.attrs)

    with tvm.transform.PassContext(opt_level=3):
        mod = r.transform.InferType()(mod)
        mod = ftransform(mod)

    output_defs = [{'shape': [x.value for x in t.shape], 'dtype': t.dtype}
                   for t in rettype.fields]
    if verbose:
        print(mod)
    print(f'Instrumentation finished in {time.time() - start_time:.3f} seconds.')
    return InstModPack(mod, extra_params, output_defs)

def instrument_module_npc(mod, modecfg, overall_cov=False, verbose=False):
    """Instruments the module for the NPC mode.
    Returns (instrumented_mod, extra_params, output_defs).
    If ref_builder is True, the module is instrumented for the reference
    builder mode.
    If overall_cov is True, overall coverage is returned instead of per-layer
    results.
    Does not support skipping or quantized models."""

    mode = modecfg.mode.lower()
    assert mode in ['npcb', 'npc', 'npcv']

    extra_params = []
    rettype = None
    @r.transform.function_pass(opt_level=1)
    def ftransform(func: r.Function, mod: IRModule, ctx: PassContext):
        nonlocal extra_params, rettype, overall_cov

        if not func.same_as(mod["main"]):
            return func

        interesting_exprs = []
        class InterestingExprCollector(BPVisitor):
            def visit_layerish(self, layerish):
                if isinstance(layerish, r.Var) and not layerish.name_hint.startswith('input'):
                    return
                if isinstance(layerish, r.Tuple):
                    # Handcrafted Relay tuples, treated specially later
                    return
                interesting_exprs.append(layerish)
        iec = InterestingExprCollector(func.body)
        iec.run()
        print(f'Found {len(interesting_exprs)} interesting exprs in {len(iec.handled)} layers.')

        if mode == 'npcb':
            # Note the last type would be for the reference input
            rettype = r.TupleType([func.ret_type] + [e.checked_type for e in interesting_exprs])
            return r.Function([x for x in func.params] + extra_params,
                    r.Tuple([func.body] + interesting_exprs),
                    rettype,
                    func.type_params, func.attrs)

        layer_ret_types = [func.ret_type]
        new_body = None

        # Extra params for ref-y's
        extra_params = [r.var(f'__ep_{i}',
                                shape=e.checked_type.concrete_shape,
                                dtype=e.checked_type.dtype)
                        for i, e in enumerate(interesting_exprs)]

        # A map from each interesting layer to the change in its output
        delta_louts_map = {e: type_inferred(e - ep, inplace=True)
                            for e, ep in zip(interesting_exprs, extra_params)}

        # A map to store each layer's children multipliers (unordered)
        children_mps_map = {e: [] for e in interesting_exprs}

        # Assign the initial multipliers to the model output layer
        model_out = interesting_exprs[0]
        pred_labels = r.argmax(model_out, axis=1, keepdims=True)
        children_mps_map[model_out].append(
            thread_first(model_out,
                         r.zeros_like,
                         (r.scatter, pred_labels, r.ones(get_shape(pred_labels), 'float32'), 1))
        )

        # Obtain per-layer contributions
        layer_contributions = []
        for lish in interesting_exprs:
            delta_y = delta_louts_map[lish]
            parents = layerish_parents(lish)
            # Currently we deem uninteresting parents as constant hyperparameters, etc.
            # so we just give them all zero deltas
            delta_xs = [delta_louts_map.get(p, r.zeros_like(p)) for p in parents]

            children_mps = children_mps_map[lish]
            mp_out = reduce(r.add, children_mps)

            contrib = delta_y * mp_out
            layer_contributions.append(contrib)

            mps = b.dl_backward(lish, delta_y, delta_xs, mp_out)

            assert len(mps) <= len(parents)
            for p, mp in zip(parents, mps):
                if isinstance(p, r.Tuple):
                    # Arg for e.g. concatenate() is a tuple, so we try to pass
                    # the multipliers (also a tuple) through it
                    # Note that we only support one layer's passthrough now
                    for fi, f in enumerate(p.fields):
                        children_mps_map[f].append(r.TupleGetItem(mp, fi))
                    continue
                if p not in interesting_exprs:
                    continue
                children_mps_map[p].append(mp)

        if mode == 'npcv':  # Visulisation mode

            if not overall_cov:
                layer_ret_types.extend([get_type(c) for c in layer_contributions])
                new_body = r.Tuple([func.body] + layer_contributions)
            else:
                # Convert each layer's contribution to top-k indices (1 for now) followed by
                # total number of neurons in the layer (for plotting purposes)
                indices_mat = r.stack(
                    [r.reshape(r.argmax(c), (-1,)) for c in layer_contributions],
                    0
                )
                nneurons_mat = r.const(np.array([prod(get_shape(c))
                                                    for c in layer_contributions]).reshape(-1, 1))
                lcovs_mat = r.concatenate([indices_mat, nneurons_mat], axis=1)
                layer_ret_types.append(get_type(lcovs_mat))
                new_body = r.Tuple([func.body, lcovs_mat])

        elif mode == 'npc':

            # Lower workload by only doing computation for dozens of layers close to the end
            new_interesting_exprs = []
            new_layer_contributions = []
            for lish, c in zip(interesting_exprs, layer_contributions):
                curr_len = len(new_interesting_exprs)
                if curr_len >= 10 and curr_len >= len(interesting_exprs) * 0.1:
                    break
                if is_any_type(
                    layerish_core(lish),
                    'add', 'nn.bias_add', 'reshape', 'nn.dropout', 'concatenate',
                    'nn.relu', 'nn.batch_norm'
                ):
                    continue
                new_interesting_exprs.append(lish)
                new_layer_contributions.append(c)
            interesting_exprs = new_interesting_exprs
            layer_contributions = new_layer_contributions
            print(f'Limited contribution calculation to {len(interesting_exprs)} layers.')
            if verbose:
                print('Details:')
                [print_expr(e) for e in interesting_exprs]
                print('---')

            # Add extra params for safe neurons for each layer
            nclasses = get_shape(model_out)[1]
            layer_nneurons = [prod(get_shape(e)[1:]) for e in interesting_exprs]
            safe_neurons = [r.var(f'__ep_{len(extra_params)+i}',
                                  shape=(nclasses, layer_nneurons[i]),
                                  dtype='int32')
                            for i, _ in enumerate(interesting_exprs)]
            extra_params.extend(safe_neurons)

            # Obtain "newly seen" neurons for each layer
            layer_newly_seens = []
            for lc, ls in zip(layer_contributions, safe_neurons):
                lc = r.reshape(lc, (0, -1))
                ls = thread_first(ls,
                                  (r.take, pred_labels, 0),
                                  (r.squeeze, r.const(np.array([1]))))
                max_idxs = r.argmax(lc, axis=1, keepdims=True)
                min_idxs = r.argmin(lc, axis=1, keepdims=True)
                # TODO: Normalise
                ln = thread_first(get_shape(lc),
                                  (r.zeros, 'int32'),
                                  (r.scatter, max_idxs, r.full_like(max_idxs, r.const(1.)), 1),
                                  (r.scatter, min_idxs, r.full_like(min_idxs, r.const(-1.)), 1),
                                  (r.multiply, ls),
                                  (r.sum, 0))
                layer_newly_seens.append(ln)

            if not overall_cov:
                layer_ret_types.extend([get_type(n) for n in layer_newly_seens])
                new_body = r.Tuple([func.body] + layer_newly_seens)
            else:
                # Calculate the percentage of newly seen neurons across all layers
                ocov, typ = lcovs2ocov_basic(
                    layer_newly_seens, [get_type(n) for n in layer_newly_seens]
                )
                ocov = -ocov  # Convert to suspicious scores
                layer_ret_types.append(typ)
                new_body = r.Tuple([func.body, ocov])

        else:
            assert False

        new_params = [x for x in func.params] + extra_params
        rettype = r.TupleType(layer_ret_types)
        return r.Function(new_params,
                new_body,
                rettype,
                func.type_params, func.attrs)

    with tvm.transform.PassContext(opt_level=3):
        mod = r.transform.InferType()(mod)
        mod = ftransform(mod)

    output_defs = [{'shape': [x.value for x in t.shape], 'dtype': t.dtype}
                   for t in rettype.fields]

    return InstModPack(mod, extra_params, output_defs)

def instrument_module_gn(
    mod, modecfg, overall_cov=False, gn_last_n_layers=1, hic: HybridInstContainer = None,
    verbose=False
):
    """Instruments the module for the GradNorm mode.
    Returns (instrumented_mod, extra_params, output_defs).
    If ref_builder is True, the module is instrumented for the reference
    builder mode.
    If overall_cov is True, overall coverage is returned instead of per-layer
    results.
    Does not support skipping or quantized models."""

    mode = modecfg.mode.lower()
    assert mode in ('gn1', 'gn2', 'gninf')

    rettype = None
    @r.transform.function_pass(opt_level=1)
    def ftransform(func: r.Function, mod: IRModule, ctx: PassContext):
        nonlocal rettype, overall_cov

        if not func.same_as(mod["main"]):
            return func

        interesting_exprs = []
        # An ordered dict of layerish -> grad_out_expr to store the grads for the layers
        # that we're interested in calculating GradNorm for.
        interesting_lish_grads = {}
        class InterestingExprCollector(BPVisitor):
            def visit_layerish(self, layerish):
                if is_any_type(layerish, 'nn.conv2d', 'nn.dense', 'qnn.conv2d', 'qnn.dense'):
                    # We're interested in their weight's grad_out
                    interesting_lish_grads[layerish.args[1]] = None
                interesting_exprs.append(layerish)
        iec = InterestingExprCollector(func.body)
        iec.run()
        print(f'Found {len(interesting_exprs)} interesting exprs in {len(iec.handled)} layers.')

        layer_ret_types = [func.ret_type]
        new_body = None

        # A map to store each layer's grad_out's (unordered)
        grads_out_map = {e: [] for e in interesting_exprs}

        # Assign initial grad_out to the model output layer
        model_out = func.body
        nclasses = r.const(get_shape(model_out)[1], dtype='float32')
        c1 = r.const(1.)
        grads_out_map[model_out].append(
            -c1/nclasses * (c1 - nclasses * r.nn.fast_softmax(model_out))
        )

        # Below we'll handle both QNN and normal models. For QNN, we keep track
        # of gradients in fp32, which are obtained by calling the grad_fn on
        # artificially created non-QNN counterparts of the QNN layers. Also,
        # quantize, dequantize, requantize layers will simply pass the gradients
        # through.

        for i, lish in enumerate(interesting_exprs):
            core = layerish_core(lish)
            parents = layerish_parents(lish)
            grads_out = grads_out_map[lish]

            if is_any_type(core, r.Var, r.Constant, r.Tuple) and not grads_out:
                # Some layers' grad_fn may not return grad_in for weights
                continue

            grad_out = reduce(r.add, grads_out)
            if lish in interesting_lish_grads:
                interesting_lish_grads[lish] = grad_out

            if len(parents) == 0:
                continue

            if is_any_type(core, 'qnn.quantize', 'qnn.dequantize', 'qnn.requantize'):
                grads_in = [grad_out]
            elif isinstance(core, r.Call) and get_op_name(core.op).startswith('qnn.'):
                unqexpr, _deqout = unquant_expr(core)
                grad_fn = b.get_grad_fn(unqexpr)
                grads_in = grad_fn(unqexpr, grad_out)
            elif isinstance(core, r.Tuple):  # Handcrafted Relay tuples
                grads_in = [r.TupleGetItem(grad_out, i) for i in range(len(parents))]
            else:
                grad_fn = b.get_grad_fn(core)
                grads_in = grad_fn(core, grad_out)

            assert len(grads_in) <= len(parents)
            for p, g in zip(parents, grads_in):
                if p not in interesting_exprs:
                    continue
                grads_out_map[p].append(g)

        if mode == 'gn1':
            norm_fn = lambda x: thread_first(x, r.abs, r.sum, (r.reshape, (-1,)))
        elif mode == 'gn2':
            norm_fn = lambda x: thread_first(x * x, r.sum, (r.reshape, (-1,)))
        elif mode == 'gninf':
            norm_fn = lambda x: thread_first(x, r.abs, r.max, (r.reshape, (-1,)))
        else:
            assert False

        # Shorten interesting_lish_grads and calculate norm
        grad_norms = []
        for i, (k, v) in enumerate(interesting_lish_grads.items()):
            assert v, f'{desc_expr(k)} has no grad_out'
            if i >= gn_last_n_layers:
                break
            grad_norms.append(norm_fn(v))
        print(f'Limiting to {len(grad_norms)} grad_norms')

        cov_exprs = []
        if not overall_cov:
            cov_exprs += grad_norms
        else:
            ocov, _typ = lcovs2ocov_basic(
                grad_norms, [get_type(n) for n in grad_norms]
            )
            cov_exprs.append(ocov)

        if hic:
            hic.cov_exprs.extend(cov_exprs)
            cov_exprs = hic.cov_exprs

        new_params = [x for x in func.params]
        if hic:
            new_params += hic.extra_param_vars

        layer_ret_types += [get_type(x) for x in cov_exprs]
        new_body = r.Tuple([func.body, *cov_exprs])
        rettype = r.TupleType(layer_ret_types)
        return r.Function(new_params, new_body, rettype, func.type_params, func.attrs)

    with tvm.transform.PassContext(opt_level=3):
        mod = r.transform.InferType()(mod)
        mod = ftransform(mod)

    output_defs = [{'shape': [x.value for x in t.shape], 'dtype': t.dtype}
                   for t in rettype.fields]

    return InstModPack(mod, [], output_defs)

def instrument_module_odin_vfuture(
    mod, temp=1000, eps=0.0012, verbose=False
):

    class FuncCloner(r.ExprMutator):
        def visit_var(self, var):
            return r.var(var.name_hint, var.type_annotation)

    def sm_score(logits, temp=1):
        return thread_first(logits / r.const(temp, dtype='float32'),
                            (r.nn.softmax, -1),
                            (r.max, -1))

    # Constructs an augmented function that returns the original prediction
    # outputs with the gradient w.r.t. the input
    def create_augf(origf):
        # We first create an intermediate fn that can return the log_sm_score
        tempfn = FuncCloner().visit(origf)
        orig_logits = tempfn.body
        log_sm_score = r.log(sm_score(orig_logits, temp=temp))
        tempfn_out = r.Tuple([orig_logits, log_sm_score])
        tempfn = r.Function(
            tempfn.params, tempfn_out, get_type(tempfn_out),
            tempfn.type_params, tempfn.attrs
        )

        # Infer type for later use, and update the references
        tempfn = type_inferred(tempfn)
        orig_logits, log_sm_score = tempfn.body.fields

        grads_out_map = {}
        input_var = tempfn.params[0]
        input_grad = None
        class BPExprCollector(BPVisitor):
            def visit_layerish(self, layerish):
                grads_out_map[layerish] = []
        BPExprCollector(log_sm_score).run()
        grads_out_map[log_sm_score].append(r.const(1.))

        for lish, grads_out in grads_out_map.items():
            core = layerish_core(lish)
            parents = layerish_parents(lish)

            if is_any_type(core, r.Var, r.Constant, r.Tuple) and not grads_out:
                # Some layers' grad_fn may not return grad_in for weights
                continue

            grad_out = reduce(r.add, grads_out)
            if lish == input_var:
                input_grad = grad_out
                break

            if not parents:
                continue

            if isinstance(core, r.Tuple):  # Handcrafted Relay tuples
                grads_in = [r.TupleGetItem(grad_out, i) for i in range(len(parents))]
            else:
                grad_fn = b.get_grad_fn(core)
                grads_in = grad_fn(core, grad_out)

            assert len(grads_in) <= len(parents)
            for p, g in zip(parents, grads_in):
                if p not in grads_out_map:
                    continue
                grads_out_map[p].append(g)

        assert input_grad is not None, f'No grad for the input var {desc_expr(input_var)}'

        new_out = r.Tuple([orig_logits, input_grad])
        return r.Function(
            tempfn.params, new_out, get_type(new_out),
            tempfn.type_params, tempfn.attrs
        )

    def xform_input(x, grad):
        return x - r.const(eps, dtype='float32') * r.sign(-grad)

    mod = r.transform.InferType()(mod)
    mainf = mod['main']
    origx = mainf.params[0]
    weights = mainf.params[1:]
    origf = FuncCloner().visit(mainf)
    augf = create_augf(origf)
    augf = type_inferred(augf)

    augf_call = augf(origx, *weights)
    orig_logits, grad = [r.TupleGetItem(augf_call, i) for i in range(2)]
    newx = xform_input(origx, grad)
    new_logits = origf(newx, *weights)
    sus_score = r.const(1.) - sm_score(new_logits)  # The larger the more suspicious
    new_out = r.Tuple([orig_logits, sus_score])

    new_mainf = r.Function(
        mainf.params, new_out, get_type(new_out), mainf.type_params, mainf.attrs
    )

    mod = thread_first(tvm.IRModule({'main': new_mainf}),
                       r.transform.LambdaLift(),
                       r.transform.InferType())

    output_defs = [{'shape': [x.value for x in t.shape], 'dtype': t.dtype}
                   for t in get_type(new_out).fields]

    return InstModPack(mod, [], output_defs)

# It looks like instrument_module_odin_vfuture is too advanced for
# GraphExecutor, so we add this function which invokes that one and manually
# inline all function calls (relay.transform.Inline doesn't seem to work as
# expected) by expanding them.
def instrument_module_odin(
    mod, temp=1000, eps=0.0012, verbose=False
):
    """The lambdas in mod should have been lifted to the global scope."""
    imp = instrument_module_odin_vfuture(mod, temp=temp, eps=eps, verbose=verbose)
    mod = imp.irmod

    def inlined_call(call, mod):
        assert isinstance(call.op, r.GlobalVar)
        assert call.op in mod.get_global_vars()
        fn = mod[call.op]
        assert isinstance(fn, r.Function)
        assert len(fn.params) == len(call.args)
        params2args = {p: a for p, a in zip(fn.params, call.args)}
        class FnCopyExpander(r.ExprMutator):
            def visit_var(self, var):
                return params2args[var]
        return FnCopyExpander().visit(fn.body)

    class CallInliner(r.ExprMutator):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def visit_call(self, call):
            call = super().visit_call(call)
            if isinstance(call.op, r.GlobalVar):
                return inlined_call(call, self.mod)
            return call

    new_mainf = CallInliner(mod).visit(mod['main'])

    mod = tvm.IRModule({'main': new_mainf})
    mod = r.transform.InferType()(mod)

    return InstModPack(mod, imp.extra_param_vars, imp.output_defs)

# vim: set fdm=marker:
