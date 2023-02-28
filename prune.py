import tvm.relay as r
import tvm.ir as ir
from tvm.ir import IRModule
from tvm.ir.transform import PassContext
import tvm
import numpy as np

from cgutils import *

# SPrune {{{

def sprune_param(params_dict, prune_name, prune_idxs, prune_axis):
    a = params_dict[prune_name]
    print(f'Pruning param {prune_name} ({a.shape})')
    if not isinstance(a, np.ndarray):
        a = a.numpy()
    a = np.delete(a, prune_idxs, axis=prune_axis)
    params_dict[prune_name] = tvm.nd.array(a)

def sprune_convish(params_dict, convish, prune_idxs, prune_axis):
    """Perform structured pruning on the weight, bias, and other batch_norm
    params of a convish."""
    components = convish_components(convish)
    weight = simple_convish_weight(components[0])
    print(f'Pruning weight {weight.name_hint}')
    sprune_param(params_dict, weight.name_hint, prune_idxs, prune_axis)
    if len(components) > 1:
        bias_adder = components[1]
        op_name = get_op_name(bias_adder.op)
        if op_name == 'nn.batch_norm':
            for i in (1, 2, 3, 4):
                sprune_param(params_dict, bias_adder.args[i].name_hint, prune_idxs, prune_axis)
        elif op_name == 'nn.bias_add':
            sprune_param(params_dict, bias_adder.args[1].name_hint, prune_idxs, prune_axis)

class ConvGrouper(r.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.seen = set()
        # A list of lists of convishs whose shapes need to change together
        self.conv_groups = []

    def find_group_from_expr(self, expr):
        """Recursively find all convs in the same group as `expr`"""
        if expr in self.seen:
            return
        self.seen.add(expr)

        # If this is a convish, add it to the current group and stop
        if is_convish(expr):
            self.conv_groups[-1].append(expr)
            [self.seen.add(c) for c in convish_components(expr)]
            return

        assert isinstance(expr, r.Call), f"unrecognised {type(expr)}"

        # If this isn't shape-preserving, stop
        if not is_shape_preserving(expr):
            return

        # Otherwise recurse on the args
        [self.find_group_from_expr(a) for a in expr.args]

    def visit_call(self, call):
        # Create group if needed
        if not len(self.conv_groups) or len(self.conv_groups[-1]):
            self.conv_groups.append([])
        self.find_group_from_expr(call)
        return super().visit_call(call)

    def visit_tuple_getitem(self, t):
        if not len(self.conv_groups) or len(self.conv_groups[-1]):
            self.conv_groups.append([])
        self.find_group_from_expr(t)
        return super().visit_tuple_getitem(t)

    def get_conv_groups(self, preserve_final_dense=False):
        if not len(self.conv_groups[-1]):
            self.conv_groups.pop()
        if not preserve_final_dense:
            first_group = self.conv_groups[0]
            first_call = first_group[0]
            if len(first_group) == 1 and is_dense_convish(first_call):
                self.conv_groups.pop(0)
        return self.conv_groups

class DownstreamConvishFinder(ConvishVisitor):
    def __init__(self):
        super().__init__()
        # A conv -> list[convish] map to store which convishes need to have
        # their weights changed as a result of a conv being changed.
        self.downstream_convs = {}

    def add_as_downstream(self, up, down):
        if up not in self.downstream_convs:
            self.downstream_convs[up] = []
        self.downstream_convs[up].append(down)

    @staticmethod
    def get_upstreams(call):
        if not is_convish(call):
            return call.args
        return convish_components(call)[0].args

    def handle_upstream_convs(self, call, upstreams):
        """Traverses the upstreams, adding the call as a downstream to all its
        upstream convs."""

        for up in upstreams:
            if not any(isinstance(up, x) for x in [r.TupleGetItem, r.Call]):
                continue

            if is_convish(up):
                self.add_as_downstream(up, call)
                continue

            assert isinstance(up, r.Call), f"unrecognised {type(up)}"

            # If this isn't a shape-preserving expr, stop, unless convish is a
            # dense layer (which can be put after an avgpool)
            if not is_shape_preserving(up) and not is_dense_convish(call):
                continue

            self.handle_upstream_convs(call, self.get_upstreams(up))

    def visit_convish(self, conv):
        # Make sure convs with no downstream convs still appear in the
        # downstream_convs map
        if conv not in self.downstream_convs:
            self.downstream_convs[conv] = []
        self.handle_upstream_convs(conv, self.get_upstreams(conv))

    def get_downstream_convishes(self):
        return self.downstream_convs

class FBodyUpdater(r.ExprMutator):
    def __init__(self, params_map):
        super().__init__()
        # A map from param name to the var instance to use
        self.params_map = params_map

    def visit_var(self, var):
        if var.name_hint in self.params_map:
            return self.params_map[var.name_hint]
        return var

    def visit_call(self, call):
        op_name = get_op_name(call.op)
        if op_name == 'nn.conv2d':
            new_args = [self.visit(arg) for arg in call.args]
            attrs_map = {k: call.attrs[k] for k in call.attrs.keys() if k not in ['channels', 'kernel_size']}
            return r.nn.conv2d(*new_args, **attrs_map)
        if op_name == 'nn.batch_norm':
            # FIXME: batch_norm thinks args[0] has the wrong shape
            # new_args = [self.visit(call.args[0])] + [self.params_map[arg.name_hint] for arg in call.args[1:]]
            new_args = [self.visit(arg) for arg in call.args]
            attrs_map = {k: call.attrs[k] for k in call.attrs.keys()}
            new_type_args = [get_type(arg) for arg in new_args]
            # print(f'new bn type args: {new_type_args}')
            return r.nn.batch_norm(*new_args, **attrs_map).astuple()
            # return r.Call(call.op, new_args, call.attrs, new_type_args, call.span)
            # return r.Call(call.op, new_args, call.attrs, None, call.span)
        if op_name in ['nn.bias_add', 'nn.dense']:
            # FIXME: MatmulRel thinks args[0] has the wrong shape
            new_args = [self.visit(arg) for arg in call.args]
            return r.Call(call.op, new_args, call.attrs, call.type_args, call.span)
        return super().visit_call(call)

def sprune(mod, params, prunef):
    """Performs structured pruning.
    prunef is a fn that takes the weight groups and returns a map from
    group index to list of filter indices indicating the filters to be
    pruned."""

    @r.transform.function_pass(opt_level=0)
    def ftransform(func: r.Function, mod: IRModule, ctx: PassContext):

        if not func.same_as(mod["main"]):
            return func

        cg = ConvGrouper()
        cg.visit(func.body)
        conv_groups = cg.get_conv_groups()
        if not len(conv_groups[-1]):
            conv_groups.pop()
        print(f'Found {len(conv_groups)} convish groups.')
        # print([[len(convish_components(x)) for x in g] for g in conv_groups])

        dcf = DownstreamConvishFinder()
        dcf.visit(func.body)
        downstream_convs = dcf.get_downstream_convishes()
        print(f'Collected downstream convishes for {len(downstream_convs)} convishes.')

        # print(mod)

        # Turn conv_groups into weight_groups
        weight_groups = [[convish_weight(conv) for conv in group] for group in conv_groups]
        # Get the filters to prune
        prune_map = prunef(weight_groups)
        print(f'Pruning {len(prune_map)} filter groups.')

        # Collect all downstream calls for each group
        group_downstream_convs = []
        for group in conv_groups:
            group_downstream_convs.append(set())
            for conv in group:
                for downstream in downstream_convs[conv]:
                    group_downstream_convs[-1].add(downstream)

        # Prune params

        # First, prune the convs in each group
        for group_idx, prune_indices in prune_map.items():
            group = conv_groups[group_idx]
            for conv in group:
                sprune_convish(params, conv, prune_indices, 0)

        # Then, update the downstream conv weights
        for group_idx, prune_indices in prune_map.items():
            group = group_downstream_convs[group_idx]
            print(f'Pruning {len(group)} downstream convs for group {group_idx}')
            for conv in group:
                weight_name = convish_weight(conv).name_hint
                sprune_param(params, weight_name, prune_indices, 1)

        # Update the graph

        # Update func params with new shapes
        new_params = []
        for fp in func.params:
            fp_name = fp.name_hint
            if fp_name not in params:
                new_params.append(fp)
                continue
            new_param = r.var(fp_name, shape=params[fp_name].shape, dtype=fp.checked_type.dtype)
            new_params.append(new_param)

        # Update func body
        new_params_map = {p.name_hint: p for p in new_params}
        new_body = FBodyUpdater(new_params_map).visit(func.body)

        return r.Function(new_params, new_body,
                          func.ret_type, func.type_params, func.attrs)

    with tvm.transform.PassContext(opt_level=3):
        mod = r.transform.InferType()(mod)
        mod = ftransform(mod)

    print(f'Pruning done.')
    return mod

# }}}

def upruned_legacy(params_dict, prunef=None, prune_dict=None, threshold=None, random_frac=None, percentile=None):
    """Note that threshold and percentile work in terms of absolute values."""

    assert sum(1 for x in [prunef, prune_dict, threshold, random_frac, percentile] if x) == 1
    np_weights = {k: v.numpy() if isinstance(v, tvm.nd.NDArray) else v for k, v in params_dict.items() if len(v.shape) == 4}
    if prunef:
        prunef(np_weights)
    elif prune_dict:
        for k, v in prune_dict.items():
            np_weights[k][v] = 0
    elif threshold:
        for k, v in np_weights.items():
            abs_v = np.abs(v)
            v[abs_v < threshold] = 0
    elif random_frac:
        for k, v in np_weights.items():
            idxs = np.random.choice(np.arange(v.size), replace=False, size=int(v.size * random_frac))
            v[idxs] = 0
    elif percentile:
        for k, v in np_weights.items():
            abs_v = np.abs(v)
            v[abs_v < np.percentile(abs_v, percentile)] = 0
    else:
        assert False
    pruned_params = params_dict.copy()
    for k, v in np_weights.items():
        pruned_params[k] = tvm.nd.array(v)
    return pruned_params

def get_ignored_components_legacy(params, fi_frac, li_frac, nws=True, as_eps=False, irmod=None):
    """Given a dict of params (supposedly after unstructured pruning) and an irmod,
    returns a set of layer weight names to ignore, as well as a map from weight
    names to neuron indices to ignore (the map will be empty if nws is False).
    If (fi_frac * 100)% of the weights in a filter are zero, the filter is flagged.
    If (li_frac * 100)% of the filters in a layer are flagged, the layer is ignored.
    If as_eps is True, layer weight names will be converted to the names of their
    corresponding extra params. irmod is required in this case."""

    ignored_weights = set()
    ignored_neurons = {}

    for pname, param in params.items():
        if isinstance(param, tvm.nd.NDArray):
            param = param.numpy()
        if len(param.shape) != 4:
            continue
        for i, f in enumerate(param):
            if np.count_nonzero(f == 0) / f.size > fi_frac:
                if pname not in ignored_neurons:
                    ignored_neurons[pname] = set()
                ignored_neurons[pname].add(i)
        if len(ignored_neurons.get(pname, ())) / param.shape[0] > li_frac:
            ignored_weights.add(pname)

    if as_eps:
        assert irmod
        weight_names = []
        class WeightNamesFinder(ConvishVisitor):
            def __init__(self):
                super().__init__(post_order=True)
            def visit_convish(self, convish):
                weight_names.append(convish_weight(convish).name_hint)
        WeightNamesFinder().visit(irmod['main'])
        weights_to_eps = {wname: f'__ep_{i}' for i, wname in enumerate(weight_names)}
        ignored_weights = set([weights_to_eps[wname] for wname in ignored_weights])
        ignored_neurons = {
            weights_to_eps[wname]: ignored_neurons[wname] for wname in ignored_neurons
        }

    if not nws:
        ignored_neurons = {}

    print(f'Ignoring {len(ignored_weights)} weights; and also {sum(len(ignored_neurons[k]) for k in ignored_neurons)} neurons in {len(ignored_neurons)} weights.')
    return ignored_weights, ignored_neurons

def get_ignored_components(params, frac, nws=False, as_eps=False, irmod=None):
    """Given a dict of params and an irmod, returns a set of layer weight names
    to ignore, as well as a map from weight names to neuron indices to ignore
    (the map will be empty if nws is False).
    This function uses the MinWeight metrics on layers if nws is False and on
    filters if it's True.
    If as_eps is True, layer weight names will be converted to the names of their
    corresponding extra params. irmod is required in this case."""

    conv_params = {}
    for pname, param in params.items():
        if isinstance(param, tvm.nd.NDArray):
            param = param.numpy()
        if len(param.shape) != 4:
            continue
        conv_params[pname] = param

    ignored_weights = set()
    ignored_neurons = {}

    mw = lambda weights: (weights**2).sum() / weights.size
    sorted_scores_dict = lambda d: dict(sorted(d.items(), key=lambda x: x[1]))

    if not nws:
        param_scores = {pname: mw(param) for pname, param in conv_params.items()}
        param_scores = sorted_scores_dict(param_scores)
        ignored_weights = set(list(param_scores.keys())[:int(frac * len(param_scores))])
    else:
        neuron_scores = {}
        for pname, param in conv_params.items():
            for i, f in enumerate(param):
                neuron_scores[(pname, i)] = mw(f)
        neuron_scores = sorted_scores_dict(neuron_scores)
        ignored_neurons_list = list(neuron_scores.keys())[:int(frac * len(neuron_scores))]
        for pname, i in ignored_neurons_list:
            if pname not in ignored_neurons:
                ignored_neurons[pname] = set()
            ignored_neurons[pname].add(i)
        # If all neurons in a layer are ignored, use LWS to ignore the layer instead
        ignored_weights = set([
            pname
            for pname, neurons in ignored_neurons.items()
            if len(neurons) == conv_params[pname].shape[0]
        ])
        ignored_neurons = {k: v for k, v in ignored_neurons.items() if k not in ignored_weights}

    if as_eps:
        assert irmod
        weight_names = []
        class WeightNamesFinder(ConvishVisitor):
            def __init__(self):
                super().__init__(post_order=True)
            def visit_convish(self, convish):
                weight_names.append(convish_weight(convish).name_hint)
        WeightNamesFinder().visit(irmod['main'])
        weights_to_eps = {wname: f'__ep_{i}' for i, wname in enumerate(weight_names)}
        ignored_weights = set([weights_to_eps[wname] for wname in ignored_weights])
        ignored_neurons = {
            weights_to_eps[wname]: ignored_neurons[wname] for wname in ignored_neurons
        }

    print(f'Ignoring {len(ignored_weights)} weights; and also {sum(len(ignored_neurons[k]) for k in ignored_neurons)} neurons in {len(ignored_neurons)} weights.')
    return ignored_weights, ignored_neurons

def ignored_neurons_applied_to_extra_params(params, mod, ignored_neurons, mode, eps_mode=False):
    """Given the ignored_neurons generated by get_ignored_components,
    returns a dict of params with the ignored neurons applied to their
    corresponding extra params.
    If as_eps was True in get_ignored_components, enable eps_mode here (in
    which case mod is not needed)."""

    # We first use a ConvishVisitor to collect all conv weight names
    weight_names = []
    if eps_mode:
        weight_names = list(ignored_neurons.keys())
    else:
        class WeightNamesFinder(ConvishVisitor):
            def __init__(self):
                super().__init__(post_order=True)
            def visit_convish(self, convish):
                weight_names.append(convish_weight(convish).name_hint)
        WeightNamesFinder().visit(mod['main'])

    ret = params.copy()
    ndeleted = 0
    for i, wn in enumerate(weight_names):
        if wn not in ignored_neurons:
            continue
        epn = wn if eps_mode else f'__ep_{i}'
        epp = params[epn]
        if isinstance(epp, tvm.nd.NDArray):
            epp = epp.numpy()
        ndeleted += len(ignored_neurons[wn])
        epp = np.delete(epp, list(ignored_neurons[wn]), axis=-1)
        if mode == 'covar':
            epp = np.delete(epp, list(ignored_neurons[wn]), axis=-2)
        if isinstance(epp, np.ndarray):
            epp = tvm.nd.array(epp)
        ret[epn] = epp

    print(f'Deleted {ndeleted} ignored neurons in extra params.')

    return ret

def calc_uprune_stats(mod, ignored_weights, ignored_neurons, eps_mode=False):
    """Returns the number of ignored neurons (whether ignored by whole layer or
    individual neurons) and total neuron count of the model.
    Supports as_eps mode of get_ignored_components through the eps_mode param."""

    # TODO: Also count layers ignored

    total_ignored = 0
    total_neurons = 0
    convish_idx = 0
    class Counter(ConvishVisitor):
        def __init__(self):
            super().__init__(post_order=True)
        def visit_convish(self, convish):
            nonlocal total_ignored, total_neurons, convish_idx
            w = convish_weight(convish)
            wn = f'__ep_{convish_idx}' if eps_mode else w.name_hint
            wneurons = get_type(w).concrete_shape[0]
            total_neurons += wneurons
            if wn in ignored_weights:
                total_ignored += wneurons
            elif wn in ignored_neurons:
                total_ignored += len(ignored_neurons[wn])
            convish_idx += 1
    Counter().visit(mod['main'])
    return total_ignored, total_neurons

# vim: set fdm=marker:
