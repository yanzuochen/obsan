import tvm.relay as r

from utils import *

def simple_convish_weight(conv):
    return conv.args[1]

def conv_shape(conv):
    return f'{simple_convish_weight(conv).checked_type.shape}'

def is_conv(call):
    return get_op_name(call.op) in ['nn.conv2d', 'qnn.conv2d']

def is_dense(call):
    return get_op_name(call.op) in ['nn.dense', 'qnn.dense']

def is_simple_convish(expr):
    if not isinstance(expr, r.Call):
        return False
    return is_conv(expr) or is_dense(expr)

def is_convish(expr):
    """A convish can be a [q]{nn.conv2d, nn.dense} or a {TupleGetItem(0)+nn.batch_norm,
    nn.bias_add}-wrapped [q]{nn.conv2d, nn.dense}.
    It represents a small logical group in which the operators' parameters' shape
    change together."""
    if is_simple_convish(expr):
        return True
    if isinstance(expr, r.TupleGetItem) and \
        expr.index == 0 and isinstance(expr.tuple_value, r.Call) and \
        get_op_name(expr.tuple_value.op) == 'nn.batch_norm' and \
        is_simple_convish(expr.tuple_value.args[0]):
        return True
    if not isinstance(expr, r.Call):
        return False
    op_name = get_op_name(expr.op)
    if op_name == 'nn.bias_add' and \
        isinstance(expr.args[0], r.Call) and is_simple_convish(expr.args[0]):
        return True
    return False

def convish_components(convish):
    assert is_convish(convish)
    if is_simple_convish(convish):
        return [convish]
    if isinstance(convish, r.TupleGetItem):
        return [convish.tuple_value.args[0], convish.tuple_value, convish]
    op_name = get_op_name(convish.op)
    if op_name == 'nn.bias_add':
        return [convish.args[0], convish]
    raise NotImplementedError(f'{op_name} is not supported')

def is_dense_convish(convish):
    return is_dense(convish_components(convish)[0])

def is_shape_preserving(call):
    op_name = get_op_name(call.op)
    return not any(x in op_name for x in ['pool'])

def convish_weight(convish):
    return simple_convish_weight(convish_components(convish)[0])

def is_layerish(expr):
    if is_any_type(expr, r.Call, r.Var, r.Tuple, r.Constant):
        return True
    if isinstance(expr, r.TupleGetItem) and \
            expr.index == 0 and \
            is_any_type(expr.tuple_value, 'nn.batch_norm', 'nn.dropout'):
        return True
    return False

def layerish_components(layerish):
    assert is_layerish(layerish)
    if isinstance(layerish, r.TupleGetItem):
        return [layerish.tuple_value, layerish]
    return [layerish]

def layerish_core(layerish):
    return layerish_components(layerish)[0]

def layerish_parents(layerish, layerishs_only=False, pred=None):
    core = layerish_core(layerish)
    if isinstance(core, r.Call):
        return [x for x in core.args if
                (not layerishs_only or is_layerish(x)) and
                (not pred or pred(x))]
    if isinstance(core, r.Tuple):
        return [x for x in core.fields if
                (not layerishs_only or is_layerish(x)) and
                (not pred or pred(x))]
    return []

class ConvishVisitor(r.ExprVisitor):
    def __init__(self, post_order=False):
        super().__init__()
        self.handled = set()  # TODO: Check correctness
        self.post_order = post_order

    def visit_convish(self, convish):
        raise NotImplementedError()

    def visit_maybe_convish(self, convish, superf):
        if convish in self.handled or not is_convish(convish):
            return superf(convish)
        [self.handled.add(c) for c in convish_components(convish)]
        if self.post_order:
            ret = superf(convish)
            self.visit_convish(convish)
            return ret
        self.visit_convish(convish)
        return superf(convish)

    def visit_call(self, call):
        return self.visit_maybe_convish(call, super().visit_call)

    def visit_tuple_getitem(self, t):
        return self.visit_maybe_convish(t, super().visit_tuple_getitem)

class LayerishVisitor(r.ExprVisitor):
    def __init__(self, post_order=False):
        super().__init__()
        # A set to allow the passthrough traversal behaviour for the first
        # n-1 components in a complex layerish.
        self.passthru = set()
        self.post_order = post_order

    def visit_layerish(self, layerish):
        raise NotImplementedError()

    def visit_maybe_layerish(self, layerish, superf):
        if layerish in self.passthru:
            return superf(layerish)
        if not is_layerish(layerish):
            raise NotImplementedError(f'{desc_expr(layerish)} is not a layerish')
        [self.passthru.add(c) for c in layerish_components(layerish)[:-1]]
        if self.post_order:
            ret = superf(layerish)
            self.visit_layerish(layerish)
            return ret
        self.visit_layerish(layerish)
        return superf(layerish)

    def visit_call(self, call):
        return self.visit_maybe_layerish(call, super().visit_call)

    def visit_tuple_getitem(self, t):
        return self.visit_maybe_layerish(t, super().visit_tuple_getitem)

    def visit_var(self, var):
        return self.visit_maybe_layerish(var, super().visit_var)

    def visit_tuple(self, var):
        # Handcrafted Relay tuples
        return self.visit_maybe_layerish(var, super().visit_tuple)

    def visit_constant(self, const):
        return self.visit_maybe_layerish(const, super().visit_constant)

class LayerishChildrenFinder(LayerishVisitor):
    def __init__(self):
        super().__init__()
        self.children = {}

    def visit_layerish(self, layerish):
        core = layerish_core(layerish)
        for parent in layerish_parents(layerish, layerishs_only=True):
            if parent not in self.children:
                self.children[parent] = set()
            self.children[parent].add(layerish)
        if layerish not in self.children:  # For the first visited layerish
            self.children[layerish] = set()

    def get_children_map(self):
        return self.children

    def get_children(self, layerish):
        if layerish not in self.children:
            raise KeyError(f'{desc_expr(layerish)}')
        return self.children[layerish]

class BPVisitor(LayerishVisitor):
    """A visitor that ensures every node's children are visited before itself.
    Suitable for backpropagation."""

    def __init__(self, expr):
        super().__init__()
        self.expr = expr
        self.handled = set()
        self.cf = LayerishChildrenFinder()

        self.cf.visit(self.expr)
        # [print(f'{desc_expr(x)}') for x in self.cf.get_children_map().keys()]

    def visit(self, expr):
        # A hack to disable memo_map as we want to defer the visited check
        self.memo_map = {}
        return super().visit(expr)

    def visit_maybe_layerish(self, layerish, superf):
        if layerish in self.passthru:
            return superf(layerish)
        if layerish in self.handled:
            return  # We're not a mutator so just return nothing
        children = self.cf.get_children(layerish)
        if not children.issubset(self.handled):
            return  # We're not a mutator so just return nothing
        [self.passthru.add(c) for c in layerish_components(layerish)[:-1]]
        self.handled.add(layerish)
        self.visit_layerish(layerish)
        return superf(layerish)

    def run(self):
        self.handled.clear()
        return super().visit(self.expr)
