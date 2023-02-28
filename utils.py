import os
import sys
from typing import NamedTuple, Tuple
import tvm.relay as r
import tvm.ir as ir
from tvm.ir import IRModule
import pickle
from tvm.relay.qnn.op import dequantize

known_ops = {ir.op.Op.get(x): x for x in ir.op.Op.list_op_names()}

def ensure_dir_of(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)

def save(obj, filepath, merge=True):
    ensure_dir_of(filepath)
    if merge and os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            orig_obj = pickle.load(f)
            if isinstance(orig_obj, dict):
                obj = {**orig_obj, **obj}
            elif is_any_type(orig_obj, list, tuple):
                obj = orig_obj + obj
            else:
                raise ValueError(f'Cannot merge {type(obj)} into {type(orig_obj)}')
    with open(filepath, 'wb+') as f:
        pickle.dump(obj, f)

def load(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def warn(msg):
    print(f'⚠️  Warning: {msg}', file=sys.stderr)

def thread_first(data, *calls):
    """An approximation of the "->" macro in Clojure."""

    def tryexec(f, func_name):
        try:
            return f()
        except Exception as e:
            print(f'thread_first failed to execute {func_name}: {e}')
            raise

    for c in calls:
        c = list(c) if isinstance(c, tuple) else [c]
        f = c[0]
        if len(c) == 2 and isinstance(c[1], dict):
            data = tryexec(lambda: f(data, **c[1]), f)
            continue
        args = [data] + c[1:]
        data = tryexec(lambda: f(*args), f)
    return data

def get_op_name(op):
    return known_ops[op]

def is_any_type(expr, *types):
    is_call = isinstance(expr, r.Call)
    for type_or_opname in types:
        if isinstance(type_or_opname, str):
            if is_call and get_op_name(expr.op) == type_or_opname:
                return type_or_opname
        elif isinstance(expr, type_or_opname):
            return type_or_opname
    return False

def type_inferred(expr, inplace=False):
    """Note: When inplace is True, only the type of the local expr is
    inferred."""

    assert expr, "type_inferred called on None"

    try:
        if inplace:
            r.transform.InferTypeLocal(expr)
            return expr
        expr_mod = r.transform.InferType()(IRModule.from_expr(expr))
        if isinstance(expr, r.Function):
            return expr_mod['main']
        return expr_mod['main'].body
    except:
        print(f'Failed to infer type for...', file=sys.stderr)
        print(str(expr).strip(), file=sys.stderr)
        print('^~~~~ ...this expression', file=sys.stderr)
        raise

def get_type(expr):
    assert expr, "get_type called on None"
    try:
        return expr.checked_type
    except:
        return type_inferred(expr).checked_type

def get_shape(expr):
    return get_type(expr).concrete_shape

def get_dtype(expr):
    return get_type(expr).dtype

def desc_expr_type(expr):
    typ = get_type(expr)
    if isinstance(typ, r.TensorType):
        return f'{typ.concrete_shape}@{typ.dtype}'
    return str(typ)

def _desc_expr(expr):
    lhs = expr.__class__.__name__
    if isinstance(expr, r.Call):
        arg_hint = ''
        if len(expr.args) >= 2:
            arg1 = expr.args[1]
            if isinstance(arg1, r.Var):
                arg_hint = f'<{arg1.name_hint}>'
        input_type = ''
        if len(expr.args) >= 1:
            input_type = desc_expr_type(expr.args[0])
        op = expr.op
        op_name = f'@{op.name_hint}' if isinstance(op, r.GlobalVar) else get_op_name(op)
        lhs = f'{op_name}{arg_hint}({input_type})'
    elif isinstance(expr, r.TupleGetItem):
        lhs = f'{_desc_expr(expr.tuple_value)}[{expr.index}]'
    elif isinstance(expr, r.Var):
        lhs = f'%{expr.name_hint}'
    return lhs

def desc_expr(expr):
    delim = ' -> '
    if isinstance(expr, r.Var):
        delim = ': '
    elif isinstance(expr, r.Tuple):
        delim = ''
    lhs = _desc_expr(expr)
    typ = desc_expr_type(expr)
    return f'{lhs}{delim}{typ}'

def desc_exprs(exprs):
    return [desc_expr(x) for x in exprs]

def print_expr(expr):
    print(desc_expr(expr))

def print_exprs(exprs):
    for x in exprs:
        print_expr(x)

def unquant_expr(qexpr) -> Tuple[r.Expr, r.Expr]:
    """Takes a QNN expr and returns a non-quantised equivalent expr
    as well as the dequantised result of the given expr.
    We also use a hack to populate checked_type onto the new exprs."""
    qop_name = get_op_name(qexpr.op)
    if qop_name in {'qnn.add', 'qnn.subtract', 'qnn.mul'}:
        op = ir.Op.get(qop_name[4:])
        qlhs, qrhs, qlhs_scale, qlhs_zero_point, qrhs_scale, qrhs_zero_point, qoutput_scale, qoutput_zero_point = qexpr.args
        lhs_axis, rhs_axis = qexpr.attrs['lhs_axis'], qexpr.attrs['rhs_axis']

        lhs = dequantize(qlhs, qlhs_scale, qlhs_zero_point, axis=lhs_axis)
        rhs = dequantize(qrhs, qrhs_scale, qrhs_zero_point, axis=rhs_axis)
        unqexpr = r.Call(op, [lhs, rhs])
        deqout = dequantize(qexpr, qoutput_scale, qoutput_zero_point)

        [type_inferred(x, inplace=True) for x in [lhs, rhs, unqexpr, deqout]]
        return unqexpr, deqout
    elif qop_name in {'qnn.conv2d', 'qnn.dense'}:
        qdata, qweight, qdata_z, qweight_z, qdata_s, qweight_s = qexpr.args
        dequant_scale = qdata_s * qweight_s
        attrs = {k: qexpr.attrs[k] for k in qexpr.attrs.keys() if k not in {'out_dtype', 'units'}}

        data = dequantize(qdata, qdata_s, qdata_z)
        weight = dequantize(qweight, qweight_s, qweight_z)
        unqexpr = r.nn.__dict__[qop_name[4:]](data, weight, **attrs)
        deqout = dequantize(qexpr, dequant_scale, r.const(0))

        [type_inferred(x, inplace=True) for x in [data, weight, unqexpr, deqout]]
        return unqexpr, deqout
    elif qop_name == 'qnn.concatenate':
        qdata, qdata_scales, qdata_zero_points, output_scale, output_zero_point = qexpr.args
        axis = qexpr.attrs['axis']

        inputs = [dequantize(x, s, z) for x, s, z in zip(qdata, qdata_scales, qdata_zero_points)]
        unqexpr = r.concatenate(inputs, axis)
        deqout = dequantize(qexpr, output_scale, output_zero_point)

        [type_inferred(x, inplace=True) for x in [unqexpr, deqout, *unqexpr.args]]
        return unqexpr, deqout
    else:
        raise ValueError(f'Don\'t know how to unquant {desc_expr(qexpr)}')
