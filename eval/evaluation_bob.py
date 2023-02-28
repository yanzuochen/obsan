#! /usr/bin/env python3

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from typing import NamedTuple, Any, Tuple
from itertools import product

import dataman
import evalutils
import inst
import modman
from evalconfig import *

model_names = ['resnet50', 'googlenet', 'densenet121']
# model_names += [f'Q{x}' for x in model_names]

combs = []
combs += list(product(model_names, ['gn1', 'gn2', 'gninf'], [1]))
combs += list(product(model_names, ['gn2'], [1, 2, 3, 4, 5]))

val_loader = None
val_input_data = None

class EvaluationBobResult(NamedTuple):
    infer_ms: float
    infer_perf: Any
    threshold: Tuple[float, float]
    fp_pct: float
    ae_fn_pct: float
    ud_fn_pct: float
    pb_fn_pct: float
    val_aed_result: evalutils.AEDResult
    ae_aed_result: evalutils.AEDResult
    ud_aed_result: evalutils.AEDResult
    pb_aed_result: evalutils.AEDResult

def evaluation_bob(model_name, mode, nlayers, args):
    assert mode in inst.gn_modes
    print(f'Running {model_name}-{mode}-{nlayers}\n')

    mod, params = modman.get_irmod(
        model_name, dataset, mode, batch_size, image_size
    )

    mod, _extra_params_vars, output_defs = inst.instrument_module(
        mod, mode, overall_cov=1, verbose=0, gn_last_n_layers=nlayers
    )

    export_path = f'{args.output_dir}/built/{model_name}-{dataset}-{mode}-{nlayers}.so'
    rtmod, lib = modman.build_module(mod, params, export_path=export_path)

    infer_ms, infer_perf = -1, None
    if not args.no_infer_perf:
        _, _, infer_perf = modman.run_module(rtmod, val_input_data, output_defs, benchmark=True)
        infer_ms = infer_perf.mean*1e3
        print(infer_perf)

    sffr = evalutils.benchmark_standard_fp_fn(args, model_name, mode, rtmod, output_defs)

    return EvaluationBobResult(
        infer_ms=infer_ms,
        infer_perf=infer_perf,
        **sffr._asdict(),
    )

if __name__ == '__main__':
    parser = evalutils.get_base_arg_parser('bob')
    parser.add_argument('--no-infer-perf', '-I', action='store_true')
    args = parser.parse_args()

    val_loader = dataman.get_benign_loader(dataset, image_size, 'test', batch_size)

    val_input_data = next(iter(val_loader))[0]

    results = {}
    evalutils.save_results(results, args)
    print(f'Combinations: {combs}')
    for model_name, mode, nlayers in combs:
        key = f'{model_name}-{mode}-{nlayers}'
        if evalutils.should_skip_existing(key, args):
            continue
        ret = evaluation_bob(model_name, mode, nlayers, args)
        results[key] = ret
        evalutils.save_results(results, args)
        print('---------------------\n')
