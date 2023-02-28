#! /usr/bin/env python3

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from typing import NamedTuple, Any, Tuple
from itertools import product

import dataman
import evalutils
import inst
import modman
import prune
from evalconfig import *
import utils

model_names = ['resnet50', 'googlenet', 'densenet121']
model_names += [f'Q{x}' for x in model_names]

modes = ['NBC']

# model_name, mode, nws, params
combs = []

sweep_params = [
    .2, .4, .6, .8, 1.0,
]

combs += list(product(model_names, modes, [False], sweep_params))
combs += list(product(model_names, modes, [True], sweep_params))

val_loader = None
val_input_data = None

class EvaluationSelResult(NamedTuple):
    nskipped: int
    nneurons: int
    skipped_pct: float
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

def get_mod_upruned(model_name, mode, frac, output_dir, with_nws=False, force=False):
    if model_name.startswith('Q'):
        qmod, qparams = modman.get_irmod(
            model_name, dataset, mode, batch_size, image_size
        )
        omod, oparams = modman.get_irmod(
            model_name[1:], dataset, mode, batch_size, image_size
        )

        skipped_eps, skipped_neurons = prune.get_ignored_components(
            oparams, frac, as_eps=True, irmod=omod, nws=with_nws
        )
        qparams = prune.ignored_neurons_applied_to_extra_params(
            qparams, None, skipped_neurons, mode, eps_mode=True
        )
        nignored, nneurons = prune.calc_uprune_stats(qmod, skipped_eps, skipped_neurons, eps_mode=True)

        if not nignored and not force:
            return None, None, 0, nneurons

        params = qparams
        mod, _, output_defs = inst.instrument_module(
            qmod, mode, overall_cov=1, skipped_weights=skipped_eps,
            skipped_neurons=skipped_neurons, skip_as_eps=True, verbose=0
        )
    else:
        mod, params = modman.get_irmod(
            model_name, dataset, mode, batch_size, image_size
        )

        skipped_weights, skipped_neurons = prune.get_ignored_components(
            params, frac, nws=with_nws
        )
        params = prune.ignored_neurons_applied_to_extra_params(params, mod, skipped_neurons, mode)
        nignored, nneurons = prune.calc_uprune_stats(mod, skipped_weights, skipped_neurons)

        if not nignored and not force:
            return None, None, 0, nneurons

        mod, _, output_defs = inst.instrument_module(
            mod, mode, overall_cov=1, skipped_weights=skipped_weights, skipped_neurons=skipped_neurons, verbose=0
        )

    outfile = f'{output_dir}/built/{model_name}-{dataset}-{mode}-{frac}{"-nws" if with_nws else ""}.so'
    utils.ensure_dir_of(outfile)
    rtmod, _lib = modman.build_module(mod, params, export_path=outfile)
    return rtmod, output_defs, nignored, nneurons

def benchmark_uprune(model_name, mode, frac, nws, args, force=False):
    rtmod, output_defs, nskipped, nneurons = get_mod_upruned(
        model_name, mode, frac, args.output_dir, with_nws=nws, force=force
    )
    print(f'Skipped neurons: {nskipped}/{nneurons} ({nskipped/nneurons:.2%})')

    if nskipped == 0 and not force:
        return None

    infer_ms, infer_perf = -1, None
    if not args.no_infer_perf:
        _, _, infer_perf = modman.run_module(rtmod,
                                    val_input_data, output_defs=output_defs,
                                    benchmark=1)
        infer_ms = infer_perf.mean*1e3
        print(infer_perf)

    sffr = evalutils.benchmark_standard_fp_fn(args, model_name, mode, rtmod, output_defs)

    return EvaluationSelResult(
        nskipped=nskipped,
        nneurons=nneurons,
        skipped_pct=nskipped/nneurons*100,
        infer_ms=infer_ms,
        infer_perf=infer_perf,
        **sffr._asdict(),
    )

if __name__ == '__main__':
    parser = evalutils.get_base_arg_parser('sel')
    parser.add_argument('--no-infer-perf', '-I', action='store_true')
    args = parser.parse_args()

    val_loader = dataman.get_benign_loader(dataset, image_size, 'test', batch_size)
    val_input_data = next(iter(val_loader))[0]

    results = {}
    evalutils.save_results(results, args)
    print(f'Combinations: {combs}')
    for (model_name, mode, nws, params) in combs:
        frac = params
        key = f'{model_name}-{mode}-{frac}{"-nws" if nws else ""}'
        if evalutils.should_skip_existing(key, args):
            continue
        print(key)
        result = benchmark_uprune(
            model_name, mode, frac, nws, args, force=False
        )
        results[key] = result
        evalutils.save_results(results, args)
        print('---------------------\n')
