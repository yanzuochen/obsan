#! /usr/bin/env python3

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
import numpy as np
import dataman
from eval import evalutils
import inst
import modman
from support import detectors
from eval.evalconfig import *
import time
from typing import NamedTuple, Any, Tuple
from itertools import product
import sys

sys.setrecursionlimit(5000)  # For densenet121-odin

model_names = ['resnet50', 'googlenet', 'densenet121']
# model_names = ['resnet50']
# modes = ['none', 'NBC', 'gn1', 'gn2', 'gninf']
modes = ['none', 'NBC', 'gn2']
# modes = ['NBC']

combs = []
combs += list(product(model_names, modes))
combs += list(product([f'Q{x}' for x in model_names], modes))

val_loader = None
val_input_data = None

###

class EvaluationBaseResult(NamedTuple):
    inst_sec: float
    compile_sec: float
    compile_mib: float
    infer_ms: float
    infer_perf: Any
    correct_pct: float
    threshold: Tuple[float, float]
    fp_pct: float
    ae_fn_pct: float
    ud_fn_pct: float
    pb_fn_pct: float
    val_aed_result: evalutils.AEDResult
    ae_aed_result: evalutils.AEDResult
    ud_aed_result: evalutils.AEDResult
    pb_aed_result: evalutils.AEDResult

def evaluation_7(model_name, mode, args):
    assert mode in {'none', 'NBC', 'covar'} | inst.gn_modes
    print(f'Running {model_name}-{mode}\n')

    mod, params = modman.get_irmod(
        model_name, dataset, mode, batch_size, image_size
    )

    inst_time, output_defs = -1, None
    if mode != 'none':
        start = time.time()
        mod, _extra_params_vars, output_defs = inst.instrument_module(
            mod, mode, overall_cov=1, verbose=0
        )
        inst_time = time.time() - start

    start = time.time()
    export_path = f'{args.output_dir}/built/{model_name}-{dataset}-{mode}.so'
    rtmod, lib = modman.build_module(mod, params, export_path=export_path)
    compile_mib = os.path.getsize(export_path) / 1024**2
    compile_time = time.time() - start

    infer_ms, infer_perf = -1, None
    if not args.no_infer_perf:
        _, _, infer_perf = modman.run_module(rtmod, val_input_data, output_defs, benchmark=True)
        infer_ms = infer_perf.mean*1e3
        print(infer_perf)

    correct_pct = -1
    if mode == 'none' and not args.no_correct_pct:
        correct_pct = evalutils.check_accuracy(rtmod, val_loader) * 100

    sffr = evalutils.benchmark_standard_fp_fn(args, model_name, mode, rtmod, output_defs)

    return EvaluationBaseResult(
        inst_sec=inst_time,
        compile_sec=compile_time,
        compile_mib=compile_mib,
        infer_ms=infer_ms,
        infer_perf=infer_perf,
        correct_pct=correct_pct,
        **sffr._asdict(),
    )

def eval_ext_case(imp: modman.IRModPack, threshold_or_mode, args, output_defs=None):
    mod, params = imp
    if output_defs is None:
        output_defs = modman.DEFAULT_COV_OUTPUT_DEFS

    start = time.time()
    rtmod, _lib = modman.build_module(mod, params, export_path=None)
    compile_time = time.time() - start

    infer_ms, infer_perf = -1, None
    if not args.no_infer_perf:
        _, _, infer_perf = modman.run_module(rtmod, val_input_data, output_defs, benchmark=True)
        infer_ms = infer_perf.mean*1e3
        print(infer_perf)

    mode = 'default-mode'
    threshold = threshold_or_mode
    if threshold_or_mode == 'gn':
        sam_train_loader = dataman.get_sampling_benign_loader(dataset, image_size, 'train', batch_size, sam_train_frac)
        threshold = evalutils.estimate_threshold(model_name, mode, rtmod, output_defs, sam_train_loader)
        # ID data have larger magnitudes than OOD data
        threshold = (threshold[1], np.inf)
        print(f'{threshold=}')
    elif isinstance(threshold_or_mode, str):
        mode = threshold_or_mode
        threshold = None

    sffr = evalutils.benchmark_standard_fp_fn(
        args, model_name, mode, rtmod, output_defs, threshold=threshold
    )

    return EvaluationBaseResult(
        inst_sec=-1,
        compile_sec=compile_time,
        compile_mib=-1,
        infer_ms=infer_ms,
        infer_perf=infer_perf,
        correct_pct=-1,
        **sffr._asdict(),
    )

if __name__ == '__main__':
    parser = evalutils.get_base_arg_parser('base')
    parser.add_argument('--no-infer-perf', '-I', action='store_true')
    parser.add_argument('--no-correct-pct', '-C', action='store_true')
    parser.add_argument('--no-base-cases', '-B', action='store_true')
    parser.add_argument('--no-ext-cases', '-E', action='store_true')
    args = parser.parse_args()

    val_loader = dataman.get_benign_loader(dataset, image_size, 'test', batch_size)
    val_input_data = next(iter(val_loader))[0]

    results = {}
    evalutils.save_results(results, args)

    if not args.no_base_cases:
        print(f'Base cases: {combs}')
        for model_name, mode in combs:
            key = f'{model_name}-{mode}'
            if evalutils.should_skip_existing(key, args):
                continue
            ret = evaluation_7(model_name, mode, args)
            results[key] = ret
            evalutils.save_results(results, args)
            print('---------------------\n')

    if not args.no_ext_cases:

        input_shape = (batch_size, 3, image_size, image_size)
        for model_name in model_names:
            tm = modman.get_torch_mod(model_name, dataset)

            key = f'{model_name}-dg'
            if not evalutils.should_skip_existing(key, args):
                print(f'Running {key}')
                mod, params = modman.get_irmod(model_name, dataset, 'NBC', batch_size, image_size)
                mod, _, _output_defs = inst.instrument_module(
                    mod, 'NBC', overall_cov=True
                )
                results[key] = eval_ext_case(modman.IRModPack(mod, params), (0, 1e-9), args)
                evalutils.save_results(results, args)
                print('---------------------\n')

            key = f'{model_name}-dla'
            if not evalutils.should_skip_existing(key, args):
                print(f'Running {key}')
                state_dicts = torch.load(
                    f'./support/detectors/data/dla-data-{model_name}.pt', map_location=torch.device('cpu')
                )
                dla_tm = detectors.dla.DLAProtectedModule.from_state_dicts(tm, state_dicts)
                dla_imp = modman.torch2irmod(dla_tm, input_shape)
                results[key] = eval_ext_case(dla_imp, (0, .1), args)
                evalutils.save_results(results, args)
                print('---------------------\n')

            key = f'{model_name}-anr'
            if not evalutils.should_skip_existing(key, args):
                print(f'Running {key}')
                anr_tm = detectors.anr.ANRProtectedModule(tm)
                anr_imp = modman.torch2irmod(anr_tm, input_shape)
                results[key] = eval_ext_case(anr_imp, (0, .5), args)
                evalutils.save_results(results, args)
                print('---------------------\n')

            key = f'{model_name}-odin'
            if not evalutils.should_skip_existing(key, args):
                print(f'Running {key}')
                mod, params = modman.get_irmod(model_name, dataset, 'odin', batch_size, image_size)
                mod, _, _output_defs = inst.instrument_module(
                    mod, 'odin', overall_cov=True
                )
                results[key] = eval_ext_case(modman.IRModPack(mod, params), 'covar', args)
                evalutils.save_results(results, args)
                print('---------------------\n')

            key = f'{model_name}-gn'
            if not evalutils.should_skip_existing(key, args):
                print(f'Running {key}')
                mod, params = modman.get_irmod(model_name, dataset, 'gn1', batch_size, image_size)
                mod, _, _output_defs = inst.instrument_module(
                    mod, 'gn1', overall_cov=True
                )
                results[key] = eval_ext_case(modman.IRModPack(mod, params), 'gn1', args)
                evalutils.save_results(results, args)
                print('---------------------\n')

            key = f'{model_name}-vim'
            if not evalutils.should_skip_existing(key, args):
                print(f'Running {key}')
                R, alpha = torch.load(
                    f'./support/detectors/data/vim-data-{model_name}.pt', map_location=torch.device('cpu')
                )
                fc = getattr(tm, 'fc', getattr(tm, 'classifier', None))
                vim_tm = detectors.vim.ViMProtectedModule(tm, fc, R, alpha)
                vim_imp = modman.torch2irmod(vim_tm, input_shape)
                results[key] = eval_ext_case(vim_imp, 'vim', args)
                evalutils.save_results(results, args)
                print('---------------------\n')
