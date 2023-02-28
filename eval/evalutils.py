import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from tqdm import tqdm
import torch
import time
import argparse
from typing import NamedTuple, Tuple

from support import coverage, tool

import modman
import inst
import utils
import dataman

try:  # Hacky
    import evalconfig as ec
except:
    import eval.evalconfig as ec

def get_base_arg_parser(exp_name, default_fbasename='results') -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default=f'./results/{exp_name}')
    parser.add_argument('--file-base-name', '-b', type=str, default=default_fbasename)
    parser.add_argument('--tag', '-t', type=str, default='')
    parser.add_argument('--no-merge-results', '-M', action='store_false', dest='merge_results')
    parser.add_argument('--skip-existing', '-S', action='store_true')
    parser.add_argument('--no-val', '-V', action='store_true')
    parser.add_argument('--no-ae', '-A', action='store_true')
    parser.add_argument('--no-undef', '-U', action='store_true')
    parser.add_argument('--no-pb', '-P', action='store_true')
    return parser

def get_output_fpath(args):
    tag = ''
    if args.tag:
        tag = f'-{args.tag}'
    return f'{args.output_dir}/{args.file_base_name}{tag}.pkl'

def save_results(ret, args):
    return utils.save(ret, get_output_fpath(args), merge=args.merge_results)

def should_skip_existing(key, args, quiet=False):
    if not args.skip_existing:
        return False
    fpath = get_output_fpath(args)
    if not os.path.exists(fpath):
        return False
    results = utils.load(fpath)
    should_skip = key in results
    if not quiet and should_skip:
        print(f'Skipping existing: {key}')
    return should_skip

def get_stats(a):
    return np.array([f(a) for f in [np.mean, np.median, np.max, np.min, np.std]])

def check_accuracy(rtmod, data_loader, nclasses=10, input_name='input0', topn=1):
    ncorrect = 0
    for xs, ys in tqdm(data_loader):
        runret = modman.run_module(
            rtmod, xs, input_name=input_name, default_nclasses=nclasses, output_labels=True
        )
        preds = runret.outputs[0][:, :topn]
        ys = ys.numpy().reshape(-1, 1)
        ncorrect += np.sum(ys == preds)
    accuracy = ncorrect / len(data_loader)
    print(f'Top-{topn} accuracy: {accuracy:.2%}')
    return accuracy

class AEDResult(NamedTuple):
    total: int
    stats: np.ndarray
    nneg: int
    neg_rate: float
    correct: int
    threshold: Tuple[float, float]
    scores: np.ndarray

def check_ae_detectability(
    rtmod, output_defs, loader, threshold, incorrect_only=False, topn=1, nsamples=10000, collect_scores=False, quiet=False
) -> AEDResult:
    assert loader.batch_size == 1
    assert len(output_defs) == 2 and output_defs[1]['shape'][0] == 1

    if isinstance(threshold, float) or isinstance(threshold, int):
        threshold = tuple(sorted((0, threshold)))
    assert len(threshold) == 2

    scores = []
    total, nneg, ncorrect = 0, 0, 0
    output_bufs = modman.make_output_bufs(output_defs)
    for _i, (imgs, ys) in enumerate(tqdm(loader)):
        if nsamples and total == nsamples:
            break
        outputs, *_ = modman.run_module(
            rtmod, imgs, output_defs=output_defs, output_bufs=output_bufs, output_labels=True
        )
        predictions = outputs[0][0, :topn]
        correct = any(ys.numpy().reshape(-1, 1) == predictions)
        if incorrect_only and correct:
            continue
        total += 1
        score = outputs[1][0]
        scores.append(score)
        negative = threshold[0] <= score <= threshold[1]
        if negative:
            nneg +=1
            if correct:
                ncorrect += 1
    if incorrect_only:
        goal = (nsamples if nsamples else len(loader)) * 0.9
        if total < goal:
            print(f'WARNING: Only {total} samples were collected. Consider increasing nsamples.')
    if not quiet:
        print(f'{total=}, {nneg=}, {nneg/total=:.2%}, {ncorrect=}, {threshold=}, {get_stats(scores)=}')
    return AEDResult(
        total=total,
        stats=get_stats(scores),
        nneg=nneg,
        neg_rate=nneg/total,
        correct=ncorrect,
        threshold=threshold,
        scores=np.array(scores) if collect_scores else None
    )

def determine_ae_threshold_pct(rtmod, output_defs, benign_loader, nsamples=None, percentiles=None, quiet=False):
    if not percentiles:
        percentiles = (1, 99.9)
    benign_scores = check_ae_detectability(
        rtmod, output_defs, benign_loader, 0, nsamples=nsamples, quiet=quiet, collect_scores=True
    ).scores
    t = np.percentile(benign_scores, percentiles)
    if not quiet:
        print(f'Threshold: {t}')
    return t

def determine_ae_threshold_pos(rtmod, output_defs, benign_loader, nsamples=None, position=0.35, quiet=False):
    benign_scores = check_ae_detectability(
        rtmod, output_defs, benign_loader, 0, nsamples=nsamples, quiet=quiet, collect_scores=True
    ).scores
    benign_scores = benign_scores[benign_scores > 0]
    t = (benign_scores.max() - benign_scores.min()) * position + benign_scores.min()
    t = (0, t)
    if not quiet:
        print(f'Threshold: {t}')
    return t

def estimate_threshold(
    model_name, mode, rtmod, output_defs, benign_loader, nsamples=None,
):
    extra_args = {}
    if mode in inst.gn_modes | {'covar', 'npc'}:
        if model_name.startswith('Q'):
            extra_args['percentiles'] = (1, 99.5)
        return determine_ae_threshold_pct(
            rtmod, output_defs, benign_loader, nsamples=nsamples, **extra_args
        )
    elif mode == 'vim':  # As vim can have negative scores
        return determine_ae_threshold_pct(
            rtmod, output_defs, benign_loader, nsamples=nsamples, percentiles=(0, 95)
        )
    else:
        if model_name.startswith('Q'):
            extra_args['position'] = 0.30
        return determine_ae_threshold_pos(
            rtmod, output_defs, benign_loader, nsamples=nsamples, **extra_args
        )

def benchmark_torch(model_name, dataset, mode, input_data):
    model = modman.get_torch_mod(model_name, dataset)
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.Tensor(input_data)
    lout_sizes = tool.get_layer_output_sizes(model, input_data)
    covlayer = {
        'NBC': coverage.LayerNBC(model, -123, lout_sizes),
        'TopK': coverage.LayerTopK(model, 10, lout_sizes),
        'none': None
    }[mode]
    times = []
    nexecs = 100
    orig_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.no_grad():
        if not covlayer:
            for _ in range(nexecs+10):
                start_time = time.time()
                output = model(input_data)
                times.append(time.time() - start_time)
        else:
            for _ in range(nexecs+10):
                start_time = time.time()
                _allcov = covlayer.all_coverage(covlayer.calculate(input_data))
                times.append(time.time() - start_time)
    times = times[10:]
    stats = get_stats(times)
    print(f'{stats*1e3} (mean, median, max, min, std; ms)')
    torch.set_num_threads(orig_num_threads)
    return stats

class StandardFPFNResult(NamedTuple):
    threshold: Tuple[float, float]
    fp_pct: float
    ae_fn_pct: float
    ud_fn_pct: float
    pb_fn_pct: float
    val_aed_result: AEDResult
    ae_aed_result: AEDResult
    ud_aed_result: AEDResult
    pb_aed_result: AEDResult

def benchmark_standard_fp_fn(
    args, model_name, mode, rtmod, output_defs, threshold=None
) -> StandardFPFNResult:

    fp_pct, val_aed_result = -1, None
    ae_fn_pct, ae_aed_result = -1, None
    ud_fn_pct, ud_aed_result = -1, None
    pb_fn_pct, pb_aed_result = -1, None

    if mode != 'none' and not all([args.no_val, args.no_ae, args.no_undef, args.no_pb]):
        sam_train_loader = dataman.get_sampling_benign_loader(ec.dataset, ec.image_size, 'train', ec.batch_size, ec.sam_train_frac)
        val_loader = dataman.get_benign_loader(ec.dataset, ec.image_size, 'test', ec.batch_size)
        if threshold is None:
            threshold = estimate_threshold(model_name, mode, rtmod, output_defs, sam_train_loader)

        if not args.no_val:
            val_aed_result = check_ae_detectability(
                rtmod, output_defs, val_loader, threshold, collect_scores=True
            )
            fp_pct = (1 - val_aed_result.neg_rate) * 100
            print(f'Val: FP: {fp_pct}%')

        if not args.no_ae:
            ae_loader = dataman.get_ae_loader(model_name, ec.dataset, ec.batch_size)
            ae_aed_result = check_ae_detectability(
                rtmod, output_defs, ae_loader, threshold, collect_scores=True,
                nsamples=ec.nreal_ae_samples, incorrect_only=True  # Important
            )
            ae_fn_pct = ae_aed_result.neg_rate * 100
            print(f'AE: FN: {ae_fn_pct}%')

        if not args.no_undef:
            ud_loader = dataman.get_undef_loader(ec.undef_dataset, ec.image_size)
            ud_aed_result = check_ae_detectability(
                rtmod, output_defs, ud_loader, threshold, collect_scores=True
            )
            ud_fn_pct = ud_aed_result.neg_rate * 100
            print(f'Undef: FN: {ud_fn_pct}%')

        if not args.no_pb:
            pb_loader = dataman.get_undef_loader(ec.pb_dataset, ec.image_size)
            pb_aed_result = check_ae_detectability(
                rtmod, output_defs, pb_loader, threshold, collect_scores=True
            )
            pb_fn_pct = pb_aed_result.neg_rate * 100
            print(f'PB: FN: {pb_fn_pct}%')

    return StandardFPFNResult(
        threshold=threshold,
        fp_pct=fp_pct,
        ae_fn_pct=ae_fn_pct,
        ud_fn_pct=ud_fn_pct,
        pb_fn_pct=pb_fn_pct,
        val_aed_result=val_aed_result,
        ae_aed_result=ae_aed_result,
        ud_aed_result=ud_aed_result,
        pb_aed_result=pb_aed_result,
    )
