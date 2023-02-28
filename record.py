import os
import numpy as np
import tvm
from tqdm import tqdm
import torch

import modman
import utils

def maybe_save_to_file(outfile, data):
    if outfile:
        if '/' in outfile:
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
        torch.save(data, outfile)
        print(f'Saved to {outfile}.')

def updated_bounds(bounds, batch_bounds, pred_label):
    for i, (layer_bounds, (layer_batch_lows, layer_batch_highs)) in enumerate(zip(bounds, batch_bounds)):
        if pred_label is not None:
            layer_bounds = layer_bounds[pred_label]
        layer_lows, layer_highs = layer_bounds
        lows_updates = layer_batch_lows < layer_lows
        highs_updates = layer_batch_highs > layer_highs
        updated = np.array([lows_updates * layer_batch_lows + ~lows_updates * layer_lows,
                            highs_updates * layer_batch_highs + ~highs_updates * layer_highs])
        if pred_label is not None:
            bounds[i][pred_label] = updated
        else:
            bounds[i] = updated

def updated_base_covs(base_covs, batch_covs, pred_label):
    for i, layer_batch_covs in enumerate(batch_covs):
        if pred_label is not None:
            base_covs[i][pred_label] |= layer_batch_covs
        else:
            base_covs[i] |= layer_batch_covs

def record_basic(rtmod, output_defs, data_loader, updater, input_name='input0', outfile=None, class_cond=True):
    wrmod = modman.WrappedRtMod(rtmod, output_defs, input_name=input_name)
    ret = None
    for xs, _ys in tqdm(data_loader):
        if xs.shape[0] != data_loader.batch_size:
            utils.warn(f'Discarding batch: batch size {xs.shape[0]} != {data_loader.batch_size}')
            continue
        logits, *cov_outputs = wrmod.run(xs, rettype='all')
        pred_label = None
        if class_cond:
            pred_labels = np.argmax(logits, axis=1)
            ulabels, counts = np.unique(pred_labels, return_counts=True)
            pred_label = ulabels[np.argmax(counts)]
        if not ret:
            nclasses = output_defs[0]['shape'][1]
            shape_prefix = (nclasses,) if class_cond else ()
            ret = [np.zeros(shape_prefix + co.shape, dtype=co.dtype) for co in cov_outputs]
        updater(ret, cov_outputs, pred_label)
    maybe_save_to_file(outfile, ret)
    return ret

def record_npc(rtmod, output_defs, data_loader, input_name='input0', outfile=None, class_cond=True):
    assert class_cond
    wrmod = modman.WrappedRtMod(rtmod, output_defs, input_name=input_name)
    nclasses = output_defs[0]['shape'][1]
    ret = None
    for xs, _ys in tqdm(data_loader):
        if xs.shape[0] != data_loader.batch_size:
            utils.warn(f'Discarding batch: batch size {xs.shape[0]} != {data_loader.batch_size}')
            continue
        logits, *cov_outputs = wrmod.run(xs, rettype='all')
        pred_labels = np.argmax(logits, axis=1)
        ulabels, counts = np.unique(pred_labels, return_counts=True)
        pred_label = ulabels[np.argmax(counts)]
        if not ret:
            ret = [np.zeros((nclasses, co.shape[0]), dtype=co.dtype) for co in cov_outputs]
        for ni, layer_co in enumerate(cov_outputs):
            ret[ni][pred_label] += layer_co
    maybe_save_to_file(outfile, ret)
    return ret

def record_bounds(rtmod, output_defs, data_loader, input_name='input0', outfile=None, class_cond=True):
    return record_basic(rtmod, output_defs, data_loader, updated_bounds, input_name, outfile, class_cond=class_cond)

def record_base_covs(rtmod, output_defs, data_loader, input_name='input0', outfile=None, class_cond=True):
    return record_basic(rtmod, output_defs, data_loader, updated_base_covs, input_name, outfile, class_cond=class_cond)

def record_covars(rtmod, output_defs, data_loader, input_name='input0', outfile=None, class_cond=False):
    assert not class_cond
    covars = [tvm.nd.array(np.zeros(**d)) for d in output_defs[1:]]
    for xs, _ys in tqdm(data_loader):
        if xs.shape[0] != data_loader.batch_size:
            utils.warn(f'Discarding batch: batch size {xs.shape[0]} != {data_loader.batch_size}')
            continue
        [rtmod.set_input(f'__ep_{i}', cov) for i, cov in enumerate(covars)]
        rtmod.set_input(input_name, xs)
        rtmod.run()
        # We don't need the prediction results (first output)
        covars = [rtmod.get_output(i+1, cov) for i, cov in enumerate(covars)]
    covars = [cov.numpy() for cov in covars]
    maybe_save_to_file(outfile, covars)
    return covars

def record_ref_louts(rtmod, output_defs, data_loader, input_name='input0', outfile=None, class_cond=None):
    assert not class_cond
    # We'll just use data_loader to get the input shape without using its data
    input_shape = next(iter(data_loader))[0].shape
    ref_input = np.zeros(input_shape)
    ref_louts = [tvm.nd.array(np.zeros(**d)) for d in output_defs[1:]]
    rtmod.set_input(input_name, ref_input)
    rtmod.run()
    ref_louts = [rtmod.get_output(i+1, lout).numpy() for i, lout in enumerate(ref_louts)]
    maybe_save_to_file(outfile, ref_louts)
    return ref_louts

def get_record_fn(epb_mode):
    return {
        'rb': record_bounds,
        'cb': record_covars,
        'npcb': record_ref_louts,
        'npc': record_npc,
    }.get(epb_mode, record_base_covs)
