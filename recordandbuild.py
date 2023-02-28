#! /usr/bin/env python3

import os
import torch
from itertools import product

import dataman
import inst
import modman
import record
import prune

building_batch_size = 1  # For AE detection, the output lib should have batch size 1

# Should be a multiple of both nimgs_per_class (required by NPC) and dataset_size (required by all)
recording_batch_size = 25

datasets = ['CIFAR10']
model_names = ['resnet50', 'googlenet', 'densenet121']
q_model_names = [f'Q{x}' for x in model_names]
modes = ['none', 'NBC', 'gn1', 'gn2', 'gninf']
# modes = ['NBC']

combinations = list(product(datasets, model_names, modes))
# combinations += list(product(['MNIST'], ['lenet1', 'lenet5'], modes))
# combinations += list(product(['CIFAR10_2'], ['resnet50_2'], modes))
combinations += list(product(['CIFAR10'], q_model_names, modes))

def get_nchans(dataset):
    return 1 if dataset == 'MNIST' else 3

def record_extra_params_npc(model_name, dataset, batch_size, output_dir):
    data_loader = dataman.get_benign_loader(dataset, image_size, 'train', batch_size)

    # First use npcb mode to obtain ref-y's for each layer
    output_file = f'{output_dir}/{dataset}-{model_name}-npcb.pth'
    if os.path.exists(output_file):
        print(f'Skipping recording {output_file}')
    else:
        mod, params = modman.get_irmod(
            model_name, dataset, 'npcb', batch_size, image_size, nchannels=get_nchans(dataset)
        )
        mod, _, output_defs = inst.instrument_module(
            mod, 'npcb', overall_cov=0, verbose=0
        )
        rtmod, lib = modman.build_module(mod, params)
        record.get_record_fn('npcb')(rtmod, output_defs, data_loader, outfile=output_file)
    rlout_data = torch.load(output_file)

    # Use the rlout data and zeroed safe neurons extra params to record the safe neurons
    output_file = f'{output_dir}/{dataset}-{model_name}-npc.pth'
    if os.path.exists(output_file):
        print(f'Skipping recording {output_file}')
        return
    mod, params = modman.get_irmod(
        model_name, dataset, 'npc', batch_size, image_size,
        include_extra_params=False, nchannels=get_nchans(dataset)
    )
    mod, extra_params_vars, output_defs = inst.instrument_module(
        mod, 'npc', overall_cov=0, verbose=0
    )
    # Obtain a params dict with unchanged weights, correct ref-y's, zeroed safe neurons
    params = modman.extra_params_removed(params)
    params = {**params, **modman.create_zeroed_extra_params_dict(extra_params_vars, ones=True)}
    for i, rlo in enumerate(rlout_data):  # First n extra params are for ref-y's
        params[f'__ep_{i}'] = rlo
    # Build the npc rtmod for extra params recording
    rtmod, lib = modman.build_module(mod, params)
    # Record extra params based on the data loader
    safe_neurons = record.get_record_fn('npc')(rtmod, output_defs, data_loader)
    # Manually merge two parts and save
    torch.save(rlout_data + safe_neurons, output_file)
    print(f'Saved to {output_file}.')

def maybe_record_extra_params(model_name, dataset, batch_size, mode, output_dir):
    if mode == 'none':
        print(f'No recording needed')
        return

    epb_mode = inst.cov_mode_configs[mode].epb_mode
    if not epb_mode:
        print(f'No recording needed')
        return

    output_file = f'{output_dir}/{dataset}-{model_name}-{mode}.pth'
    if os.path.exists(output_file):
        print(f'Skipping recording {output_file}')
        return

    if mode == 'npc':
        return record_extra_params_npc(model_name, dataset, batch_size, output_dir)

    # Use the last xx% of data in each class, leaving the rest for threshold determining
    data_loader = dataman.get_sampling_benign_loader(
        dataset, image_size, 'train', batch_size, 0.90, start_frac=0.10
    )
    # Get the irmod in epb (extra params building) mode so no extra params are required or loaded
    mod, params = modman.get_irmod(
        model_name, dataset, epb_mode, batch_size, image_size, nchannels=get_nchans(dataset),
        include_extra_params=(epb_mode != mode)
    )
    mod, extra_params_vars, output_defs = inst.instrument_module(
        mod, epb_mode, overall_cov=0, verbose=0
    )
    params = {**params, **modman.create_zeroed_extra_params_dict(extra_params_vars)}
    # Build the epb rtmod for extra params recording
    rtmod, lib = modman.build_module(mod, params)
    # Record extra params based on the data loader
    record.get_record_fn(epb_mode)(rtmod, output_defs, data_loader, outfile=output_file)

def maybe_build_mod(model_name, dataset, mode, output_dir, tunelog=None):
    output_file = f'{output_dir}/{model_name}-{dataset}-{mode}-{bool(tunelog)}.so'
    if os.path.exists(output_file):
        print(f'Skipping building {output_file}')
        return
    # Load the irmod together with the recorded extra params
    mod, params = modman.get_irmod(
        model_name, dataset, mode, building_batch_size, image_size, nchannels=get_nchans(dataset)
    )
    if mode != 'none':
        mod, _extra_params_vars, output_defs = inst.instrument_module(
            mod, mode, overall_cov=1, verbose=0
        )
    # Build the .so lib with extra params embedded
    rtmod, lib = modman.build_module(mod, params, tunelog=tunelog, export_path=output_file)

def maybe_build_hybrid_mod(model_name, dataset, output_dir):
    assert model_name.startswith('Q')

    mode1, mode2 = 'NBC', 'gn2'
    mode = f'{mode1}+{mode2}'
    frac, nws = 0.2, False
    gn_last_n_layers = 1

    output_file = f'{output_dir}/{model_name}-{dataset}-{mode}-{frac}.so'
    if os.path.exists(output_file):
        print(f'Skipping building {output_file}')
        return

    qmod, qparams = modman.get_irmod(
        model_name, dataset, mode1, building_batch_size, image_size
    )
    omod, oparams = modman.get_irmod(
        model_name[1:], dataset, mode1, building_batch_size, image_size
    )

    skipped_eps, skipped_neurons = prune.get_ignored_components(
        oparams, frac, as_eps=True, irmod=omod, nws=nws
    )
    qparams = prune.ignored_neurons_applied_to_extra_params(
        qparams, None, skipped_neurons, mode1, eps_mode=True
    )
    nignored, nneurons = prune.calc_uprune_stats(qmod, skipped_eps, skipped_neurons, eps_mode=True)

    params = qparams
    hic = inst.make_empty_hic()
    _mod, _, output_defs = inst.instrument_module(
        qmod, mode1, overall_cov=1, skipped_weights=skipped_eps, hic=hic,
        skipped_neurons=skipped_neurons, skip_as_eps=True, verbose=0
    )

    mod, extra_params_vars, output_defs = inst.instrument_module(
        qmod, mode2, overall_cov=1, verbose=0, hic=hic, gn_last_n_layers=gn_last_n_layers
    )

    rtmod, lib = modman.build_module(mod, params, export_path=output_file)

if __name__ == '__main__':
    for dataset, model_name, mode in combinations:
        image_size = 28 if model_name == 'lenet1' else 32
        print(f'{model_name=} {dataset=} {mode=} {image_size=}')
        maybe_record_extra_params(model_name, dataset, recording_batch_size, mode, '/export/d1/alex/aesan-data/Coverage')
        maybe_build_mod(model_name, dataset, mode, './built')
        print('-----------------')
    for model_name in q_model_names:
        print(f'Hybrid: {model_name}')
        maybe_build_hybrid_mod(model_name, dataset, './built')
        print('-----------------')
