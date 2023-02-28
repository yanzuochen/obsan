# CCO: Compatible Coverage Objects to be used with the fuzzer

from math import prod
import numpy as np
from tqdm import tqdm
import os
import torch

import modman
import inst
import record

def list2dict(l):
    return {str(k): v for k, v in enumerate(l)}

def maybe_tqdm(x):
    try:
        if len(x) == 1:
            return x
        return tqdm(x, total=len(x))
    except TypeError:
        return tqdm(x)

handle_output_def = lambda d: {'dtype': d['dtype'], 'shape': [x.value for x in d['shape']]}
handle_output_defs = lambda ds: [handle_output_def(d) for d in ds]

class CoverageObject:
    def __init__(self, model_name, mode, dataset, batch_size, image_size, input_name='input0'):
        self.model_name = model_name
        self.mode = mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.input_name = input_name

        # Wrapped RtMod
        self.wrmod = None
        self.extra_params = None

        # The current detailed coverage data
        self.cov_data = None
        # The current overall coverage value
        self.current = 0.0

    def compile(self):
        """Compiles the model to be used by this coverage, taking into
        consideration any updated extra params (if applicable).
        Usually must be called before calling coverage-related functions."""
        raise NotImplementedError

    def update_extra_params(self, data_list):
        """Using the list of input data, updates the extra params of the model.
        May not be needed by all modes.
        The implementation may choose to invalidate the current wrmod.
        May need to call compile() after this."""
        raise NotImplementedError

    def calc_and_update(self, data_list):
        """Uses the given list of data to update the current coverage state."""
        raise NotImplementedError

    def calculate(self, data):
        """Given the input data, returns the per-layer coverages which are the
        "combined" coverage of the new data and the training data (precise
        definition relies on the concrete implementation).
        The return value can be used in update() and gain().
        This function is pure."""
        raise NotImplementedError

    def all_coverage(self, cov_data):
        """Calculates the overall coverage of the given cov_data. Pure
        function."""
        raise NotImplementedError

    def update(self, cov_data, delta=None):
        """Given the cov_data, sets it as the current state and also updates the
        recorded overall coverage value with it.
        If delta is given, uses it directly to update the overall coverage."""
        self.cov_data = cov_data
        if delta is not None:
            self.current += delta
            return
        self.current = self.all_coverage(cov_data)

    def gain(self, cov_data_new):
        """Given the coverage data computed by calculate(), returns the overall
        difference between it and the current state."""
        new_rate = self.all_coverage(cov_data_new)
        return new_rate - self.current

    def predict(self, data):
        """Given the input data, returns the model prediction."""
        return self.wrmod.run(data, rettype='pred')

    def save(self, path):
        if '/' in path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.cov_data, path)

    def load(self, path):
        self.cov_data = torch.load(path)
        loaded_cov = self.all_coverage(self.coverage_dict)
        print('Overall coverage of loaded state: %f' % loaded_cov)

    def _get_and_instrument_irmod(self, batch_size=None, mode=None):
        if not mode:
            mode = self.mode
        if not batch_size:
            batch_size = self.batch_size
        irmod, params = modman.get_irmod(
            self.model_name, self.dataset, mode, batch_size, self.image_size,
            include_extra_params=False
        )
        irmod, extra_params_vars, output_defs = inst.instrument_module(
            irmod, mode, overall_cov=0, verbose=0
        )
        params = {**params, **modman.create_zeroed_extra_params_dict(extra_params_vars)}
        return irmod, params, output_defs

class PerNeuronCoverageObject(CoverageObject):
    """Suitable for coverage criteria that can indicate whether each of the neurons
    in a layer is covered, e.g. NBC, KMN, TopK."""

    def __init__(self, model_name, mode, dataset, batch_size, image_size, input_name='input0') -> None:
        super().__init__(model_name, mode, dataset, batch_size, image_size, input_name)
        self.modecfg = inst.cov_mode_configs[mode]
        # If True, the criterion's extra params are lower and higher bounds.
        # If False, the criterion's extra params are the layer coverages
        # obtained using the training set. This also means we don't need
        # a separate extra params builder to build() the base coverage.
        self.bounds_based = self.modecfg.epb_mode == 'rb'

        # Wrapped rtmod for extra params building
        self.epb_wrmod = None

    def compile(self):
        print('Compiling rtmod...')
        irmod, params, output_defs = self._get_and_instrument_irmod()
        if self.bounds_based:
            assert self.extra_params
            extra_params_dict = {f'__ep_{idx}': v for idx, v in enumerate(self.extra_params)}
            params = {**params, **extra_params_dict}

        rtmod, _lib = modman.build_module(irmod, params)
        self.wrmod = modman.WrappedRtMod(rtmod, output_defs, input_name=self.input_name)
        if not self.cov_data:
            self.cov_data = [np.zeros(**d) for d in modman.ensure_output_defs(output_defs[1:])]

    def update_extra_params(self, data_list, batch_size=None):
        assert self.bounds_based
        self.wrmod = None
        if not self.epb_wrmod:
            print(f'Initialising epb rtmod{f" with batch size {batch_size}" if batch_size else ""}...')
            irmod, params, output_defs = self._get_and_instrument_irmod(batch_size=batch_size, mode='rb')
            # Range builder doesn't actually need extra params
            epb_rtmod, _lib = modman.build_module(irmod, params)
            self.epb_wrmod = modman.WrappedRtMod(epb_rtmod, output_defs, input_name=self.input_name)
        # Update the extra params (lower and higher bounds)
        for data in maybe_tqdm(data_list):
            # Class-cond
            logits, *cov_outputs = self.epb_wrmod.run(data, rettype='all')
            pred_labels = np.argmax(logits, axis=1)
            ulabels, counts = np.unique(pred_labels, return_counts=True)
            pred_label = ulabels[np.argmax(counts)]
            self.extra_params = record.updated_bounds(self.extra_params, cov_updates, pred_label)

    def calculate(self, data):
        cov_updates = self.wrmod.run(data)
        return [cov | self.cov_data[i] for i, cov in enumerate(cov_updates)]

    def all_coverage(self, cov_data):
        if self.mode == 'KMN':
            # lcov is nneuron x (k+1), where 0th column indicates invalid
            nelements = sum([lcov.shape[0] * (lcov.shape[1]-1) for lcov in cov_data])
            ncovered = sum([np.sum(lcov[:, 1:]) for lcov in cov_data])
            return ncovered / nelements

        per_neuron_max = 1
        if self.mode == 'NBC':
            cov_data = [(lcov >> 1) + (lcov & 1) for lcov in cov_data]
            per_neuron_max = 2
        ncovered = sum([np.sum(lcov) for lcov in cov_data])
        nneurons = sum([lcov.shape[0] for lcov in cov_data])  # lcovs are 1-dim
        return ncovered / (nneurons * per_neuron_max)

    def calc_and_update(self, data_list):
        for data in maybe_tqdm(data_list):
            self.cov_data = self.calculate(data)
        self.update(self.cov_data)  # Trigger update for overall coverage
