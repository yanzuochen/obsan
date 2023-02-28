#! /usr/bin/env python3

from typing import NamedTuple, Any
import re
import numpy as np

log_entry_re = re.compile(r'^(?P<niters>\d+) (?P<init_failed>INIT FAILED )?.+success rate=(?P<success>\d+)/\d+ .+ defenses=(?P<ndefenses>\d+)$')

class BlackboxAELogEntry(NamedTuple):
    init_failed: bool
    success: bool
    nqueries: int
    ndefenses: int

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file')
    args = parser.parse_args()
    globals().update(args.__dict__)

    log_entries = []

    with open(log_file) as f:
        for line in f:
            if any(x in line for x in ['Running attack with', '#FP', 'initial accuracy', 'thresholds=']):
                print(line.strip())
                continue
            m = log_entry_re.match(line)
            if not m:
                continue
            nqueries = int(m.group('niters'))
            init_failed = m.group('init_failed') is not None
            success = m.group('success') != '0'
            ndefenses = int(m.group('ndefenses'))
            log_entries.append(BlackboxAELogEntry(init_failed, success, nqueries, ndefenses))

    successful_entries = [x for x in log_entries if x.success]

    print(f'Successful/Total: {len(successful_entries)}/{len(log_entries)} ({len(successful_entries)/len(log_entries):.2%}) <<<')

    if successful_entries:
        print(f'Avg. queries among successful: {np.mean([x.nqueries for x in successful_entries]):.2f} <<<')
        print(f'Median queries among successful: {np.median([x.nqueries for x in successful_entries]):.2f}')
        print(f'Max queries among successful: {np.max([x.nqueries for x in successful_entries]):.2f}')
        print(f'Min queries among successful: {np.min([x.nqueries for x in successful_entries]):.2f}')
        print(f'Avg. defenses among successful: {np.mean([x.ndefenses for x in successful_entries]):.2f}')
    else:
        print(f'(No successful entries.)')
    print(f'Avg. defenses among failed: {np.mean([x.ndefenses for x in log_entries if not x.success]):.2f}')
    print()
