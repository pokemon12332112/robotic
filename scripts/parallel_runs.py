"""Assign independent single-GPU model jobs to available Docker GPU slots."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import time

from bench import KIT, MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'eval'])
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    devices = args.gpus.split(',')
    if len(devices) != len(set(devices)) or not all(x.isdigit() for x in devices):
        parser.error('GPU IDs must be unique comma-separated nonnegative integers')
    if len(args.models) != len(set(args.models)):
        parser.error('Each model may occur once; distinct experiments need distinct output directories')
    jobs, active, failures = deque(args.models), {}, []
    if args.dry_run:
        print(json.dumps({'action': args.action, 'gpu_slots': devices, 'queue': list(jobs),
                          'strategy': 'one model per GPU; remaining models wait for a free slot'}, indent=2))
        return
    logs = KIT / 'runtime/scheduler-logs'
    logs.mkdir(parents=True, exist_ok=True)
    try:
        while jobs or active:
            for device in devices:
                if device not in active and jobs:
                    model = jobs.popleft()
                    logfile = (logs / f'{args.action}-{model}-{int(time.time())}.log').open('w')
                    env = dict(os.environ, GPU_ID=device)
                    child = subprocess.Popen([str(KIT / 'docker/run.sh'), args.action, model],
                                             env=env, stdout=logfile, stderr=subprocess.STDOUT)
                    active[device] = (model, child, logfile)
                    print(f'Started {model} on GPU {device}: {logfile.name}', flush=True)
            for device, (model, child, logfile) in list(active.items()):
                status = child.poll()
                if status is not None:
                    logfile.close()
                    del active[device]
                    if status:
                        failures.append(model)
                    print(f'Finished {model}: exit {status}', flush=True)
            if active:
                time.sleep(2)
    finally:
        for model, child, logfile in active.values():
            child.terminate()
            child.wait()
            logfile.close()
    if failures:
        raise SystemExit('Failed models: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
