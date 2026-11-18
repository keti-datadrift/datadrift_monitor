"""Re-export a saved PAL run's test predictions without changing its outputs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = args.run.resolve()
    output = args.output.resolve()
    if output == run or run in output.parents:
        raise ValueError('Analysis output must be outside the source run')
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((run / 'active_learning_plan.json').read_text())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    manifest = []
    for item in plan:
        if not item['name'].startswith('eval_round_'):
            continue
        index = item['round_index']
        destination = output / ('round_%02d' % index)
        destination.mkdir(exist_ok=True)
        predictions = destination / 'test_predictions.pkl'
        completed = destination / 'inference_complete.json'
        if completed.exists() and predictions.exists():
            manifest.append(json.loads(completed.read_text()))
            print('Round %d: existing completed export' % index, flush=True)
            continue
        argv = list(item['argv'])
        argv[0] = sys.executable
        # A dumped config freezes the original model and evaluation pipeline.
        argv[2] = str(run / ('round_%02d' % index) /
                      'retinanet_voc_train_quality_ema_26e.py')
        argv[3] = str(run / ('round_%02d' % index) / 'latest.pth')
        argv[argv.index('--work-dir') + 1] = str(destination)
        argv[1:1] = ['-B']
        argv.extend(['--out', str(predictions)])
        print('Round %d: test inference started' % index, flush=True)
        start = time.time()
        with (destination / 'inference.log').open('w', encoding='utf-8') as log:
            subprocess.run(argv, cwd=str(root), env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        result = dict(round=index, argv=argv, elapsed_seconds=time.time() - start,
                      source_checkpoint=str(run / ('round_%02d' % index) / 'latest.pth'))
        completed.write_text(json.dumps(result, indent=2), encoding='utf-8')
        manifest.append(result)
        (output / 'inference_manifest.json').write_text(
            json.dumps(manifest, indent=2), encoding='utf-8')
        print('Round %d: export completed (%.1f s)' %
              (index, result['elapsed_seconds']), flush=True)


if __name__ == '__main__':
    main()
