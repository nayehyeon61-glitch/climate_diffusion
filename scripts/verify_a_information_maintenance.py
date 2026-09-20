"""Compare same-seed before/after smoke artifacts without modifying them.

Run smoke_information_process.py --only-process before and after maintenance.
The after run must include rendering. This checks behavior preservation, not
meteorological skill. No training is performed by this script.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import torch

from climate_diffusion.physical_information import digest
from climate_diffusion.train_information_process import write_json


def verify(before, after, after_report, output):
    before, after, report, output = map(Path, (before, after, after_report, output))
    if output.exists():
        raise FileExistsError(output)
    stages = {}
    for stage in 'ABC':
        name = f'process-{stage}.pt'
        old = torch.load(before / name, map_location='cpu', weights_only=False)
        new = torch.load(after / name, map_location='cpu', weights_only=False)
        equal = old['model'].keys() == new['model'].keys() and all(
            torch.equal(old['model'][k], value) for k, value in new['model'].items())
        assert equal, f'{stage}: trained tensors changed'
        old_rows = json.loads((before / f'process-{stage}.metrics.json').read_text())
        new_rows = json.loads((after / f'process-{stage}.metrics.json').read_text())
        assert len(old_rows) == len(new_rows)
        weighted_errors = []
        for a, b in zip(old_rows, new_rows):
            for split in ('train', 'validation'):
                for key in ('loss', 'selection'):
                    assert a[split][key] == b[split][key], (stage, split, key)
                values = b[split]
                weighted_errors.append(abs(sum(v for k, v in values.items()
                                               if k.startswith('weighted_')) - values['loss']))
        assert max(weighted_errors) < 1e-5
        stages[stage] = dict(tensors_bitwise_equal=equal, tensor_count=len(new['model']),
                            epochs=len(new_rows), best_epoch=new['best_epoch'],
                            loss_and_selection_exact=True, weighted_sum_max_error=max(weighted_errors),
                            checkpoint_sha256=digest(after / name),
                            runtime_seconds=new_rows[-1]['runtime_seconds'],
                            max_rss_kib=new_rows[-1]['max_rss_kib'])
    a = torch.load(after / 'process-A.pt', map_location='cpu', weights_only=False)
    b = torch.load(after / 'process-B.pt', map_location='cpu', weights_only=False)
    trainable = ('core.experts.', 'core.gate.correction.', 'core.history_encoder.')
    frozen = [k for k in a['model'] if not k.startswith(trainable)]
    assert all(torch.equal(a['model'][k], b['model'][k]) for k in frozen)
    with np.load(before / 'forecast-process.npz') as old, np.load(after / 'forecast-process.npz') as new:
        predictions = new['predictions']
        leads, times = new['lead_hours'], new['valid_times']
        assert np.array_equal(old['predictions'], predictions)
        assert np.array_equal(leads, np.arange(1, 21) * 6)
        assert np.array_equal(times, new['origin_time'] + leads.astype('timedelta64[h]'))
    frames = {}
    for interval, extension in ((6, 'mp4'), (12, 'gif')):
        folder = report / f'members-{interval}h'
        indices = np.arange(interval // 6 - 1, 20, interval // 6)
        with np.load(folder / 'trajectory.npz') as export:
            assert np.array_equal(export['predictions'], predictions[:, indices])
            assert np.array_equal(export['valid_times'], times[indices])
        frames[str(interval)] = []
        for member in range(len(predictions)):
            path = folder / f'member-{member:03d}.{extension}'
            if extension == 'gif':
                with Image.open(path) as image:
                    count = image.n_frames
            else:
                count = int(subprocess.check_output([
                    'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                    '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(path)], text=True))
            assert count == 120 // interval
            frames[str(interval)].append(count)
    payload = dict(scope='CPU synthetic maintenance regression; NOT ERA5 skill validation',
                   torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                   stages=stages, frozen_B_tensor_count=len(frozen), forecast_bitwise_equal=True,
                   shape=list(predictions.shape), all_member_exact_6h_12h_prefix=True,
                   frame_counts=frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'after-report', 'output'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(verify(**vars(parser.parse_args())), indent=2))
