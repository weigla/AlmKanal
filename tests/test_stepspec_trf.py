from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from almkanal import AlmKanal, preprocessing_report
from almkanal.report.methods_context import build_context_from_files
from almkanal.stim_utils.alignment_utils import summarize_alignments


def make_info(drifts: list[float], failures: int = 0) -> dict:
    trials = [{
        'drift_us_per_s': drift, 'offset_s': 0.04, 'clock_slope': 1 + drift / 1e6,
        'total_drift_ms': drift / 100, 'residual_rms_ms': 0.2, 'residual_max_ms': 0.3,
        'median_correlation': 0.98, 'n_anchor_inliers': 12,
    } for drift in drifts]
    return {
        'epoch_len_s': 1.0, 'feature': 'envelope', 'audio_cutoff_hz': 80,
        'hw_delay_s': 0.0165, 'applied_hw_delay_s': 0.016,
        'realign_audio': True, 'audio_channels': ['recorded'],
        'alignment_kwargs': {
            'sync_sfreq': 500, 'window_s': 10, 'step_s': 5, 'max_lag_s': 0.25,
            'min_corr': 0.3, 'min_anchors': 5, 'reject_outliers': True,
        },
        'preserve_annotations': True, 'on_alignment_error': 'skip',
        'alignment_info': {
            'trials': trials, 'failures': [{'error': 'missing'}] * failures,
            'n_trials_found': len(trials) + failures, 'n_trials_aligned': len(trials),
            'n_trials_failed': failures, 'n_trials_epoched': len(trials),
            'summary': summarize_alignments(trials),
        },
    }


def write_pipeline_json(path: Path, info: dict) -> Path:
    pipeline = AlmKanal(steps=[])
    pipeline.info = {'steps': ['EpochTRF'], 'steps_info': {'EpochTRF': {'TRF_info': info}}}
    pipeline.generate_json(str(path))
    return path


def test_preprocessing_report_pools_all_trials_after_json_truncation(tmp_path: Path) -> None:
    first = list(np.arange(75, dtype=float))
    second = [-100.0, 200.0]
    files = [
        write_pipeline_json(tmp_path / 'first.json', make_info(first, failures=2)),
        write_pipeline_json(tmp_path / 'second.json', make_info(second)),
    ]
    exported = json.loads(files[0].read_text())['EpochTRF']['TRF_info']['alignment_info']
    assert exported['trials'] == {'length': 75}
    assert exported['summary']['drift_us_per_s']['n'] == 75
    ctx = build_context_from_files(files)
    results = ctx.steps[0].results
    stats = results['metrics']['drift_us_per_s']
    assert stats['n'] == 77
    assert stats['mean'] == pytest.approx(np.mean(first + second))
    assert stats['sd'] == pytest.approx(np.std(first + second))
    assert stats['min'] == -100
    assert stats['max'] == 200
    assert results['n_trials_failed'] == 2
    assert results['n_recordings'] == 2

    output = preprocessing_report(files, tmp_path / 'methods.md').read_text()
    assert '77 of 79 trials' in output
    assert '2 failed and were skipped' in output
    assert f'{np.mean(first + second):.3f} ± {np.std(first + second):.3f} µs/s' in output
    assert 'residual RMS timing error 0.200 ± 0.000 ms' in output
    assert 'physical-delay correction of 16.500 ms was applied after realignment' in output
    assert 'positive values advance neural events relative to the WAV features to compensate playback-to-ear delay' in output
    assert 'The WAV feature channels were left unchanged.' in output
    assert 'sampling grid was 16.000 ms' in output


def test_report_checks_configuration_but_allows_different_results(tmp_path: Path) -> None:
    one = make_info([1.0])
    two = deepcopy(one)
    two['hw_delay_s'] = 0.02
    files = [write_pipeline_json(tmp_path / 'one.json', one), write_pipeline_json(tmp_path / 'two.json', two)]
    with pytest.raises(ValueError, match="Settings mismatch in 'EpochTRF'"):
        build_context_from_files(files)


@pytest.mark.parametrize('info', [
    {'epoch_len_s': 5.0},
    {'epoch_len_s': 5.0, 'realign_audio': False, 'hw_delay_s': -0.0165},
])
def test_old_or_disabled_alignment_reports_remain_supported(tmp_path: Path, info: dict) -> None:
    path = write_pipeline_json(tmp_path / 'legacy.json', info)
    text = preprocessing_report([path], tmp_path / 'methods.md').read_text()
    assert 'Data were epoched in 5.0s long epochs.' in text
    assert 'clock drift' not in text


def test_one_trial_summary_has_zero_sd(tmp_path: Path) -> None:
    path = write_pipeline_json(tmp_path / 'one.json', make_info([5.0]))
    metrics = build_context_from_files([path]).steps[0].results['metrics']
    assert metrics['drift_us_per_s'] == {'n': 1, 'mean': 5.0, 'sd': 0.0, 'min': 5.0, 'max': 5.0}
