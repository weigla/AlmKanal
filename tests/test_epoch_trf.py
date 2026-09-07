from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
import pytest
from scipy.io import wavfile

from almkanal import EpochTRF, TRFSpanSpec
from almkanal.almkanal_steps import trf_utils
from almkanal.stim_utils.audio_utils import prepare_audio


@pytest.fixture
def timing_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    sfreq = 500.0
    duration = 4.0
    slope = 1.2  # Large enough to distinguish delay-before from delay-after.
    offset = 0.04
    onset = 0.8
    times = np.arange(3500) / sfreq
    neural = (times - onset - offset) / slope
    raw = mne.io.RawArray(
        np.vstack([neural, np.zeros(len(times))]),
        mne.create_info(['EEG 001', 'recorded'], sfreq, ['eeg', 'misc']),
        first_samp=1234,
        verbose=False,
    )
    wav = tmp_path / 'stimulus.wav'
    wavfile.write(wav, 1000, np.sin(2 * np.pi * 100 * np.arange(4000) / 1000).astype(np.float32))
    alignment = {
        'offset_s': offset, 'clock_slope': slope, 'wav_duration_s': duration,
        'raw_sfreq': sfreq, 'drift_us_per_s': (slope - 1) * 1e6,
        'total_drift_ms': (slope - 1) * duration * 1000,
        'residual_rms_ms': 0.1, 'residual_max_ms': 0.2,
        'median_correlation': 0.99, 'n_anchor_inliers': 8,
    }

    def estimate(raw_trial, wav_file, **kwargs):
        assert raw_trial.first_samp == raw.first_samp + round(onset * sfreq)
        assert kwargs['audio_channels'] == ['recorded']
        if not Path(wav_file).exists():
            raise FileNotFoundError(wav_file)
        return alignment.copy()

    monkeypatch.setattr(trf_utils, 'estimate_raw_wav_alignment', estimate)
    return {
        'raw': raw, 'spec': TRFSpanSpec({'stimulus': (1634, 4134)}),
        'path': tmp_path, 'wav': wav, 'alignment': alignment,
    }


@pytest.mark.parametrize('delay', [-0.2, 0.0, 0.2])
def test_physical_delay_is_applied_after_clock_correction(timing_data, delay) -> None:
    raw = timing_data['raw']
    before = raw.get_data().copy()
    epochs = trf_utils.build_trf_epochs(
        raw, timing_data['spec'], timing_data['path'], audio_channels=['recorded'],
        hw_delay_s=delay, epoch_len_s=1.0, verbose=False,
    )
    assert len(epochs) == 4  # Both endpoints survive either sign of delay.
    neural = epochs.get_data(picks=['EEG 001'])[:, 0]
    expected = np.arange(2000).reshape(4, 500) / 500 + delay
    np.testing.assert_allclose(neural[:, 100:-100], expected[:, 100:-100], atol=0.004)
    audio, _, _, _ = prepare_audio(str(timing_data['wav']), target_fs=500)
    np.testing.assert_allclose(epochs.get_data(picks=['env_rms'])[:, 0], audio[0, :2000].reshape(4, 500))
    np.testing.assert_array_equal(raw.get_data(), before)
    np.testing.assert_allclose(epochs.metadata['wav_t_on'], [0, 1, 2, 3])
    np.testing.assert_allclose(epochs.metadata['t_on'], 0.8 + 0.04 + 1.2 * (np.arange(4) + delay))


def test_annotations_and_epoch_metadata_survive_rejection(timing_data) -> None:
    raw = timing_data['raw']
    raw.set_annotations(mne.Annotations([0.8 + 0.04 + 1.2 * 1.5], [0.12], ['BAD movement']))
    epochs = trf_utils.build_trf_epochs(
        raw, timing_data['spec'], timing_data['path'], audio_channels=['recorded'],
        hw_delay_s=0.2, epoch_len_s=1.0, verbose=False,
    )
    assert len(epochs) == 3
    assert epochs.metadata['epoch_index_in_segment'].tolist() == [0, 2, 3]
    np.testing.assert_allclose(epochs.metadata['wav_t_on'], [0, 2, 3])
    np.testing.assert_allclose(epochs.metadata['t_on'], 0.84 + 1.2 * (np.array([0, 2, 3]) + 0.2))


def test_repeated_wav_trials_keep_their_own_times(timing_data, monkeypatch) -> None:
    raw = mne.concatenate_raws([timing_data['raw'].copy(), timing_data['raw'].copy()], verbose=False)
    spec = TRFSpanSpec(
        {'first': (1634, 4134), 'repeat': (5134, 7634)},
        wav_by_label={'first': 'stimulus.wav', 'repeat': 'stimulus.wav'},
    )
    monkeypatch.setattr(trf_utils, 'estimate_raw_wav_alignment', lambda *args, **kwargs: timing_data['alignment'])
    epochs = trf_utils.build_trf_epochs(
        raw, spec, timing_data['path'], audio_channels=['recorded'],
        hw_delay_s=-0.2, epoch_len_s=1, verbose=False,
    )
    assert len(epochs) == 8
    assert epochs.metadata['label'].tolist() == ['first'] * 4 + ['repeat'] * 4
    np.testing.assert_allclose(epochs.metadata['wav_t_on'], [0, 1, 2, 3] * 2)
    np.testing.assert_allclose(epochs.metadata['t_on'].iloc[4:].to_numpy() - epochs.metadata['t_on'].iloc[:4], 7)
    np.testing.assert_allclose(epochs.get_data()[:4], epochs.get_data()[4:])


def test_alignment_can_use_wav_duration_without_end_sample(timing_data) -> None:
    epochs = trf_utils.build_trf_epochs(
        timing_data['raw'], TRFSpanSpec({'stimulus': (1634, None)}), timing_data['path'],
        audio_channels=['recorded'], hw_delay_s=0.2, epoch_len_s=1, verbose=False,
    )
    assert len(epochs) == 4


def test_alignment_disabled_retains_fixed_delay_processing(timing_data, monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail('Disabled realignment must not estimate alignment.')

    monkeypatch.setattr(trf_utils, 'estimate_raw_wav_alignment', unexpected)
    epochs = trf_utils.build_trf_epochs(
        timing_data['raw'], timing_data['spec'], timing_data['path'],
        hw_delay_s=-0.0165, epoch_len_s=1.0, verbose=False,
    )
    assert len(epochs) == 4
    expected = timing_data['raw'].get_data(picks=['EEG 001'], start=392, stop=2392)
    np.testing.assert_array_equal(epochs.get_data(picks=['EEG 001']).reshape(-1), expected.reshape(-1))


def test_failed_alignment_can_raise_or_skip(timing_data) -> None:
    spec = TRFSpanSpec(
        {'missing': (1634, 4134), 'stimulus': (1634, 4134)},
        wav_by_label={'missing': 'missing.wav'},
    )
    step = EpochTRF(
        lambda raw: spec, timing_data['path'], audio_channels=['recorded'],
        hw_delay_s=0, epoch_len_s=1, verbose=False,
    )
    with pytest.raises(RuntimeError, match='Audio alignment failed for missing'):
        step.run(timing_data['raw'], {})
    step.on_alignment_error = 'skip'
    with pytest.warns(UserWarning, match='Skipping audio trial missing'):
        result = step.run(timing_data['raw'], {})
    alignment = result['TRF_info']['alignment_info']
    assert alignment['n_trials_found'] == 2
    assert alignment['n_trials_failed'] == 1
    assert alignment['n_trials_aligned'] == 1
    assert alignment['summary']['drift_us_per_s']['n'] == 1
    assert len(result['data']) == 4


def test_insufficient_physical_delay_margin_is_an_error(timing_data) -> None:
    with pytest.raises(RuntimeError, match='Audio alignment failed') as exc:
        trf_utils.build_trf_epochs(
            timing_data['raw'], timing_data['spec'], timing_data['path'],
            audio_channels=['recorded'], hw_delay_s=-1.0, epoch_len_s=1, verbose=False,
        )
    assert 'before the Raw object' in str(exc.value.__cause__)


@pytest.mark.parametrize('kwargs, message', [
    ({'audio_channels': []}, 'audio_channels'),
    ({'audio_channels': 'recorded'}, 'audio_channels'),
    ({'alignment_kwargs': {'min_corr': 0.9}}, 'requires audio_channels'),
    ({'hw_delay_s': np.nan}, 'must be finite'),
    ({'epoch_len_s': 0}, 'must be finite'),
    ({'on_alignment_error': 'ignore'}, 'on_alignment_error'),
])
def test_invalid_configuration(timing_data, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        trf_utils.build_trf_epochs(timing_data['raw'], timing_data['spec'], timing_data['path'], **kwargs)
