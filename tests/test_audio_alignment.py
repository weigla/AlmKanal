from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import mne
import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import butter, hilbert, sosfiltfilt

from almkanal import AudioTrialRealignment
from almkanal.report.stepspecs.registry import get_registry, load_stepspec_package
from almkanal.stim_utils.alignment_utils import (
    apply_raw_wav_alignment,
    estimate_raw_wav_alignment,
    find_audio_trials,
)

RAW_SFREQ = 500.0
WAV_SFREQ = 1_000
WAV_DURATION_S = 8.0
OFFSET_S = 0.040
CLOCK_SLOPE = 1.0015

ALIGNMENT_KWARGS: dict[str, Any] = {
    'sync_sfreq': 500.0,
    'audio_band': (50.0, 220.0),
    'envelope_lowpass': 20.0,
    'window_s': 1.5,
    'step_s': 0.75,
    'max_lag_s': 0.1,
    'min_corr': 0.5,
    'min_anchors': 6,
}


@pytest.fixture(scope='module')
def synthetic_audio_alignment(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    wav_times = np.arange(round(WAV_DURATION_S * WAV_SFREQ)) / WAV_SFREQ
    modulation = sosfiltfilt(
        butter(3, 8.0, fs=WAV_SFREQ, output='sos'),
        rng.normal(size=len(wav_times)),
    )
    modulation /= np.max(np.abs(modulation))
    wav = (0.55 + 0.4 * modulation) * (
        np.sin(2 * np.pi * 110 * wav_times) + 0.35 * np.sin(2 * np.pi * 173 * wav_times)
    )
    wav = (wav / np.max(np.abs(wav))).astype(np.float32)

    trigger_time_s = 0.5
    raw_duration_s = trigger_time_s + OFFSET_S + CLOCK_SLOPE * WAV_DURATION_S + 0.5
    raw_times = np.arange(round(raw_duration_s * RAW_SFREQ)) / RAW_SFREQ
    wav_time_at_raw_samples = (raw_times - trigger_time_s - OFFSET_S) / CLOCK_SLOPE
    recorded = np.interp(wav_time_at_raw_samples, wav_times, wav, left=0.0, right=0.0)
    recorded += rng.normal(scale=0.005, size=len(recorded))

    onset_sample = round(trigger_time_s * RAW_SFREQ)
    end_sample = round((trigger_time_s + OFFSET_S + CLOCK_SLOPE * WAV_DURATION_S + 0.05) * RAW_SFREQ)
    stim = np.zeros(len(raw_times))
    stim[onset_sample : onset_sample + 2] = 11
    stim[end_sample : end_sample + 2] = 99
    info = mne.create_info(
        ['EEG 001', 'MISC flat', 'audio', 'stim'],
        RAW_SFREQ,
        ['eeg', 'misc', 'misc', 'stim'],
    )
    raw = mne.io.RawArray(
        np.vstack([rng.normal(scale=1e-6, size=len(raw_times)), np.zeros(len(raw_times)), recorded, stim]),
        info,
        verbose=False,
    )
    annotation_onset = trigger_time_s + OFFSET_S + CLOCK_SLOPE * 2.0
    raw.set_annotations(mne.Annotations([annotation_onset], [0.2 * CLOCK_SLOPE], ['inside trial']))

    wav_path = Path(tmp_path_factory.mktemp('audio_alignment')) / 'stimulus.wav'
    wavfile.write(wav_path, WAV_SFREQ, wav)
    raw_trial = raw.copy().crop(trigger_time_s, end_sample / RAW_SFREQ)
    alignment = estimate_raw_wav_alignment(
        raw_trial,
        wav_path,
        audio_channels=['MISC flat', 'audio'],
        verbose=False,
        **ALIGNMENT_KWARGS,
    )
    return {
        'raw': raw,
        'wav': wav,
        'wav_times': wav_times,
        'wav_path': wav_path,
        'onset_sample': onset_sample,
        'alignment': alignment,
    }


def test_find_audio_trials_handles_initial_and_unrelated_events() -> None:
    stim = np.array([11, 11, 0, 7, 0, 99, 99, 0], dtype=float)
    raw = mne.io.RawArray(
        stim[np.newaxis],
        mne.create_info(['trigger'], 100.0, ['stim']),
        verbose=False,
    )

    trials = find_audio_trials(raw, {11: 'trial.wav'}, cast('int', np.int64(99)))

    assert trials == [
        {
            'onset_code': 11,
            'wav_file': 'trial.wav',
            'onset_sample': 0,
            'end_code': 99,
            'end_sample': 5,
        }
    ]


def test_estimation_and_application_correct_known_drift(synthetic_audio_alignment: dict[str, Any]) -> None:
    alignment = synthetic_audio_alignment['alignment']
    assert alignment['selected_recorded_channel'] == 'audio'
    assert alignment['offset_s'] == pytest.approx(OFFSET_S, abs=0.002)
    assert alignment['clock_slope'] == pytest.approx(CLOCK_SLOPE, abs=0.0003)
    assert alignment['median_correlation'] > 0.95
    assert alignment['n_anchor_inliers'] >= ALIGNMENT_KWARGS['min_anchors']
    assert 'candidate_times_s' not in alignment

    aligned = apply_raw_wav_alignment(
        synthetic_audio_alignment['raw'],
        synthetic_audio_alignment['onset_sample'],
        alignment,
        verbose=False,
    )

    assert aligned.ch_names == synthetic_audio_alignment['raw'].ch_names
    assert aligned.n_times == round(WAV_DURATION_S * RAW_SFREQ)
    annotation_index = list(aligned.annotations.description).index('inside trial')
    annotation_onset = aligned.annotations.onset[annotation_index] - aligned.first_time
    assert annotation_onset == pytest.approx(2.0, abs=0.004)
    assert aligned.annotations.duration[annotation_index] == pytest.approx(0.2, abs=0.002)

    recorded = aligned.get_data(picks=['audio'])[0]
    wav_at_raw_sfreq = np.interp(
        np.arange(aligned.n_times) / RAW_SFREQ,
        synthetic_audio_alignment['wav_times'],
        synthetic_audio_alignment['wav'],
    )
    bandpass = butter(3, [50.0, 220.0], btype='bandpass', fs=RAW_SFREQ, output='sos')
    recorded_envelope = np.abs(hilbert(sosfiltfilt(bandpass, recorded)))
    wav_envelope = np.abs(hilbert(sosfiltfilt(bandpass, wav_at_raw_sfreq)))
    assert np.corrcoef(recorded_envelope, wav_envelope)[0, 1] > 0.75


def test_audio_trial_realignment_step_writes_corrected_event(
    synthetic_audio_alignment: dict[str, Any],
) -> None:
    wav_path = synthetic_audio_alignment['wav_path']
    step = AudioTrialRealignment(
        onset_trigger_to_wav={11: wav_path.name},
        end_triggers=99,
        audio_channels=['MISC flat', 'audio'],
        wav_root=wav_path.parent,
        stim_channel='stim',
        alignment_kwargs=ALIGNMENT_KWARGS,
        verbose=False,
    )

    result = step.run(synthetic_audio_alignment['raw'], {})
    aligned = result['data']
    events = mne.find_events(
        aligned,
        stim_channel='stim',
        shortest_event=1,
        initial_event=True,
        verbose=False,
    )

    assert events.tolist() == [[aligned.first_samp, 0, 11]]
    assert result['realignment_info']['n_trials_aligned'] == 1
    assert 'alignments' not in result['realignment_info']
    assert any(description.startswith('audio_trial/000/') for description in aligned.annotations.description)


def test_alignment_stepspec_only_compares_configuration() -> None:
    load_stepspec_package('almkanal.report.stepspecs')
    settings = get_registry()['AudioTrialRealignment'].settings_fn(
        {
            'stim_channel': 'stim',
            'audio_channels': ['audio'],
            'n_trials_found': 4,
            'n_trials_aligned': 3,
            'output_duration_s': 24.0,
        }
    )

    assert settings == {'stim_channel': 'stim', 'audio_channels': ['audio']}
