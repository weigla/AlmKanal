from __future__ import annotations

import warnings
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import librosa
import mne
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, correlate, hilbert, resample_poly, sosfiltfilt
from scipy.stats import theilslopes

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

VARIANCE_EPSILON = 1e-12
ENERGY_EPSILON = 1e-20


def resolve_stim_channel(raw: mne.io.BaseRaw, stim_channel: str | None) -> str:
    """Return an explicit stim-channel name, requiring an unambiguous choice."""
    if stim_channel is not None:
        if stim_channel not in raw.ch_names:
            raise ValueError(f'Stim channel {stim_channel!r} is not present in the Raw object.')
        return stim_channel

    picks = mne.pick_types(raw.info, meg=False, stim=True)
    if len(picks) != 1:
        names = [raw.ch_names[pick] for pick in picks]
        raise RuntimeError(f'Specify stim_channel explicitly; detected stim channels: {names}')
    return raw.ch_names[picks[0]]


def find_audio_trials(
    raw: mne.io.BaseRaw,
    onset_trigger_to_wav: Mapping[int, str | Path],
    end_triggers: int | Sequence[int],
    *,
    stim_channel: str | None = None,
) -> list[dict[str, Any]]:
    """Find complete trials while ignoring all unrelated trigger values."""
    if not onset_trigger_to_wav:
        raise ValueError('onset_trigger_to_wav must contain at least one onset-trigger/WAV pair.')

    stim_channel = resolve_stim_channel(raw, stim_channel)
    onset_codes = {int(code) for code in onset_trigger_to_wav}
    end_codes = {int(end_triggers)} if isinstance(end_triggers, int) else {int(code) for code in end_triggers}
    if not end_codes:
        raise ValueError('end_triggers must contain at least one trigger value.')
    overlap = onset_codes & end_codes
    if overlap:
        raise ValueError(f'Trigger values cannot be both onset and end triggers: {sorted(overlap)}')

    events = mne.find_events(
        raw,
        stim_channel=stim_channel,
        shortest_event=1,
        consecutive=True,
        verbose=False,
    )

    trials: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for event_sample, _, raw_event_value in events:
        sample = int(event_sample)
        event_value = int(raw_event_value)

        if event_value in onset_codes:
            if current is not None:
                raise RuntimeError('New audio onset before the preceding end trigger.')
            current = {
                'onset_code': event_value,
                'wav_file': str(onset_trigger_to_wav[event_value]),
                'onset_sample': sample,
            }
        elif current is not None and event_value in end_codes:
            current['end_code'] = event_value
            current['end_sample'] = sample
            trials.append(current)
            current = None

    if current is not None:
        warnings.warn('The final audio trial has no end trigger and was ignored.', stacklevel=2)

    return trials


def _make_sync_envelope(
    signal: NDArray[Any],
    sfreq: float,
    *,
    target_sfreq: float,
    band: tuple[float, float],
    envelope_lowpass: float,
) -> NDArray[np.float64]:
    signal = np.asarray(signal, dtype=float)
    signal -= np.mean(signal)
    if np.std(signal) < VARIANCE_EPSILON:
        raise ValueError('Audio signal has essentially zero variance.')

    nyquist = sfreq / 2.0
    low = band[0] / nyquist
    high = min(band[1] / nyquist, 0.99)
    if low <= 0 or low >= high:
        raise ValueError(f'Invalid audio band {band} for sfreq={sfreq}.')

    signal = sosfiltfilt(butter(4, [low, high], btype='bandpass', output='sos'), signal)
    envelope = np.abs(hilbert(signal))

    lowpass = envelope_lowpass / nyquist
    if lowpass >= 1:
        raise ValueError('envelope_lowpass must be below the Nyquist frequency.')
    envelope = sosfiltfilt(butter(4, lowpass, btype='lowpass', output='sos'), envelope)

    if not np.isclose(sfreq, target_sfreq):
        ratio = Fraction(target_sfreq / sfreq).limit_denominator(10_000)
        envelope = resample_poly(envelope, up=ratio.numerator, down=ratio.denominator)

    envelope -= np.mean(envelope)
    scale = np.std(envelope)
    if scale < VARIANCE_EPSILON:
        raise ValueError('Envelope has essentially zero variance.')
    return np.asarray(envelope / scale, dtype=float)


def _normalized_xcorr_valid(signal: NDArray[np.float64], template: NDArray[np.float64]) -> NDArray[np.float64]:
    signal = np.asarray(signal, dtype=float)
    template = np.asarray(template, dtype=float)
    n_template = len(template)
    if len(signal) < n_template:
        raise ValueError('signal must be longer than template.')

    template_centered = template - np.mean(template)
    template_energy = np.sum(template_centered**2)
    if template_energy < ENERGY_EPSILON:
        return np.full(len(signal) - n_template + 1, np.nan)

    numerator = correlate(signal, template_centered, mode='valid', method='fft')
    cumulative = np.concatenate(([0.0], np.cumsum(signal)))
    cumulative_squared = np.concatenate(([0.0], np.cumsum(signal**2)))
    window_sum = cumulative[n_template:] - cumulative[:-n_template]
    window_sum_squared = cumulative_squared[n_template:] - cumulative_squared[:-n_template]
    window_energy = np.maximum(window_sum_squared - window_sum**2 / n_template, 0.0)
    denominator = np.sqrt(window_energy * template_energy)

    result = np.full_like(numerator, np.nan, dtype=float)
    good = denominator > ENERGY_EPSILON
    result[good] = numerator[good] / denominator[good]
    return result


def _estimate_window_lag(
    recorded: NDArray[np.float64],
    reference: NDArray[np.float64],
    *,
    center_s: float,
    sfreq: float,
    window_s: float,
    max_lag_s: float,
) -> tuple[float | None, float | None]:
    n_window = int(round(window_s * sfreq))
    n_max_lag = int(round(max_lag_s * sfreq))
    reference_start = int(round((center_s - window_s / 2) * sfreq))
    reference_stop = reference_start + n_window
    recorded_start = reference_start - n_max_lag
    recorded_stop = reference_stop + n_max_lag

    if reference_start < 0 or recorded_start < 0 or reference_stop > len(reference) or recorded_stop > len(recorded):
        return None, None

    reference_segment = reference[reference_start:reference_stop]
    recorded_segment = recorded[recorded_start:recorded_stop]
    if np.std(reference_segment) < VARIANCE_EPSILON or np.std(recorded_segment) < VARIANCE_EPSILON:
        return None, None

    correlations = _normalized_xcorr_valid(recorded_segment, reference_segment)
    if not np.any(np.isfinite(correlations)):
        return None, None

    peak_index = int(np.nanargmax(correlations))
    peak_correlation = float(correlations[peak_index])
    peak_position = float(peak_index)
    if 0 < peak_index < len(correlations) - 1:
        left, center, right = correlations[peak_index - 1 : peak_index + 2]
        denominator = left - 2 * center + right
        if abs(denominator) > VARIANCE_EPSILON:
            delta = 0.5 * (left - right) / denominator
            if abs(delta) <= 1:
                peak_position += delta

    lag_s = (peak_position - n_max_lag) / sfreq
    return float(lag_s), peak_correlation


def estimate_raw_wav_alignment(  # noqa: C901, PLR0912, PLR0915
    raw_trial: mne.io.BaseRaw,
    wav_file: str | Path,
    *,
    audio_channels: tuple[str, ...],
    sync_sfreq: float = 500.0,
    audio_band: tuple[float, float] = (80.0, 2000.0),
    envelope_lowpass: float = 30.0,
    window_s: float = 10.0,
    step_s: float = 5.0,
    max_lag_s: float = 0.250,
    min_corr: float = 0.30,
    min_anchors: int = 5,
    reject_outliers: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Estimate ``t_raw = offset + clock_slope * t_wav`` for one trial."""
    raw_sfreq = float(raw_trial.info['sfreq'])
    missing = [channel for channel in audio_channels if channel not in raw_trial.ch_names]
    if missing:
        raise ValueError(f'Missing audio channels: {missing}')

    recorded = raw_trial.get_data(picks=list(audio_channels))
    wav, wav_sfreq = librosa.load(Path(wav_file), sr=None, mono=False)
    wav = np.asarray(wav, dtype=float)
    if wav.ndim == 1:
        wav = wav[np.newaxis, :]
    wav_sfreq = float(wav_sfreq)
    wav_duration_s = wav.shape[-1] / wav_sfreq

    recorded_sync = np.vstack(
        [
            _make_sync_envelope(
                channel,
                raw_sfreq,
                target_sfreq=sync_sfreq,
                band=audio_band,
                envelope_lowpass=envelope_lowpass,
            )
            for channel in recorded
        ]
    )
    wav_sync = np.vstack(
        [
            _make_sync_envelope(
                channel,
                wav_sfreq,
                target_sfreq=sync_sfreq,
                band=audio_band,
                envelope_lowpass=envelope_lowpass,
            )
            for channel in wav
        ]
    )

    usable_duration = min(recorded_sync.shape[-1], wav_sync.shape[-1]) / sync_sfreq
    first_center = window_s / 2 + max_lag_s
    last_center = usable_duration - window_s / 2 - max_lag_s
    if last_center <= first_center:
        raise RuntimeError('Trial is too short for the requested synchronization settings.')
    centers = np.arange(first_center, last_center + 1e-12, step_s)

    pair_results: dict[tuple[int, int], dict[str, Any]] = {}
    for recorded_index in range(recorded_sync.shape[0]):
        for wav_index in range(wav_sync.shape[0]):
            pair_lags: list[float] = []
            pair_correlations: list[float] = []
            for center in centers:
                lag, correlation = _estimate_window_lag(
                    recorded_sync[recorded_index],
                    wav_sync[wav_index],
                    center_s=float(center),
                    sfreq=sync_sfreq,
                    window_s=window_s,
                    max_lag_s=max_lag_s,
                )
                if lag is not None and correlation is not None and np.isfinite(correlation):
                    pair_lags.append(lag)
                    pair_correlations.append(correlation)
            if pair_correlations:
                pair_results[(recorded_index, wav_index)] = {
                    'median_corr': float(np.median(pair_correlations)),
                    'mean_corr': float(np.mean(pair_correlations)),
                    'lags': np.asarray(pair_lags, dtype=float),
                    'corrs': np.asarray(pair_correlations, dtype=float),
                }

    if not pair_results:
        raise RuntimeError('No usable recorded/WAV channel pair could be evaluated.')
    best_pair = max(pair_results, key=lambda pair: pair_results[pair]['median_corr'])
    best_recorded_index, best_wav_index = best_pair

    candidate_times: list[float] = []
    candidate_lags: list[float] = []
    candidate_correlations: list[float] = []
    for center in centers:
        lag, correlation = _estimate_window_lag(
            recorded_sync[best_recorded_index],
            wav_sync[best_wav_index],
            center_s=float(center),
            sfreq=sync_sfreq,
            window_s=window_s,
            max_lag_s=max_lag_s,
        )
        if lag is not None and correlation is not None and np.isfinite(correlation):
            candidate_times.append(float(center))
            candidate_lags.append(lag)
            candidate_correlations.append(correlation)

    candidate_times_array = np.asarray(candidate_times, dtype=float)
    candidate_lags_array = np.asarray(candidate_lags, dtype=float)
    candidate_correlations_array = np.asarray(candidate_correlations, dtype=float)
    accepted = candidate_correlations_array >= min_corr
    anchor_times = candidate_times_array[accepted]
    anchor_lags = candidate_lags_array[accepted]
    anchor_correlations = candidate_correlations_array[accepted]
    if len(anchor_times) < min_anchors:
        raise RuntimeError(
            f'Only {len(anchor_times)} synchronization anchors were accepted; required {min_anchors}. '
            f'Best median correlation: {pair_results[best_pair]["median_corr"]:.3f}.'
        )

    initial_drift, initial_offset, _, _ = theilslopes(anchor_lags, anchor_times)
    initial_residuals = anchor_lags - (initial_offset + initial_drift * anchor_times)
    inliers = np.ones(len(anchor_times), dtype=bool)
    outlier_threshold_s: float | None = None
    if reject_outliers:
        residual_median = np.median(initial_residuals)
        mad = np.median(np.abs(initial_residuals - residual_median))
        outlier_threshold_s = max(2.0 / sync_sfreq, 5.0 * 1.4826 * mad)
        inliers = np.abs(initial_residuals - residual_median) <= outlier_threshold_s
    if np.sum(inliers) < min_anchors:
        raise RuntimeError(f'Only {np.sum(inliers)} anchors remain after outlier rejection; required {min_anchors}.')

    drift, offset = np.polyfit(anchor_times[inliers], anchor_lags[inliers], 1)
    clock_slope = 1.0 + drift
    if clock_slope <= 0:
        raise RuntimeError(f'Invalid clock slope: {clock_slope}')
    residuals = anchor_lags - (offset + drift * anchor_times)

    alignment: dict[str, Any] = {
        'offset_s': float(offset),
        'drift_fraction': float(drift),
        'clock_slope': float(clock_slope),
        'resample_sfreq': float(raw_sfreq / clock_slope),
        'raw_sfreq': raw_sfreq,
        'wav_sfreq': wav_sfreq,
        'wav_duration_s': float(wav_duration_s),
        'drift_us_per_s': float(drift * 1e6),
        'total_drift_ms': float(drift * wav_duration_s * 1000),
        'residual_rms_ms': float(np.sqrt(np.mean(residuals[inliers] ** 2)) * 1000),
        'residual_max_ms': float(np.max(np.abs(residuals[inliers])) * 1000),
        'selected_recorded_channel': audio_channels[best_recorded_index],
        'selected_wav_channel': int(best_wav_index),
        'candidate_times_s': candidate_times_array,
        'candidate_lags_s': candidate_lags_array,
        'candidate_corrs': candidate_correlations_array,
        'anchor_times_s': anchor_times,
        'anchor_lags_s': anchor_lags,
        'anchor_corrs': anchor_correlations,
        'anchor_inliers': inliers,
        'outlier_threshold_s': outlier_threshold_s,
    }

    if verbose:
        print(
            f'Alignment {Path(wav_file).name}: offset={offset * 1000:+.3f} ms, '
            f'drift={drift * 1e6:+.3f} us/s, residual RMS={alignment["residual_rms_ms"]:.3f} ms'
        )
    return alignment


def _warp_annotations(
    raw: mne.io.BaseRaw,
    *,
    source_start_sample: int,
    source_stop_sample: int,
    clock_slope: float,
    output_duration_s: float,
) -> mne.Annotations:
    """Crop and clock-warp annotations into an aligned trial's time base."""
    annotations = raw.annotations
    if len(annotations) == 0:
        return mne.Annotations([], [], [])

    starts = raw.time_as_index(
        annotations.onset,
        use_rounding=True,
        origin=annotations.orig_time,
    )
    stops = raw.time_as_index(
        annotations.onset + annotations.duration,
        use_rounding=True,
        origin=annotations.orig_time,
    )
    if annotations.orig_time is not None:
        starts += raw.first_samp
        stops += raw.first_samp

    sfreq = float(raw.info['sfreq'])
    output_onsets: list[float] = []
    output_durations: list[float] = []
    output_descriptions: list[str] = []
    output_ch_names: list[tuple[str, ...]] = []

    for index, (annotation_start, annotation_stop) in enumerate(zip(starts, stops, strict=True)):
        start = int(annotation_start)
        stop = int(annotation_stop)
        is_point = stop == start
        if is_point:
            if not source_start_sample <= start < source_stop_sample:
                continue
            mapped_onset = (start - source_start_sample) / (sfreq * clock_slope)
            if not 0 <= mapped_onset < output_duration_s:
                continue
            mapped_duration = 0.0
        else:
            overlap_start = max(start, source_start_sample)
            overlap_stop = min(stop, source_stop_sample)
            if overlap_stop <= overlap_start:
                continue
            mapped_onset = (overlap_start - source_start_sample) / (sfreq * clock_slope)
            mapped_stop = (overlap_stop - source_start_sample) / (sfreq * clock_slope)
            mapped_onset = max(0.0, mapped_onset)
            mapped_stop = min(output_duration_s, mapped_stop)
            if mapped_stop <= mapped_onset:
                continue
            mapped_duration = mapped_stop - mapped_onset

        output_onsets.append(float(mapped_onset))
        output_durations.append(float(mapped_duration))
        output_descriptions.append(str(annotations.description[index]))
        output_ch_names.append(tuple(annotations.ch_names[index]))

    return mne.Annotations(
        output_onsets,
        output_durations,
        output_descriptions,
        orig_time=None,
        ch_names=output_ch_names,
    )


def apply_raw_wav_alignment(
    raw: mne.io.BaseRaw,
    onset_sample: int,
    alignment: Mapping[str, Any],
    *,
    preserve_annotations: bool = True,
    verbose: bool = True,
) -> tuple[mne.io.RawArray, dict[str, Any]]:
    """Extract and clock-correct one trial from a continuous Raw object."""
    raw_sfreq = float(raw.info['sfreq'])
    expected_sfreq = float(alignment['raw_sfreq'])
    if not np.isclose(raw_sfreq, expected_sfreq):
        raise ValueError(f'Raw sampling frequency {raw_sfreq} does not match alignment {expected_sfreq}.')

    onset_sample = int(onset_sample)
    onset_index = onset_sample - raw.first_samp
    if onset_index < 0 or onset_index >= raw.n_times:
        raise ValueError('onset_sample is outside the supplied Raw object.')

    offset_s = float(alignment['offset_s'])
    clock_slope = float(alignment['clock_slope'])
    resample_sfreq = float(alignment['resample_sfreq'])
    wav_duration_s = float(alignment['wav_duration_s'])
    source_start_float = onset_index + offset_s * raw_sfreq
    source_end_float = onset_index + (offset_s + clock_slope * wav_duration_s) * raw_sfreq
    if source_start_float < 0:
        raise RuntimeError(f'Required trial begins {-source_start_float / raw_sfreq:.6f} s before the Raw object.')
    if source_end_float > raw.n_times:
        raise RuntimeError(
            f'Required trial extends {(source_end_float - raw.n_times) / raw_sfreq:.6f} s beyond the Raw object.'
        )

    source_start_index = int(round(source_start_float))
    source_end_index = min(int(np.ceil(source_end_float)) + 4, raw.n_times)
    segment = raw.copy().crop(
        tmin=source_start_index / raw_sfreq,
        tmax=(source_end_index - 1) / raw_sfreq,
    )
    segment.load_data()
    n_before_resample = segment.n_times
    segment.resample(sfreq=resample_sfreq, npad='auto')
    n_after_resample = segment.n_times

    target_n_samples = int(round(wav_duration_s * raw_sfreq))
    if segment.n_times < target_n_samples:
        raise RuntimeError(
            f'Clock correction produced {segment.n_times} samples; {target_n_samples} samples are required.'
        )
    corrected_data = segment.get_data(start=0, stop=target_n_samples)
    aligned = mne.io.RawArray(corrected_data, raw.info.copy(), first_samp=0, verbose=False)

    source_start_sample = raw.first_samp + source_start_index
    logical_source_stop = min(
        source_start_sample + int(np.ceil(target_n_samples * clock_slope)),
        raw.first_samp + raw.n_times,
    )
    if preserve_annotations:
        aligned.set_annotations(
            _warp_annotations(
                raw,
                source_start_sample=source_start_sample,
                source_stop_sample=logical_source_stop,
                clock_slope=clock_slope,
                output_duration_s=target_n_samples / raw_sfreq,
            )
        )

    actual_source_start_s = (source_start_index - onset_index) / raw_sfreq
    applied_info = {
        'onset_sample': onset_sample,
        'source_start_sample': source_start_sample,
        'source_end_sample': raw.first_samp + source_end_index - 1,
        'offset_s': offset_s,
        'actual_source_start_relative_to_trigger_s': actual_source_start_s,
        'source_start_rounding_error_s': actual_source_start_s - offset_s,
        'clock_slope': clock_slope,
        'resample_sfreq': resample_sfreq,
        'wav_duration_s': wav_duration_s,
        'n_samples_before_resample': n_before_resample,
        'n_samples_after_resample': n_after_resample,
        'target_n_samples': target_n_samples,
        'n_samples_aligned': aligned.n_times,
        'aligned_duration_s': aligned.n_times / raw_sfreq,
        'n_annotations_preserved': len(aligned.annotations),
    }
    if verbose:
        print(
            f'Applied alignment: source={n_before_resample} samples, corrected={aligned.n_times} samples, '
            f'output sfreq={aligned.info["sfreq"]:.6f} Hz'
        )
    return aligned, applied_info