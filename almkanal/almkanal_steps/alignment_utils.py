from __future__ import annotations

import html
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import mne
import numpy as np
from attrs import define

from almkanal.almkanal import AlmKanalStep
from almkanal.stim_utils.alignment_utils import (
    apply_raw_wav_alignment,
    estimate_raw_wav_alignment,
    find_audio_trials,
    resolve_stim_channel,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@define
class AudioTrialRealignment(AlmKanalStep):
    """Realign trigger-delimited audio trials to WAV time and concatenate them.

    ``onset_trigger_to_wav`` maps each trial-onset trigger to its corresponding
    WAV file. Any event that is neither a configured onset nor end trigger is
    ignored. Relative WAV paths are resolved below ``wav_root``.

    The returned object is an MNE ``Raw`` containing all aligned trials in
    their original order. Corrected onset events are written back to the stim
    channel, and each trial receives a descriptive annotation. When an
    ``Events`` step follows, set ``initial_event=True`` so an onset at the
    first output sample is retained.
    """

    onset_trigger_to_wav: Mapping[int, str | Path]
    end_triggers: int | Sequence[int]
    audio_channels: Sequence[str]
    wav_root: str | Path | None = None
    stim_channel: str | None = None
    alignment_kwargs: Mapping[str, Any] | None = None

    preserve_annotations: bool = True
    on_alignment_error: Literal['raise', 'skip'] = 'raise'
    concat_preload: bool | str | None = None
    verbose: bool = True

    must_be_before: tuple = (
        'Events',
        'Epochs',
        'ForwardModel',
        'SpatialFilter',
        'SourceReconstruction',
    )
    must_be_after: tuple = ()

    def __attrs_post_init__(self) -> None:
        if not self.onset_trigger_to_wav:
            raise ValueError('onset_trigger_to_wav must contain at least one onset-trigger/WAV pair.')
        if isinstance(self.audio_channels, str) or not self.audio_channels:
            raise ValueError('audio_channels must contain at least one recorded audio channel.')
        if self.on_alignment_error not in {'raise', 'skip'}:
            raise ValueError("on_alignment_error must be either 'raise' or 'skip'.")

    def _resolve_wav_path(self, wav_file: str | Path) -> Path:
        path = Path(wav_file)
        if not path.is_absolute() and self.wav_root is not None:
            path = Path(self.wav_root) / path
        return path

    def run(self, data: mne.io.BaseRaw, info: dict) -> dict[str, Any]:  # noqa: C901, PLR0915
        if not isinstance(data, mne.io.BaseRaw):
            raise TypeError('AudioTrialRealignment requires continuous mne.io.BaseRaw input.')

        stim_channel = resolve_stim_channel(data, self.stim_channel)
        trials = find_audio_trials(
            data,
            self.onset_trigger_to_wav,
            self.end_triggers,
            stim_channel=stim_channel,
        )
        if not trials:
            raise RuntimeError('No complete audio trials were found.')

        sfreq = float(data.info['sfreq'])
        aligned_trials: list[mne.io.BaseRaw] = []
        trial_records: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        concatenated_start_sample = 0
        estimate_kwargs = dict(self.alignment_kwargs or {})
        estimate_kwargs['audio_channels'] = self.audio_channels
        estimate_kwargs['verbose'] = self.verbose

        for trial_index, trial in enumerate(trials):
            wav_file = self._resolve_wav_path(trial['wav_file'])
            try:
                trial_tmin = (trial['onset_sample'] - data.first_samp) / sfreq
                trial_tmax = (trial['end_sample'] - data.first_samp) / sfreq
                raw_trial = data.copy().crop(tmin=trial_tmin, tmax=trial_tmax)
                alignment = estimate_raw_wav_alignment(
                    raw_trial,
                    wav_file,
                    **estimate_kwargs,
                )
                aligned = apply_raw_wav_alignment(
                    data,
                    trial['onset_sample'],
                    alignment,
                    preserve_annotations=self.preserve_annotations,
                    verbose=self.verbose,
                )
            except (OSError, RuntimeError, ValueError) as error:
                failure = {
                    'trial_index': trial_index,
                    'onset_code': trial['onset_code'],
                    'original_onset_sample': trial['onset_sample'],
                    'wav_file': str(wav_file),
                    'error': str(error),
                }
                failures.append(failure)
                if self.on_alignment_error == 'raise':
                    raise RuntimeError(
                        f'Audio alignment failed for trial {trial_index} '
                        f'(trigger={trial["onset_code"]}, WAV={wav_file}).'
                    ) from error
                warnings.warn(f'Skipping audio trial {trial_index}: {error}', stacklevel=2)
                continue

            description = f'audio_trial/{trial_index:03d}/trigger={trial["onset_code"]}/wav={wav_file.name}'
            aligned.annotations.append(aligned.first_time, 0.0, description)

            stop_sample = concatenated_start_sample + int(aligned.n_times)
            trial_records.append(
                {
                    'trial_index': trial_index,
                    'onset_code': trial['onset_code'],
                    'end_code': trial['end_code'],
                    'wav_file': str(wav_file),
                    'original_onset_sample': trial['onset_sample'],
                    'original_end_sample': trial['end_sample'],
                    'concatenated_start_sample': concatenated_start_sample,
                    'concatenated_stop_sample_exclusive': stop_sample,
                    'concatenated_start_s': concatenated_start_sample / sfreq,
                    'duration_s': float(aligned.n_times / sfreq),
                    'offset_s': alignment['offset_s'],
                    'clock_slope': alignment['clock_slope'],
                    'drift_us_per_s': alignment['drift_us_per_s'],
                    'residual_rms_ms': alignment['residual_rms_ms'],
                    'median_correlation': alignment['median_correlation'],
                    'n_anchor_inliers': alignment['n_anchor_inliers'],
                    'selected_recorded_channel': alignment['selected_recorded_channel'],
                    'selected_wav_channel': alignment['selected_wav_channel'],
                }
            )
            concatenated_start_sample = stop_sample
            aligned_trials.append(aligned)

        if not aligned_trials:
            raise RuntimeError('No audio trials could be aligned successfully.')

        concatenated = mne.concatenate_raws(
            aligned_trials,
            preload=self.concat_preload,
            verbose=self.verbose,
        )

        # The original onset can precede WAV time zero and therefore be absent
        # from an aligned crop. Install one corrected event per trial so the
        # downstream Events -> Epochs path continues to work.
        events = np.asarray(
            [
                [concatenated.first_samp + trial['concatenated_start_sample'], 0, trial['onset_code']]
                for trial in trial_records
            ],
            dtype=int,
        )
        concatenated.add_events(events, stim_channel=stim_channel, replace=True)

        return {
            'data': concatenated,
            'realignment_info': {
                'events': events,
                'trials': trial_records,
                'failures': failures,
                'stim_channel': stim_channel,
                'audio_channels': self.audio_channels,
                'onset_trigger_to_wav': {str(code): str(path) for code, path in self.onset_trigger_to_wav.items()},
                'end_triggers': self.end_triggers,
                'alignment_kwargs': dict(self.alignment_kwargs or {}),
                'n_trials_found': len(trials),
                'n_trials_aligned': len(aligned_trials),
                'n_trials_failed': len(failures),
                'output_n_times': int(concatenated.n_times),
                'output_duration_s': float(concatenated.n_times / sfreq),
                'preserve_annotations': self.preserve_annotations,
                'on_alignment_error': self.on_alignment_error,
            },
        }

    def reports(self, data: mne.io.BaseRaw, report: mne.Report, info: dict) -> None:
        realignment_info = info['AudioTrialRealignment']['realignment_info']
        rows = []
        for trial in realignment_info['trials']:
            rows.append(
                '<tr>'
                f'<td>{trial["trial_index"]}</td>'
                f'<td>{trial["onset_code"]}</td>'
                f'<td>{html.escape(Path(trial["wav_file"]).name)}</td>'
                f'<td>{trial["duration_s"]:.3f}</td>'
                f'<td>{trial["offset_s"] * 1000:.3f}</td>'
                f'<td>{trial["drift_us_per_s"]:.3f}</td>'
                f'<td>{trial["median_correlation"]:.3f}</td>'
                f'<td>{trial["residual_rms_ms"]:.3f}</td>'
                '</tr>'
            )
        table = (
            '<table><thead><tr><th>Trial</th><th>Trigger</th><th>WAV</th>'
            '<th>Duration (s)</th><th>Offset (ms)</th><th>Drift (us/s)</th>'
            '<th>Median r</th><th>Residual RMS (ms)</th></tr></thead><tbody>'
            + ''.join(rows) +
            '</tbody></table>'
        )
        report.add_html(table, title='Audio trial realignment')
