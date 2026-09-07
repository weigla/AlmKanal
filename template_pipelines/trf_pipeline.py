# Basic TRF analysis pipeline
from pathlib import Path

import joblib
import mne
from plus_slurm import Job

from almkanal import AlmKanal, EpochTRF, TRFSpanSpec


class TRFPipe(Job):
    job_data_folder = 'data_meg'

    def run(
        self,
        subject_id: str,
        data_path: str,
        audio_path: str,
        hw_delay_s: float = -0.0165,
        epoch_len_s: float = 5.0,
    ) -> None:
        full_path = Path(data_path) / f'{subject_id}_raw.fif'
        raw = mne.io.read_raw(full_path, preload=True)

        # Preserve the stim and recorded audio channels for trial realignment.
        pick_dict = {
            'meg': True,
            'eog': True,
            'ecg': True,
            'eeg': False,
            'stim': True,
            'misc': True,
        }
        onset_trigger_to_wav = {
            11: 'story_a.wav',
            12: 'story_b.wav',
        }

        def make_spans(raw: mne.io.BaseRaw) -> TRFSpanSpec:
            # Repeated presentations receive distinct trial labels.
            return TRFSpanSpec.from_events(
                raw,
                onset_trigger_to_wav=onset_trigger_to_wav,
                end_triggers=99,
                stim_channel='STI101',
            )

        ak = AlmKanal(
            pick_params=pick_dict,
            steps=[
                EpochTRF(
                    gen_span_spec=make_spans,
                    base_audio_path=audio_path,
                    audio_channels=['AUDIO001'],
                    alignment_kwargs={'window_s': 10.0, 'step_s': 5.0, 'min_corr': 0.3},
                    # Realign first, then apply the separate physical delay.
                    hw_delay_s=hw_delay_s,
                    epoch_len_s=epoch_len_s,
                    on_alignment_error='raise',
                ),
                # Keep sufficient audio bandwidth until EpochTRF has run.
                # Add filtering and downsampling afterward as needed.
            ],
        )
        epochs, report = ak.run(raw)

        output_path = Path(self.full_output_path)
        report.save(output_path.with_suffix('.html'), overwrite=True)
        ak.generate_json(str(output_path.with_suffix('.json')))
        joblib.dump((epochs, report), self.full_output_path)


# After all jobs finish, aggregate their JSON files into Methods text:
# from almkanal import preprocessing_report
# preprocessing_report(files, 'methods.md')
# This includes drift/offset statistics and the subsequent physical delay.
