from .registry import StepSpec, keys_selector, register_step


@register_step('AudioTrialRealignment')
def audio_trial_realignment_spec() -> StepSpec:
    return StepSpec(
        settings_fn=keys_selector(
            'stim_channel',
            'audio_channels',
            'onset_trigger_to_wav',
            'end_triggers',
            'alignment_kwargs',
            'n_trials_found',
            'n_trials_aligned',
            'n_trials_failed',
            'output_duration_s',
            'preserve_annotations',
        )
    )
