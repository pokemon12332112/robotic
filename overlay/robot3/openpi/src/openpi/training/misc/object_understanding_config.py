"""LoRA fine-tuning on the source training split only."""

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import object_understanding_policy
from openpi.training import weight_loaders


def get_object_understanding_configs():
    # Deferred to follow the existing config registration pattern without a circular import.
    from openpi.training.config import DataConfig
    from openpi.training.config import SimpleDataConfig
    from openpi.training.config import TrainConfig

    model = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_horizon=16,
    )
    return [
        TrainConfig(
            name="pi05_object_understanding_lora",
            model=model,
            data=SimpleDataConfig(
                repo_id="local/object-understanding-train",
                data_transforms=object_understanding_policy.data_transforms,
                base_config=DataConfig(
                    prompt_from_task=True,
                    action_sequence_keys=("actions",),
                    repack_transforms=transforms.Group(
                        inputs=[
                            transforms.RepackTransform(
                                {
                                    "observation/image": "image",
                                    "observation/wrist_image": "wrist_image",
                                    "observation/state": "state",
                                    "actions": "actions",
                                    "prompt": "prompt",
                                }
                            )
                        ]
                    ),
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            freeze_filter=model.get_freeze_filter(),
            ema_decay=None,
            batch_size=1,
            num_workers=2,
            num_train_steps=3000,
            # Disk is limited: save at the end of this initial run, without periodic archives.
            save_interval=3000,
            keep_period=None,
            wandb_enabled=False,
        )
    ]
