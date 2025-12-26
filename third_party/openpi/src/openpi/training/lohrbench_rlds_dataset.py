# src/openpi/training/lohrbench_rlds_dataset.py

from collections.abc import Sequence
import dataclasses
import logging
import numpy as np
import openpi.shared.download as download


@dataclasses.dataclass
class RLDSDataset:
    name: str
    version: str
    weight: float
    filter_dict_path: str | None = None  # keep for compatibility, but unused for now


class LohrbenchRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[RLDSDataset],
        *,
        shuffle: bool = True,
        action_chunk_size: int = 16,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
    ):
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")

        assert abs(sum(d.weight for d in datasets) - 1.0) < 1e-6, "Dataset weights must sum to 1.0"

        def prepare_single_dataset(ds_cfg: RLDSDataset):
            builder = tfds.builder(ds_cfg.name, data_dir=data_dir, version=ds_cfg.version)

            dataset = dl.DLataset.from_rlds(
                builder,
                split="train",
                shuffle=shuffle,
                num_parallel_reads=num_parallel_reads,
            )

            dataset = dataset.repeat()

            def restructure(traj):
                obs = traj["observation"]
                traj_len = tf.shape(traj["action"])[0]
                
                # Handle language instruction
                lang_instr = traj["language_instruction"]
                
                # Strategy: always extract a scalar, then tile
                # If scalar: use directly
                # If 1D: take first element
                lang_scalar = tf.cond(
                    tf.equal(tf.rank(lang_instr), 0),
                    lambda: lang_instr,
                    lambda: lang_instr[0]
                )
                
                # Broadcast to trajectory length
                prompt = tf.fill([traj_len], lang_scalar)

                return {
                    "actions": traj["action"],  # [T, 8]
                    "observation/image": obs["base_rgb"],
                    "observation/wrist_image": obs["hand_rgb"],
                    "observation/qpos": obs["qpos"],
                    "observation/qvel": obs["qvel"],
                    "prompt": prompt,  # [T]
                }

            dataset = dataset.traj_map(restructure, num_parallel_calls=num_parallel_calls)

            def chunk_actions(traj):
                traj_len = tf.shape(traj["actions"])[0]

                idx = (
                    tf.broadcast_to(tf.range(action_chunk_size)[None], [traj_len, action_chunk_size])
                    + tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, action_chunk_size])
                )
                idx = tf.minimum(idx, traj_len - 1)

                traj["actions"] = tf.gather(traj["actions"], idx)  # [T, H, 8]
                return traj

            dataset = dataset.traj_map(chunk_actions, num_parallel_calls=num_parallel_calls)

            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

            def decode_images(frame):
                # IMPORTANT: your keys are FLAT ("observation/image"), not nested
                frame["observation/image"] = tf.io.decode_image(
                    frame["observation/image"], expand_animations=False, dtype=tf.uint8
                )
                frame["observation/wrist_image"] = tf.io.decode_image(
                    frame["observation/wrist_image"], expand_animations=False, dtype=tf.uint8
                )
                return frame

            dataset = dataset.map(decode_images, num_parallel_calls=num_parallel_calls)
            return dataset


        logging.info(f"Preparing {len(datasets)} LoHRbench RLDS datasets...")
        all_datasets = [prepare_single_dataset(d) for d in datasets]
        weights = [d.weight for d in datasets]

        final_dataset = dl.DLataset.sample_from_datasets(all_datasets, weights=weights)
        final_dataset = final_dataset.shuffle(shuffle_buffer_size)
        final_dataset = final_dataset.batch(batch_size)
        final_dataset = final_dataset.with_ram_budget(1)

        self.dataset = final_dataset

    def __iter__(self):
        for batch in self.dataset.as_numpy_iterator():
            # Convert byte strings to Python strings
            if "prompt" in batch:
                prompts = batch["prompt"]
                if isinstance(prompts, np.ndarray):
                    # Convert to list of Python strings
                    batch["prompt"] = [
                        p.decode('utf-8') if isinstance(p, bytes) else str(p)
                        for p in prompts.flat
                    ]
            yield batch

    def __len__(self):
        # you can hardcode or estimate later
        return 1_000_000
 