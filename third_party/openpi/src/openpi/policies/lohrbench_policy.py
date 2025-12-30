import dataclasses
import numpy as np
from openpi import transforms
from openpi.models import model as _model

def _to_gripper_scalar(q: np.ndarray) -> np.ndarray:
    """
    qpos/qvel are length 9 in your dataset.
    Indices 0-6: arm joints (7 dimensions)
    Indices 7-8: two gripper finger joints (2 dimensions)
    Convert gripper to a single scalar by averaging.
    """
    if q.shape[-1] >= 9:
        return np.mean(q[..., 7:9], axis=-1, keepdims=True).astype(np.float32)
    # fallback if it's actually 8 already
    return q[..., 7:8].astype(np.float32)

@dataclasses.dataclass(frozen=True)
class LohrbenchInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Make copies to ensure arrays are writable
        img = np.array(data["observation/image"], copy=True)
        wrist = np.array(data["observation/wrist_image"], copy=True)
        
        qpos9 = np.asarray(data["observation/qpos"], dtype=np.float32)
        
        # Extract arm positions (7 joints) and gripper position (average of 2 fingers)
        arm_qpos = qpos9[..., :7]
        grip_pos = _to_gripper_scalar(qpos9)

        # State is just positions: 7 arm + 1 gripper = 8 dimensions
        state = np.concatenate([arm_qpos, grip_pos], axis=-1)
        
        # Make a writable copy of actions
        out = {
            "image": {
                "primary": img,
                "wrist": wrist,
            },
            "image_mask": {  # Added: indicates which images are real (not padding)
                "primary": np.True_,
                "wrist": np.True_,
            },
            "state": state,
        }
        if "actions" in data:
            actions = np.array(data["actions"], dtype=np.float32, copy=True)
            out["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode('utf-8')
            elif isinstance(prompt, np.ndarray):
                prompt = str(prompt)
            out["prompt"] = prompt
            
        return out


@dataclasses.dataclass(frozen=True)
class LohrbenchOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        if "actions" in data:
            return {"action": data["actions"]}
        return {}