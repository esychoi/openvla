import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset
from transformers import AutoProcessor

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


VLA_PATH = "/data/models/openvla-7b"
JSON_DATA_FOLDER = Path("/data/sample_0924/preprocessed")
IMAGE_DATA_FOLDER = Path("/data/sample_0924/1.원천데이터/01.이미지데이터/01.실제환경")

class CustomRLDSDataset(IterableDataset):
    def __init__(self, data_path: Path, batch_transform: RLDSBatchTransform, dataset_name: str = "dg_univ_data") -> None:
        self.batch_transform = batch_transform
        self.dataset_name = dataset_name
        self.dataset, self.dataset_statistics = self.make_dataset(data_path)

    def make_dataset(self, data_path: Path) -> list[Dict[str, Any]]:
        def get_observation(file_name: str, frame_info: Dict):
            task_id = "_".join(file_name.split("_")[:4])
            timestep = frame_info[file_name]["sensor"]["time_step"]
            image = np.array(Image.open(IMAGE_DATA_FOLDER / task_id / (file_name + ".png")))
            observation = {
                "image_primary": np.expand_dims(image, axis=0),
                "timestep": timestep,
                "pad_mask_dict": {
                    "image_primary": np.array([True]),
                    "timestep": np.array([True])
                },
                "pad_mask": np.array([True]) 
            }
            return observation
        
        def get_task_and_action(next_frame_name: str):
            if next_frame_name is None:
                return None, None
            
            next_frame_path = JSON_DATA_FOLDER / (next_frame_name + ".json")
            with open(next_frame_path, "r") as f:
                frame_info = json.load(f)

            file_name = next_frame_path.stem #list(frame_info.keys())[0]
            task_id = "_".join(file_name.split("_")[:4])
            image = np.array(Image.open(IMAGE_DATA_FOLDER / task_id / (file_name + ".png")))
            sensors = frame_info[file_name]['sensor']
            action = [sensor_value for sensor_name, sensor_value in sensors.items() if sensor_name.startswith("gripper") or sensor_name.startswith("joint") or sensor_name.startswith("end-effector") or sensor_name.startswith("force-data") or sensor_name.startswith("torque-data")]
            
            task = {
                "language_instruction": frame_info[file_name]["video_description"].encode('utf-8'),
                "pad_mask_dict": {
                    "language_instruction": True,
                    "image_primary": True,
                    "timestep": True
                },
                "image_primary": image,
                "timestep": sensors['time_step']
            }
            return task, action

        dataset = []
        action_values = []

        for frame in data_path.iterdir():
            with open(frame, "r") as f:
                frame_info = json.load(f)
            file_name = frame.stem #list(frame_info.keys())[0] # frame.stem
            
            observation = get_observation(file_name, frame_info)
            task, action = get_task_and_action(frame_info[file_name]["next_frame"])
            if task is None: # last step of the current trajectory -> repeat observation
                task = {
                    "language_instruction": frame_info[file_name]["video_description"].encode('utf-8'),
                    "pad_mask_dict": {
                        "language_instruction": True,
                        "image_primary": True,
                        "timestep": True
                    },
                    "image_primary": observation['image_primary'],
                    "timestep": observation['timestep']
                }
                sensors = frame_info[file_name]['sensor']
                action = [sensor_value for sensor_name, sensor_value in sensors.items() if sensor_name.startswith("gripper") or sensor_name.startswith("joint") or sensor_name.startswith("end-effector") or sensor_name.startswith("force-data") or sensor_name.startswith("torque-data")]


            # task_id = "_".join(file_name.split("_")[:4])
            # timestep = frame_info[file_name]["sensor"]["time_step"]
            # image = np.array(Image.open(IMAGE_DATA_FOLDER / task_id / (file_name + ".png")))
            # if frame_info[file_name]["next_frame"]:
            #     action = self.get_action(JSON_DATA_FOLDER / (frame_info[file_name]["next_frame"] + ".json"))
            # else:
            #     action = None

            # Get frame data
            frame_data = {
                "dataset_name": self.dataset_name,
                "observation": observation, #{
                #     "image_primary": np.expand_dims(image, axis=0),
                #     "timestep": timestep,
                #     "pad_mask_dict": {
                #         "image_primary": True,
                #         "timestep": True
                #     },
                #     "pad_mask": True
                # },
                "task": task, #{
                #     "language_instruction": frame_info[file_name]["video_description"],
                #     "pad_mask_dict": {
                #         "language_instruction": True,
                #         "image_primary": True,
                #         "timestep": True
                #     },
                #     "image_primary": image,
                #     "timestep": timestep
                # },
                "action": np.expand_dims(np.array(action), axis=0)
            }

            dataset.append(frame_data)
            if action is not None:
                action_values.append(action)

        # Compute statistcs
        action_values = np.array(action_values)
        statistics = {
            self.dataset_name: {
                "action": {
                    "mean": action_values.mean(axis=0),
                    "std": action_values.std(axis=0),
                    "min": action_values.min(axis=0),
                    "max": action_values.max(axis=0),
                    "q01": np.quantile(action_values, q=0.01, axis=0),
                    "q99": np.quantile(action_values, q=0.99, axis=0),
                    "mask": np.array([True]*19)
                },
                "proprio": {
                    "mean": np.zeros(19),
                    "std": np.zeros(19),
                    "min": np.zeros(19),
                    "max": np.zeros(19),
                    "q01": np.zeros(19),
                    "q99": np.zeros(19),
                },
                "num_transitions": action_values.shape[0],
                "num_trajectories": 1
            }
        }

        return dataset, statistics

    def get_action(self, frame_path: Path):
        """Assumes `frame_path` is a valid frame path."""
        with open(frame_path, "r") as f:
            frame_info = json.load(f)
        file_name = list(frame_info.keys())[0]
        sensors = frame_info[file_name]['sensor']
        action = [sensor_value for sensor_name, sensor_value in sensors.items() if sensor_name not in ["time_step", "tactile_sensor_fx1", "tactile_sensor_fy1", "tactile_sensor_fz1", "tactile_sensor_fx2", "tactile_sensor_fy2", "tactile_sensor_fz2"]]
        return action
                

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset:
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return len(self.dataset)

if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained(VLA_PATH, trust_remote_code=True)

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in VLA_PATH else VicunaV15ChatPromptBuilder,
    )

    sample_dataset = CustomRLDSDataset(JSON_DATA_FOLDER, batch_transform)
    print(len(sample_dataset))

    save_dataset_statistics(sample_dataset.dataset_statistics, Path("statistics"))
    
    sample = next(iter(sample_dataset))
    
    for key in sample.keys():
        print(key, type(sample[key]), sample[key].shape if "shape" in dir(sample[key]) else None)

    print("-"*20)