from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from transformers import AutoProcessor

import torch
from torch.utils.data import DataLoader
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "/data/models/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # Directory Paths
    data_root_dir: Path = Path("/data/OpenX-Embodiment")        # Path to Open-X dataset directory
    dataset_name: str = "austin_buds_dataset_converted_externally_to_rlds" # Name of fine-tuning dataset (e.g., `droid_wipe`)
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 4                                            # Fine-tuning batch size
    max_steps: int = 10_000                                        # Max number of fine-tuning steps
    save_steps: int = 2000                                          # Interval for checkpoint saving
    learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Gradient accumulation steps
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
    save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
                                                                    #   continually overwrite the latest checkpoint
                                                                    #   (If False, saves all checkpoints)

    # LoRA Arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = True                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # Tracking Parameters
    wandb_project: str = "openvla-ft-test"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "diffusion-ad"                          # Name of entity to log under
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases


cfg = FinetuneConfig()

device_id = torch.cuda.set_device("cuda:2")
torch.cuda.empty_cache()

# quantization_config = None
# if cfg.use_quantization:
#     assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
#     )

processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
# vla = AutoModelForVision2Seq.from_pretrained(
#     cfg.vla_path,
#     torch_dtype=torch.float16,
#     quantization_config=quantization_config,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
# )


# # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
# if cfg.use_quantization:
#     vla = prepare_model_for_kbit_training(vla)
# else:
#     vla = vla.to(device_id)

# [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
# if cfg.use_lora:
#     lora_config = LoraConfig(
#         r=cfg.lora_rank,
#         lora_alpha=min(cfg.lora_rank, 16),
#         lora_dropout=cfg.lora_dropout,
#         target_modules="all-linear",
#         init_lora_weights="gaussian",
#     )
#     vla = get_peft_model(vla, lora_config)
#     vla.print_trainable_parameters()

# Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
# vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True, static_graph=True)
# print(type(vla), type(vla.module))

# Create Action Tokenizer
action_tokenizer = ActionTokenizer(processor.tokenizer)

batch_transform = RLDSBatchTransform(
    action_tokenizer,
    processor.tokenizer,
    image_transform=processor.image_processor.apply_transform,
    prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
)
vla_dataset = RLDSDataset(
    cfg.data_root_dir,
    cfg.dataset_name,
    batch_transform,
    resize_resolution=(224,224), #tuple(vla.config.image_sizes),
    shuffle_buffer_size=cfg.shuffle_buffer_size,
    image_aug=cfg.image_aug,
)

dataset = vla_dataset.dataset
print(len(vla_dataset))

sample = next(iter(vla_dataset))
# print(sample.keys(), dir(sample))
for key in sample.keys():
    print(key, type(sample[key]), sample[key].shape if "shape" in dir(sample[key]) else None)
print("-"*20)
# save_dataset_statistics(vla_dataset.dataset_statistics, Path("./data_stats/"))

# Create Collator and DataLoader
# collator = PaddedCollatorForActionPrediction(
#     processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
# )
# dataloader = DataLoader(
#     vla_dataset,
#     batch_size=cfg.batch_size,
#     sampler=None,
#     collate_fn=collator,
#     num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
# )

# batch = next(iter(dataloader))

# with torch.autocast("cuda", dtype=torch.float16):
#     output: CausalLMOutputWithPast = vla(
#         input_ids=batch["input_ids"].to(device_id),
#         attention_mask=batch["attention_mask"].to(device_id),
#         pixel_values=batch["pixel_values"].to(torch.float16).to(device_id),
#         labels=batch["labels"],
#     )
