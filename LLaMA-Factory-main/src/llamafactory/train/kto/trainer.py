# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/kto_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import KTOTrainer
from trl.trainer import disable_dropout_in_model

from ...extras.constants import IGNORE_INDEX
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler, get_batch_logps


if TYPE_CHECKING:
    import torch.utils.data
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomKTOTrainer(KTOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.processor = processor
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # kto hyperparams
        self.beta = finetuning_args.pref_beta
        self.desirable_weight = finetuning_args.kto_chosen_weight
        self.undesirable_weight = finetuning_args.kto_rejected_weight
        self.ftx_gamma = finetuning_args.pref_ftx

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.callback_handler.add_callback(BAdamCallback)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        r"""
        Replaces the sequential sampler of KTO Trainer created by trl with the random sampler.
        """
        return Trainer._get_train_sampler(self)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
        super()._save(output_dir, state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.processor is not None:
            getattr(self.processor, "image_processor").save_pretrained(output_dir)

    def forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"], prefix: Literal["", "kl_"] = ""
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        r"""
        Runs forward pass and computes the log probabilities.
        """
        batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error
        model_inputs = {
            "input_ids": batch["{}input_ids".format(prefix)],
            "attention_mask": batch["{}attention_mask".format(prefix)],
        }
        if "pixel_values" in batch:
            model_inputs["pixel_values"] = batch["pixel_values"]

        if "{}token_type_ids".format(prefix) in batch:
            model_inputs["token_type_ids"] = batch["{}token_type_ids".format(prefix)]

        logits = model(**model_inputs, return_dict=True, use_cache=False).logits.to(torch.float32)

        logps, valid_length = get_batch_logps(logits=logits, labels=batch["{}labels".format(prefix)])
        return logps, logps / valid_length

    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        target_logps, target_logps_avg = self.forward(model, batch)
        with torch.no_grad():
            kl_logps, _ = self.forward(model, batch, prefix="kl_")

        if len(target_logps) != len(batch["kto_tags"]):
            raise ValueError("Mismatched shape of inputs and labels.")

        chosen_logps = target_logps[batch["kto_tags"]]
        rejected_logps = target_logps[~batch["kto_tags"]]
        chosen_logps_avg = target_logps_avg[batch["kto_tags"]]
        return chosen_logps, rejected_logps, kl_logps, chosen_logps_avg

    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes log probabilities of the reference model.
        """
        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, reference_kl_logps, _ = self.concatenated_forward(
                ref_model, batch
            )

        return reference_chosen_logps, reference_rejected_logps, reference_kl_logps

    def kto_loss_score(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
        score: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the KTO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (num(chosen) in batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (num(rejected) in batch_size,)
            policy_KL_logps: Log probabilities of the policy model for the KL responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (num(chosen) in batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (num(rejected) in batch_size,)
            reference_KL_logps: Log probabilities of the reference model for the KL responses. Shape: (batch_size,)

        Returns:
            A tuple of four tensors: (losses, chosen_rewards, rejected_rewards, KL).
            The losses tensor contains the KTO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
            The KL tensor contains the detached KL divergence estimate between the policy and reference models.
        """

        kl = (policy_KL_logps - reference_KL_logps).mean().detach()

        # score: tensor([0.1290, 0.2279], device='cuda:0')
        # policy_chosen_logps: tensor([], device='cuda:0', grad_fn=<IndexBackward0>)
        # policy_KL_logps: tensor([-127.0231, -190.2205], device='cuda:0')
        # kl: tensor(0., device='cuda:0')

        kl = self.accelerator.gather(kl).mean().clamp(min=0)

        if policy_chosen_logps.shape[0] != 0 or reference_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - kl))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            chosen_losses = torch.Tensor([]).to(self.accelerator.device)
            chosen_rewards = torch.Tensor([]).to(self.accelerator.device)

        if policy_rejected_logps.shape[0] != 0 or reference_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (kl - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # lists can't be empty -- if they are, then accelerate.gather will hang
            rejected_losses = torch.Tensor([]).to(self.accelerator.device)
            rejected_rewards = torch.Tensor([]).to(self.accelerator.device)

        # score: tensor([0.1290, 0.2279], device='cuda:0')
        # self.desirable_weight: 1.0
        # self.undesirable_weight:1.0
        # rejected_losses:tensor([0.5000, 0.5000], device='cuda:0', grad_fn=<RsubBackward1>)
        # rejected_rewards:tensor([0., 0.], device='cuda:0')
        # chosen_losses: tensor([], device='cuda:0')
        # rejected_losses: tensor([0.5000, 0.5000], device='cuda:0', grad_fn=<RsubBackward1>)

        if score is not None:
           
            if chosen_losses.numel() == 0:
                losses = torch.cat(
                    (chosen_losses , (score)**2 * rejected_losses),
                        0,
                    )
            elif rejected_losses.numel() == 0:
                losses = torch.cat(
                    ( (torch.ones_like(score) - score)**2 * chosen_losses, rejected_losses ),
                        0,
                    )
            else:
                chosen_score = score > 0.5
                chosen_score_tensor = score[chosen_score]
                rejected_score_tensor = score[~chosen_score]
                # chosen_score_tensor: tensor([0.9921], device='cuda:0')
                # rejected_score_tensor: tensor([0.2328], device='cuda:0')
                # score: tensor([0.9921, 0.2328], device='cuda:0')
                losses = torch.cat(
                    ((torch.ones_like(chosen_score_tensor) - chosen_score_tensor)**2 * chosen_losses, 
                       (rejected_score_tensor)**2 * rejected_losses),
                        0,
                    )
        else:
            losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses),
                0,
            )

        return losses, chosen_rewards, rejected_rewards, kl

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        policy_chosen_logps, policy_rejected_logps, policy_kl_logps, policy_chosen_logps_avg = (
            self.concatenated_forward(model, batch)
        )
        reference_chosen_logps, reference_rejected_logps, reference_kl_logps = self.compute_reference_log_probs(
            model, batch
        )
        if "kto_score" in batch:
            score = batch["kto_score"]
        else:
            score = None
        losses, chosen_rewards, rejected_rewards, kl = self.kto_loss_score(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_kl_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_kl_logps,
            score
        )
        losses = losses.nanmean()

        if self.ftx_gamma > 1e-6 and len(policy_chosen_logps) > 0:  # remember to rescale
            sft_loss = -policy_chosen_logps_avg
            losses += self.ftx_gamma * sft_loss.nanmean() / len(policy_chosen_logps) * len(batch["labels"])

        num_chosen = torch.Tensor([len(chosen_rewards)]).to(self.accelerator.device)
        num_rejected = torch.Tensor([len(rejected_rewards)]).to(self.accelerator.device)

        all_num_chosen = self.accelerator.gather(num_chosen).sum().item()
        all_num_rejected = self.accelerator.gather(num_rejected).sum().item()

        if all_num_chosen > 0:
            metrics["rewards/chosen_sum"] = self.accelerator.gather(chosen_rewards.nansum()).nansum().item()
            metrics["logps/chosen_sum"] = self.accelerator.gather(policy_chosen_logps.nansum()).nansum().item()
            metrics["count/chosen"] = all_num_chosen

        if all_num_rejected > 0:
            metrics["rewards/rejected_sum"] = self.accelerator.gather(rejected_rewards.nansum()).nansum().item()
            metrics["logps/rejected_sum"] = self.accelerator.gather(policy_rejected_logps.nansum()).nansum().item()
            metrics["count/rejected"] = all_num_rejected

        metrics["kl"] = kl.item()

        return losses, metrics
