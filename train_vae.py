import os
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from composer.utils import dist
from composer.optim.scheduler import ConstantWithWarmupScheduler
from composer import Logger, Trainer, ComposerModel
from composer.loggers import WandBLogger
from composer.core import Callback, State, Time, TimeUnit
from composer.metrics import CrossEntropy
from composer.algorithms import GradientClipping
from composer.callbacks import SpeedMonitor
from composer.core import Precision
import torchelie.nn as tnn
from gdr import get_dataloader
from model import Transformer
from dotenv import load_dotenv
from transformers import AutoTokenizer
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy

UBATCH = 1


def make_vae(
    vocab_size, hidden_dim, num_heads, head_size, num_layers, IR_dims, vae_strength
):
    enc = Transformer(
        vocab_size, hidden_dim, num_heads, head_size, num_layers, causal=False
    )
    enc.proj = tnn.InformationBottleneck(hidden_dim, IR_dims, vae_strength)

    dec = Transformer(
        vocab_size, hidden_dim, num_heads, head_size, num_layers, causal=False
    )
    dec.vocab = nn.Linear(IR_dims, hidden_dim)
    return enc, dec


class IRRange(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("min", torch.tensor(float("inf")), dist_reduce_fx="min")
        self.add_state("max", torch.tensor(float("-inf")), dist_reduce_fx="max")

    def update(self, ir):
        self.min = torch.min(self.min, ir.min())
        self.max = torch.max(self.max, ir.max())

    def compute(self):
        return self.max - self.min


class ComposerVAE(ComposerModel):
    def __init__(self, encoder, decoder):
        super().__init__()
        vocab_size = encoder.vocab.weight.shape[0]
        self.encoder = encoder
        self.decoder = decoder
        self.train_metrics = {
            "acc": MulticlassAccuracy(num_classes=vocab_size, average="micro"),
            "ce": CrossEntropy(),
            "IR_range": IRRange(),
        }

        self.val_metrics = {
            "acc": MulticlassAccuracy(num_classes=vocab_size, average="micro")
        }

    def forward(self, batch):
        inputs = batch["input_ids"]
        ir = self.encoder(inputs)
        return self.decoder(ir), ir

    def loss(self, outputs, batch):
        inputs = batch["input_ids"]
        logits, _ = outputs
        loss = F.cross_entropy(logits.transpose(1, 2), inputs.to(logits.device))
        return loss

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        out, ir = outputs
        # I hate this shit. Have to recover what metric is it in order to know
        # how to update it
        if isinstance(metric, IRRange):
            metric.update(ir)
        else:
            metric.update(out.transpose(1, 2), batch["input_ids"])

    def get_metrics(self, is_train: bool) -> Dict[str, Metric]:
        if is_train:
            return self.train_metrics
        else:
            return self.val_metrics


class Untokenize(Callback):
    def __init__(self, interval):
        self.interval = Time.from_input(interval, TimeUnit.BATCH)
        if self.interval.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(
                f"Invalid time unit for parameter interval: " f"{self.interval.unit}"
            )

        self.tok = AutoTokenizer.from_pretrained(
            "lightonai/mamba-tokenizer-v1__en-fr-code", token=os.environ["HF_TOKEN"]
        )

        self.last_train_time_value_logged = -1

    def after_forward(self, state: State, logger: Logger) -> None:
        current_time_value = state.timestamp.get(self.interval.unit).value
        if (
            current_time_value % self.interval.value == 0
            and current_time_value != self.last_train_time_value_logged
        ):
            self.last_train_time_value_logged = current_time_value
            self._log(state.outputs[0])

    def _log(self, s):
        untokenized = self.tok.batch_decode(s.argmax(dim=-1))
        import wandb

        html = "<hr>".join(untokenized)
        wandb.log({"reconstruction": wandb.Html(html)})


def train():
    trainloader = get_dataloader(
        "/mnt/data/redpajama-v1__wikipedia__fr.train.gig.npy",
        batch_size=64,
        seq_len=512,
        n_data_parallel=dist.get_world_size(),
        rank=dist.get_global_rank(),
    )
    model = ComposerVAE(
        *make_vae(
            vocab_size=65024,
            hidden_dim=512,
            num_heads=8,
            head_size=64,
            num_layers=6,
            IR_dims=32,
            vae_strength=0.0001,
        )
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.95), foreach=True
    )
    lr_scheduler = ConstantWithWarmupScheduler("0.05dur")

    gc = GradientClipping(clipping_type="norm", clipping_threshold=1.0)
    speed_monitor = SpeedMonitor(window_size=100)

    wandb_logger = WandBLogger(
        project="diffusion",
        name="vae",
    )
    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        algorithms=[gc],
        train_dataloader=trainloader,
        max_duration="376058700tok",  #'1ep',
        loggers=wandb_logger,
        device="gpu",
        callbacks=[speed_monitor, Untokenize("10ba")],
        # fsdp_config={"activation_checkpointing": False},
        precision=Precision.AMP_BF16,
        # device_train_microbatch_size=UBATCH,
    )
    trainer.fit()


if __name__ == "__main__":
    load_dotenv()
    train()
