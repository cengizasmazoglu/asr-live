#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path

import os, subprocess
import wandb

import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model


@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: WhisperProcessor

    def __call__(self, features):
        input_features = torch.stack([f["input_features"] for f in features])
        labels = [f["labels"] for f in features]
        labels = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"input_features": input_features, "labels": labels}

def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

class WandbLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs, step=state.global_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="openai/whisper-medium")
    ap.add_argument("--train_jsonl", default="work/splits_clean/train.jsonl")
    ap.add_argument("--val_jsonl", default="work/splits_clean/val.jsonl")
    ap.add_argument("--out_dir", default="outputs/ft_whisper_medium_lora_tr")
    ap.add_argument("--language", default="turkish")
    ap.add_argument("--task", default="transcribe")

    # training knobs (safe defaults)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", default="asr-live")

    ap.add_argument("--smoke", action="store_true", help="Cheap local smoke test: validate files + sample rows only")
    ap.add_argument("--smoke_rows", type=int, default=5, help="How many rows to sample in --smoke mode")


    args = ap.parse_args()

    from pathlib import Path
    import json

    train_path = Path(args.train_jsonl)
    val_path   = Path(args.val_jsonl)

    if args.smoke:
        assert train_path.exists(), f"Missing {train_path}"
        assert val_path.exists(), f"Missing {val_path}"

        def sample(path, n):
            rows = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
                    if len(rows) >= n:
                        break
            return rows

        train_rows = sample(train_path, args.smoke_rows)
        val_rows   = sample(val_path, args.smoke_rows)

        def check_audio(rows):
            missing = 0
            for r in rows:
                ap = Path(r["audio"])
                if not ap.exists():
                    missing += 1
            return missing

        print("SMOKE OK")
        print("train_jsonl:", train_path, "sampled:", len(train_rows), "missing_audio:", check_audio(train_rows))
        print("val_jsonl:  ", val_path,   "sampled:", len(val_rows),   "missing_audio:", check_audio(val_rows))
        print("train_sample_texts:", [r["text"][:80] for r in train_rows])
        print("val_sample_texts:  ", [r["text"][:80] for r in val_rows])
        return


    use_wandb = bool(args.wandb)

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            job_type="train",
            config={**vars(args), "git_sha": git_sha()},
        )

    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    assert train_path.exists(), f"Missing {train_path}"
    assert val_path.exists(), f"Missing {val_path}"

    processor = WhisperProcessor.from_pretrained(
        args.model_name, language=args.language, task=args.task
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    # lock TR transcription prompts
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )

    # LoRA: common targets for Whisper attention projections
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora)

    ds = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    def prepare(batch):
        audio = batch["audio"]
        inputs = processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        )
        batch["input_features"] = inputs.input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    ds = ds.map(
        prepare,
        remove_columns=ds["train"].column_names,
        num_proc=max(1, args.num_workers),
    )

    data_collator = DataCollatorSpeechSeq2Seq(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=["wandb"] if use_wandb else "none",
        run_name=wandb.run.name if use_wandb else None,
        predict_with_generate=False,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        callbacks=[WandbLogCallback()] if use_wandb else None,
    )

    trainer.train()
    trainer.save_model(args.out_dir)

    if use_wandb:
        # log dataset manifests as artifacts
        data_art = wandb.Artifact("dataset-manifests", type="dataset")
        data_art.add_file(args.train_jsonl)
        data_art.add_file(args.val_jsonl)
        wandb.log_artifact(data_art)

        # log model output directory as artifact
        model_art = wandb.Artifact("ft-model", type="model")
        model_art.add_dir(args.out_dir)
        wandb.log_artifact(model_art)

        wandb.finish()

    print("Saved:", args.out_dir)


if __name__ == "__main__":
    main()
