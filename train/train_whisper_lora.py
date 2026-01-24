#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
import os, subprocess
import shutil
import wandb

import torch
from datasets import load_dataset, Features, Value, load_from_disk, DatasetDict

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
        input_features = torch.stack([torch.as_tensor(f["input_features"]) for f in features])
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


def atomic_save_datasetdict(ds_dict: DatasetDict, out_dir: str):
    """Atomic save_to_disk: write to .tmp then rename."""
    out_dir = str(Path(out_dir).resolve())
    tmp_dir = out_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    Path(tmp_dir).parent.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(tmp_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.rename(tmp_dir, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="openai/whisper-medium")
    ap.add_argument("--train_jsonl", default="work/splits_clean/train.jsonl")
    ap.add_argument("--val_jsonl", default="work/splits_clean/val.jsonl")
    ap.add_argument("--out_dir", default="outputs/ft_whisper_medium_lora_tr")
    ap.add_argument("--language", default="turkish")
    ap.add_argument("--task", default="transcribe")

    # training knobs
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="If >0, stop training after this many optimizer steps (after grad_accum).")
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--fp16", action="store_true", help="Enable fp16")
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="asr-live")

    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_rows", type=int, default=5)

    # processed dataset flags
    ap.add_argument("--preprocess_only", action="store_true")
    ap.add_argument("--processed_dir", default="")
    ap.add_argument("--load_processed", action="store_true")

    args = ap.parse_args()

    import json

    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)

    # SMOKE MODE (unchanged)
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
        val_rows = sample(val_path, args.smoke_rows)

        def check_audio(rows):
            missing = 0
            for r in rows:
                apth = Path(r["audio"])
                if not apth.exists():
                    missing += 1
            return missing

        print("SMOKE OK")
        print("train_jsonl:", train_path, "sampled:", len(train_rows), "missing_audio:", check_audio(train_rows))
        print("val_jsonl:  ", val_path,   "sampled:", len(val_rows),   "missing_audio:", check_audio(val_rows))
        print("train_sample_texts:", [r["text"][:80] for r in train_rows])
        print("val_sample_texts:  ", [r["text"][:80] for r in val_rows])
        return

    # Basic asserts
    # Only require JSONLs when we will actually load them
    if not args.load_processed:
        assert train_path.exists(), f"Missing {train_path}"
        assert val_path.exists(), f"Missing {val_path}"
    

    # CHANGE: guardrails for processed flags
    if (args.preprocess_only or args.load_processed) and not args.processed_dir:
        raise ValueError("--processed_dir is required when using --preprocess_only or --load_processed")

    # CHANGE: init wandb ONLY once, and NEVER for preprocess_only
    use_wandb = bool(args.wandb) and (not args.preprocess_only)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            job_type="train",
            config={**vars(args), "git_sha": git_sha()},
        )

    # Processor needed for preprocessing
    processor = WhisperProcessor.from_pretrained(
        args.model_name, language=args.language, task=args.task
    )

    # CHANGE: load processed dataset if requested, else preprocess and optionally save
    if args.load_processed:
        ds = load_from_disk(str(Path(args.processed_dir).resolve()))
        if not isinstance(ds, DatasetDict):
            raise ValueError(f"Expected DatasetDict at {args.processed_dir}, got {type(ds)}")
        if "train" not in ds or "validation" not in ds:
            raise ValueError(f"Processed dataset must contain 'train' and 'validation' splits: {args.processed_dir}")
    else:
        features = Features({"audio": Value("string"), "text": Value("string")})
        ds = load_dataset(
            "json",
            data_files={"train": args.train_jsonl, "validation": args.val_jsonl},
            features=features,
        )

        def prepare(batch):
            audio_path = batch["audio"]

            # Load WAV from path without datasets Audio decoding
            try:
                import soundfile as sf
                wav, sr = sf.read(audio_path, dtype="float32")
            except Exception:
                import torchaudio
                wav, sr = torchaudio.load(audio_path)
                wav = wav.squeeze(0).cpu().numpy()
                sr = int(sr)

            # ensure mono + float32
            import numpy as np
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim > 1:
                if wav.shape[0] < wav.shape[-1]:
                    wav = wav.mean(axis=0)
                else:
                    wav = wav.mean(axis=1)

            inputs = processor(wav, sampling_rate=int(sr), return_tensors="pt")
            batch["input_features"] = inputs.input_features[0]
            batch["labels"] = processor.tokenizer(batch["text"]).input_ids
            return batch

        ds = ds.map(
            prepare,
            remove_columns=ds["train"].column_names,
            num_proc=1,  # robust: avoid pickle/multiproc edge cases during preprocessing
        )

        # CHANGE: persist processed dataset if requested
        if args.processed_dir:
            atomic_save_datasetdict(ds, args.processed_dir)
            print(f"[OK] saved processed dataset to: {Path(args.processed_dir).resolve()}")

        # CHANGE: preprocess-only exit BEFORE model download/training
        if args.preprocess_only:
            print("[OK] preprocess_only done; exiting before training.")
            return

    # CHANGE: model init AFTER preprocess/load (so preprocess-only doesn’t download model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    # lock TR transcription prompts
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task=args.task
    )

    # LoRA
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora)

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
        max_steps=(args.max_steps if args.max_steps > 0 else -1),
        eval_strategy="steps",
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
        # dataset manifests (optional; never crash training completion)
        try:
            train_f = Path(args.train_jsonl)
            val_f   = Path(args.val_jsonl)

            if train_f.is_file() and val_f.is_file():
                data_art = wandb.Artifact("dataset-manifests", type="dataset")
                data_art.add_file(str(train_f))
                data_art.add_file(str(val_f))
                wandb.log_artifact(data_art)
            else:
                print(f"[WARN] Skipping manifest artifact upload; missing files: "
                    f"train={args.train_jsonl} val={args.val_jsonl}")
        except Exception as e:
            print(f"[WARN] Manifest artifact upload failed: {e}")

        # model artifact (this is the important one)
        try:
            model_art = wandb.Artifact("ft-model", type="model")
            model_art.add_dir(args.out_dir)
            wandb.log_artifact(model_art)
        except Exception as e:
            print(f"[WARN] Model artifact upload failed: {e}")

        wandb.finish()

    print("Saved:", args.out_dir)


if __name__ == "__main__":
    main()
