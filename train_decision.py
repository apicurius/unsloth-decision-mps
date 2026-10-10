"""Train an Unsloth decision model (LLM + Clef head, LoRA) on typed-decisions.

Follows the "Train with code" example in Unsloth's guide, with these changes:
the model loads in float32 (no 4-bit), can move to Apple's MPS GPU, and the
typed-decisions test set is scored before and after training.
"""

import argparse
import json
import time

from datasets import load_dataset
from transformers import TrainingArguments

from mps_shim import DecisionTrainer, FastDecisionModel

DATASET = "LocalLLaMA/typed-decisions"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3.5-0.8B")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--max_steps", type=int, default=240)
    parser.add_argument("--epochs", type=float, default=0, help="overrides --max_steps when set")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--out", default="run")
    parser.add_argument("--skip_eval", action="store_true", help="skip both test passes")
    parser.add_argument("--skip_before", action="store_true", help="skip the untrained test pass")
    return parser.parse_args()


def load_model(args):
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
    )
    model = FastDecisionModel.get_peft_model(
        model,
        r=args.r,
        lora_alpha=args.r,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    if args.device == "mps":
        model.to("mps")
    return model, tokenizer


def build_items(model, tokenizer):
    train = load_dataset(DATASET, "all", split="train")
    items, report = FastDecisionModel.build_dataset(train, tokenizer, model)
    print(f"Skipped {report['skipped']} of {report['total']} decisions", flush=True)

    train_items, holdout_items = FastDecisionModel.split_holdout(items, seed=3407)
    print(f"{len(train_items)} training rows, {len(holdout_items)} held out", flush=True)

    test = load_dataset(DATASET, "all", split="test")
    test_items, _ = FastDecisionModel.build_dataset(test, tokenizer, model)
    return train_items, holdout_items, test_items


def evaluate(model, tokenizer, items):
    FastDecisionModel.for_inference(model)
    return FastDecisionModel.evaluate(model, tokenizer, items)


def main():
    args = parse_args()
    start = time.time()
    results = {"args": vars(args)}

    model, tokenizer = load_model(args)
    print(
        "device", next(model.parameters()).device, "load s", round(time.time() - start), flush=True
    )

    train_items, holdout_items, test_items = build_items(model, tokenizer)

    if not args.skip_eval and not args.skip_before:
        eval_start = time.time()
        results["test_before"] = evaluate(model, tokenizer, test_items)
        results["eval_s"] = time.time() - eval_start
        print("BEFORE", results["test_before"], flush=True)
        FastDecisionModel.for_training(model)

    length = {"num_train_epochs": args.epochs} if args.epochs else {"max_steps": args.max_steps}
    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_items,
        eval_dataset=holdout_items,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            bf16=False,
            fp16=False,
            eval_strategy="no",
            logging_steps=1,
            output_dir=args.out,
            report_to="none",
            seed=3407,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=1,
            use_cpu=args.device == "cpu",
            **length,
        ),
    )

    train_start = time.time()
    print("TRAIN", trainer.train(), flush=True)
    results["train_s"] = time.time() - train_start

    # Fit the probability calibration on the held-out rows.
    results["calib"] = FastDecisionModel.calibrate(model, tokenizer, holdout_items)
    print("CALIB", results["calib"], flush=True)

    if not args.skip_eval:
        results["test_after"] = evaluate(model, tokenizer, test_items)
        print("AFTER", results["test_after"], flush=True)

    model.save_pretrained(f"{args.out}/adapter")
    with open(f"{args.out}/result.json", "w") as file:
        json.dump(results, file, indent=2, default=str)
    print("TOTAL s", round(time.time() - start))


if __name__ == "__main__":
    main()
