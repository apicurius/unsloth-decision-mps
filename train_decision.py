
import os, sys, time, json, argparse
p = argparse.ArgumentParser()
p.add_argument("--model", default="unsloth/Qwen3.5-0.8B")
p.add_argument("--device", default="cpu")          # cpu | mps
p.add_argument("--max_steps", type=int, default=240)
p.add_argument("--epochs", type=float, default=0)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--r", type=int, default=16)
p.add_argument("--max_seq_length", type=int, default=2048)
p.add_argument("--out", default="run")
p.add_argument("--skip_eval", action="store_true")
p.add_argument("--skip_before", action="store_true")
a = p.parse_args()
import torch
from mps_shim import FastDecisionModel, DecisionTrainer
from datasets import load_dataset
from transformers import TrainingArguments
t0 = time.time()
model, tokenizer = FastDecisionModel.from_pretrained(model_name=a.model, max_seq_length=a.max_seq_length, load_in_4bit=False)
model = FastDecisionModel.get_peft_model(model, r=a.r, lora_alpha=a.r, lora_dropout=0,
    use_gradient_checkpointing="unsloth", random_state=3407)
if a.device == "mps":
    model.to("mps")
print("device", next(model.parameters()).device, "load s", round(time.time()-t0), flush=True)
ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
items, report = FastDecisionModel.build_dataset(ds, tokenizer, model)
print(f"Skipped {report['skipped']} of {report['total']} decisions", flush=True)
train_items, eval_items = FastDecisionModel.split_holdout(items, seed=3407)
print(len(train_items), "train", len(eval_items), "held out", flush=True)
test = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
test_items, _ = FastDecisionModel.build_dataset(test, tokenizer, model)
res = {"args": vars(a)}
if not a.skip_eval and not a.skip_before:
    FastDecisionModel.for_inference(model)
    t=time.time(); res["test_before"] = FastDecisionModel.evaluate(model, tokenizer, test_items); res["eval_s"]=time.time()-t
    print("BEFORE", res["test_before"], flush=True)
    FastDecisionModel.for_training(model) if hasattr(FastDecisionModel, "for_training") else model.train()
kw = dict(num_train_epochs=a.epochs) if a.epochs else dict(max_steps=a.max_steps)
trainer = DecisionTrainer(model=model, processing_class=tokenizer, train_dataset=train_items, eval_dataset=eval_items,
    args=TrainingArguments(per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_steps=10,
        learning_rate=a.lr, lr_scheduler_type="cosine", weight_decay=0.01, bf16=False, fp16=False,
        eval_strategy="no", logging_steps=1, output_dir=a.out, report_to="none", seed=3407,
        save_strategy="steps", save_steps=50, save_total_limit=1, use_cpu=(a.device=="cpu"), **kw))
t=time.time(); st = trainer.train(); res["train_s"]=time.time()-t
print("TRAIN", st, flush=True)
res["calib"] = FastDecisionModel.calibrate(model, tokenizer, eval_items); print("CALIB", res["calib"], flush=True)
if not a.skip_eval:
    FastDecisionModel.for_inference(model)
    res["test_after"] = FastDecisionModel.evaluate(model, tokenizer, test_items); print("AFTER", res["test_after"], flush=True)
model.save_pretrained(a.out + "/adapter")
json.dump(res, open(a.out + "/result.json", "w"), indent=2, default=str)
print("TOTAL s", round(time.time()-t0))
