import torch, evaluate
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk

# ─── paths & constants ────────────────────────────────────────────────
MODEL = "ProsusAI/finbert"        # choosing the finBERT model
DATA  = "data/finbert_tweets"     # folder containing the cleaned data
BATCH = 32                        # batch size = number of sequences for each input into finBERT

# ─── configure model for SINGLE-LABEL classification ─────────────────
config = AutoConfig.from_pretrained(
    MODEL,
    num_labels=3,                               # 0 neg, 1 neu, 2 pos
    problem_type="single_label_classification", # ← forces CrossEntropy
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    config=config,
).to(device)

# ─── load tokenised dataset ───────────────────────────────────────────
dataset = load_from_disk(DATA)

# ─── metrics helpers ─────────────────────────────────────────────────
metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1_micro": metric_f1.compute(predictions=preds, references=labels, average="micro")["f1"],
        "f1_macro": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }

# ─── run evaluation ──────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
    args=TrainingArguments(
        output_dir="runs/baseline",
        per_device_eval_batch_size=BATCH,
        report_to="none"          # no wandb, tensorboard only
    ),
)

print(trainer.evaluate())
