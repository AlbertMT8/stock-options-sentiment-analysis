# finetuning.py ────────────────────────────────────────────────────────
import torch, evaluate
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_from_disk

# ─── paths & hyper-params ─────────────────────────────────────────────
MODEL_NAME = "ProsusAI/finbert"
DATA       = "data/finbert_tweets"
OUT        = "model/finbert_finetuned"
EPOCHS     = 3
BATCH      = 16
LR         = 2e-5

# ─── load tokenised dataset ───────────────────────────────────────────
dataset = load_from_disk(DATA)

# ─── configure model for SINGLE-LABEL classification ─────────────────
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}

config = AutoConfig.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    problem_type="single_label_classification",
    label2id=label2id,
    id2label=id2label,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config,
).to(device)

# ─── metrics (same as baseline) ───────────────────────────────────────
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

# ─── training arguments ──────────────────────────────────────────────
args = TrainingArguments(
    output_dir        = OUT,
    eval_strategy = "epoch",
    save_strategy       = "epoch",
    save_total_limit    = 2,
    learning_rate       = LR,
    per_device_train_batch_size = BATCH,
    per_device_eval_batch_size  = BATCH,
    num_train_epochs    = EPOCHS,
    weight_decay        = 0.01,
    load_best_model_at_end = True,
    metric_for_best_model  = "eval_loss",
    logging_steps       = 50,
    fp16                = torch.cuda.is_available(),  # ignored on M-series
)

# ─── trainer ──────────────────────────────────────────────────────────
trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = dataset["train"],
    eval_dataset    = dataset["validation"],
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()
trainer.save_model(OUT)
print(f"\n✅  Fine-tuned model saved to {OUT}")
