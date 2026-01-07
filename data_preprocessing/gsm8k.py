# %%
import json
from pathlib import Path

# 使用绝对路径，避免相对路径问题
script_dir = Path(__file__).parent
p = script_dir / "../datasets/gsm8k"
p = script_dir / "/root/autodl-tmp/colar/datasets/gsm8k"
p = p.resolve()

# %%
def load_jsonl(filepath):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 加载本地 JSONL 文件
train_full = load_jsonl(p / "train_socratic.jsonl")
test_ds = load_jsonl(p / "test_socratic.jsonl")

# 按 90%/10% 划分 train/val
split_idx = int(len(train_full) * 0.9)
train_ds = train_full[:split_idx]
val_ds = train_full[split_idx:]

# %%
for split, ds in zip(["train", "val", "test"], [train_ds, val_ds, test_ds]):
    ds_json = []
    for d in ds:
        q = d["question"]
        a = d["answer"]
        [steps, answer] = a.split("\n####")
        steps = steps.split("\n")
        answer = answer.strip()
        ds_json.append({"question": q, "steps": steps, "answer": answer})
    with open(p / f"{split}.json", "w") as f:
        json.dump(ds_json, f)

# %%
