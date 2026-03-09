# autoresearch — financial sentiment task

This is the financial sentiment experiment: train a GPT-based binary classifier
to predict whether financial text is bullish (stock up) or bearish (stock down).
The agent iterates autonomously to maximize validation accuracy.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `sentiment-mar9`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare_sentiment.py` — fixed: data download, tokenizer, dataloader, evaluation. Do not modify.
   - `train_sentiment.py` — the file you modify. GPT model + classification head + optimizer + training loop.
4. **Verify data exists**: Check that both of the following are present:
   - `~/.cache/autoresearch/tokenizer/tokenizer.pkl` — if missing, run `uv run prepare.py --num-shards 1`
   - `~/.cache/autoresearch/sentiment/financial_phrasebank.parquet` — if missing, run `uv run prepare_sentiment.py`
5. **Initialize results.tsv**: Create `results.tsv` with the header row shown below. Record `baseline` as the first entry after the first run.
6. **Confirm and go**: Confirm setup looks good, then begin the experiment loop.

## Experimentation

Each experiment runs on a single GPU/MPS device. The training script runs for a **fixed time budget of 5 minutes** (wall-clock training time, excluding startup). Launch it as: `uv run train_sentiment.py`.

**What you CAN do:**
- Modify `train_sentiment.py` — the only file you edit. Everything is fair game: model architecture, classification head, pooling strategy, optimizer, hyperparameters, `LM_LOSS_WEIGHT`, batch size, depth, etc.

**What you CANNOT do:**
- Modify `prepare_sentiment.py`. It is read-only. It contains the fixed evaluation (`evaluate_accuracy`), data loading, tokenizer, and time budget constants.
- Install new packages or add dependencies. Use only what's in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_accuracy` function in `prepare_sentiment.py` is the ground truth metric.

**The goal is simple: maximize val_accuracy.**

The time budget is fixed at 5 minutes, so all experiments are directly comparable. Explore freely:
- Depth, width (ASPECT_RATIO), HEAD_DIM
- Classification head: deeper head, different pooling (mean vs. last token vs. max), attention pooling
- LM_LOSS_WEIGHT: does adding language model auxiliary loss help regularise on this small dataset?
- CLS_DROPOUT: is 0.1 too much or too little?
- Optimizer: learning rates, warmup/warmdown ratios, Adam betas
- DEVICE_BATCH_SIZE: larger batches may stabilise the BCE loss

**VRAM** is a soft constraint. Some increase is acceptable for meaningful accuracy gains.

**Simplicity criterion**: Prefer simpler changes. A 0.005 accuracy gain that removes code beats a 0.005 gain that adds 20 lines. Always weigh improvement magnitude against complexity cost.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_accuracy:     0.789000
val_auc:          0.851234
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     2400.0
num_steps:        1842
num_params_M:     2.1
depth:            4
```

Note: `peak_vram_mb` is 0.0 on MPS (Apple Silicon) — this is expected. Extract the key metrics:

```
grep "^val_accuracy:\|^val_auc:\|^peak_vram_mb:" run.log
```

## Logging results

Log each experiment to `results.tsv` (tab-separated, NOT comma-separated).

Header and columns:

```
commit	val_accuracy	val_auc	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_accuracy achieved (e.g. 0.789000) — use 0.000000 for crashes
3. val_auc achieved (e.g. 0.851234) — use 0.000000 for crashes
4. peak memory in GB (peak_vram_mb / 1024, 1 decimal) — use 0.0 for crashes or MPS
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_accuracy	val_auc	memory_gb	status	description
a1b2c3d	0.789000	0.851234	0.0	keep	baseline depth=4 pure classifier
b2c3d4e	0.801000	0.863000	0.0	keep	LM_LOSS_WEIGHT=0.1 adds regularisation
c3d4e5f	0.775000	0.840000	0.0	discard	depth=8 overfits on small dataset
d4e5f6g	0.000000	0.000000	0.0	crash	batch size 256 OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/sentiment-mar9`).

LOOP FOREVER:

1. Check git state: the current branch/commit we're on.
2. Hack `train_sentiment.py` with an experimental idea.
3. git commit.
4. Run the experiment: `uv run train_sentiment.py > run.log 2>&1`
5. Read the results: `grep "^val_accuracy:\|^val_auc:\|^peak_vram_mb:" run.log`
6. If grep output is empty → crashed. Run `tail -n 50 run.log` to read the stack trace. Attempt a fix. If you can't fix it in a couple tries, skip it.
7. Record the results in `results.tsv`.
8. If val_accuracy improved (higher is better) → keep the commit, advance the branch.
9. If val_accuracy is equal or worse → `git reset` back to previous state.

**Timeout**: Kill runs exceeding 10 minutes, treat as failure.

**Crashes**: Fix trivial bugs (typos, missing imports). Skip fundamentally broken ideas.

**NEVER STOP**: Run indefinitely until manually interrupted. If you run out of ideas, consider:
- Combining near-miss experiments
- Trying different pooling strategies (mean → max → last → learned CLS token)
- Adding/removing dropout, layer norm positions
- Testing different LM_LOSS_WEIGHT values more systematically
- Exploring the depth/width tradeoff more finely
- More radical changes: bidirectional attention for the final layer, multi-task classification head

## Scoring live transcripts

After any successful run, the model is saved to `~/.cache/autoresearch/sentiment/model.pt`.
To score a live earnings call transcript:

```bash
echo "We delivered record revenue of $3.2B, up 28% year-over-year..." | uv run score_transcript.py
# or
uv run score_transcript.py "Management reaffirmed full-year guidance..."
# or
uv run score_transcript.py transcript.txt
```

The script handles arbitrarily long transcripts via sliding-window chunking.

## Telegram integration (optional)

### Overnight progress notifications

If `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set as environment variables,
call `notify_telegram.py` after logging each experiment result so you wake up to
a full summary on your phone instead of having to read a terminal log.

Add this step after recording in `results.tsv` (step 7 in the experiment loop):

```bash
uv run notify_telegram.py "keep | acc=0.801 | auc=0.863 | LM_LOSS_WEIGHT=0.1 adds regularisation"
# or for a discard:
uv run notify_telegram.py "discard | acc=0.775 | depth=8 overfits on small dataset"
```

Use `notify_telegram.format_experiment_result()` for consistent formatting when
calling from Python.

### Scoring bot

Run `telegram_bot.py` on any machine with the trained model. Then send any text
directly in Telegram and get a BULLISH/BEARISH score back instantly:

```bash
uv run telegram_bot.py    # runs until Ctrl-C
```

See `notify_telegram.py` for setup instructions (BotFather → token → chat ID).
