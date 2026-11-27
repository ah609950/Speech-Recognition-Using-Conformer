"""Report figures from TensorBoard logs and evaluation results"""
from __future__ import annotations

from pathlib import Path

def load_scalars(log_dir):
    """Return {tag: (steps, values)} for every scalar series in a run"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0}) # 0 = keep all points
    acc.Reload() # actually read the files
    data = {}
    for tag in acc.Tags()["scalars"]:
        events = acc.Scalars(tag)
        data[tag] = ([e.step for e in events], [e.value for e in events])

    return data

def ema(values, weight):
    """Each point = weight * previous smoothed point + (1 - weight) * new raw point"""
    smoothed, last = [], values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)

    return smoothed

def _save(fig, path):
    """Save as both PDF and PNG"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=200, bbox_inches="tight")

def plot_loss_curves(log_dir, output_path, smoothing=0.9, selected_step=None,
                     train_tag="train/loss", val_tag="val/loss"):
    import matplotlib.pyplot as plt

    data = load_scalars(log_dir)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if train_tag in data:
        steps, values = data[train_tag]
        ax.plot(steps, values, color="C0", alpha=0.2, linewidth=1) # raw, faint
        ax.plot(steps, ema(values, smoothing), color="C0", label=f"training (EMA {smoothing})")
    if val_tag in data:
        steps, values = data[val_tag]
        ax.plot(steps, values, "o-", color="C1", markersize=3, label="validation (dev-clean)")
    if selected_step is not None:
        ax.axvline(selected_step, linestyle="--", color="gray", label="selected checkpoint")
    ax.set(xlabel="Optimizer step", ylabel="RNN-T loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, output_path)
    plt.close(fig)

def plot_scalar(log_dir, tag, output_path, ylabel, selected_step=None):
    """Any single curve"""
    import matplotlib.pyplot as plt

    data = load_scalars(log_dir)
    if tag not in data:
        print(f"'{tag}' not in log. Available: {sorted(data)}")
        return
    steps, values = data[tag]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, values, color="C0")
    if selected_step is not None:
        ax.axvline(selected_step, linestyle="--", color="gray", label="selected checkpoint")
        ax.legend(frameon=False)
    ax.set(xlabel="Optimizer step", ylabel=ylabel)
    ax.grid(alpha=0.25)
    _save(fig, output_path)
    plt.close(fig)

def plot_error_breakdown(metrics_list, output_path):
    """Stacked S/D/I bars per split
    metrics_list: dicts saved by evaluate()"""
    import matplotlib.pyplot as plt
    splits = [m["split"] for m in metrics_list]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bottom = [0.0] * len(metrics_list)
    for i, kind in enumerate(("substitutions", "deletions", "insertions")):
        heights = [100 * m["word_error_counts"][kind] / m["reference_words"] for m in metrics_list]
        ax.bar(splits, heights, bottom=bottom, color=f"C{i}", label=kind)
        bottom = [b + h for b, h in zip(bottom, heights)] # stack the next one on top
    for x, total in enumerate(bottom):
        ax.text(x, total, f"{total:.1f}%", ha="center", va="bottom") # total WER on each bar
    ax.set(ylabel="Word errors (% of reference words)")
    ax.legend(frameon=False)
    _save(fig, output_path)
    plt.close(fig)