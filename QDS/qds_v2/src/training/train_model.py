"""Ranking-based model training on trajectory windows. See src/training/README.md for details."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.experiments.experiment_config import ModelConfig, TypedQueryWorkload
from src.models.trajectory_qds_model import TrajectoryQDSModel
from src.models.turn_aware_qds_model import TurnAwareQDSModel
from src.queries.query_types import NUM_QUERY_TYPES
from src.training.importance_labels import compute_typed_importance_labels
from src.training.scaler import FeatureScaler
from src.training.trajectory_batching import TrajectoryBatch, batch_windows, build_trajectory_windows

KENDALL_TIE_THRESHOLD = 0.05



def _discriminative_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_each: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top+bottom-quantile subsample for more reliable rank correlation. See src/training/README.md for details.

    Computing Kendall tau on all labelled points is O(N^2) and noisy when the
    label distribution has many near-tied pairs.  Restricting to the top and
    bottom quantiles focuses the statistic on the pairs the ranker is expected
    to separate, where the signal is strongest.
    """
    n = target.numel()
    if n <= 2 * n_each:
        return pred, target
    q = torch.quantile(target, torch.tensor([0.25, 0.75], dtype=torch.float32, device=target.device))
    bot = torch.where(target <= q[0])[0]
    top = torch.where(target >= q[1])[0]
    perm_b = torch.randperm(bot.numel(), generator=generator)[:n_each]
    perm_t = torch.randperm(top.numel(), generator=generator)[:n_each]
    idx = torch.cat([bot[perm_b], top[perm_t]])
    return pred[idx], target[idx]


@dataclass
class TrainingOutputs:
    """Training artifact container. See src/training/README.md for details."""

    model: TrajectoryQDSModel
    scaler: FeatureScaler
    labels: torch.Tensor
    labelled_mask: torch.Tensor
    history: list[dict[str, float]]
    epochs_trained: int = 0


def _sample_epoch_mix(alpha: list[float], generator: torch.Generator) -> torch.Tensor:
    """Sample epoch workload weights from a lightweight Dirichlet approximation. See src/training/README.md for details.

    A uniform floor of 0.5 * (1/K) is mixed in so no type ever receives
    near-zero training signal in a single epoch.  This keeps the per-type
    ranking heads from diverging due to gradient starvation.
    """
    a = torch.tensor(alpha, dtype=torch.float32)
    u = torch.rand((len(alpha),), generator=generator)
    raw = torch.pow(torch.clamp(u, min=1e-6), 1.0 / torch.clamp(a, min=1e-3))
    dirichlet = raw / raw.sum()
    uniform = torch.ones_like(dirichlet) / len(alpha)
    return 0.5 * dirichlet + 0.5 * uniform


def _kendall_tau(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Kendall tau for small vectors without external deps. See src/training/README.md for details."""
    n = int(x.numel())
    if n < 2:
        return 0.0
    dx = x.unsqueeze(0) - x.unsqueeze(1)
    dy = y.unsqueeze(0) - y.unsqueeze(1)
    upper = torch.triu(torch.ones_like(dx, dtype=torch.bool), diagonal=1)
    tie = dy.abs() < KENDALL_TIE_THRESHOLD
    prod = dx * dy
    concordant = int(((prod > 0) & upper & ~tie).sum().item())
    discordant = int(((prod < 0) & upper & ~tie).sum().item())
    denom = max(1, concordant + discordant)
    return float((concordant - discordant) / denom)


def _ranking_loss_for_type(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    pairs_per_type: int,
    top_quantile: float,
    margin: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Compute top-boundary-focused pairwise ranking loss for one type. See src/training/README.md for details."""
    valid_idx = torch.where(valid_mask)[0]
    if valid_idx.numel() < 2:
        return pred.new_tensor(0.0)

    y = target[valid_idx]
    q_val = torch.quantile(y, torch.tensor(top_quantile, dtype=torch.float32, device=y.device))
    top_idx = valid_idx[y >= q_val]
    if top_idx.numel() == 0:
        top_idx = valid_idx

    # Sample pairs one at a time to preserve the interleaved RNG consumption
    # of the original implementation (bit-exact on CPU).  The loss is then
    # computed in a single batched call instead of per-pair, which avoids
    # autograd-graph blowup and lets the backward pass fuse efficiently.
    target_cpu = target.detach().cpu() if target.is_cuda else target
    top_idx_cpu = top_idx.cpu() if top_idx.is_cuda else top_idx
    valid_idx_cpu = valid_idx.cpu() if valid_idx.is_cuda else valid_idx
    a_list: list[int] = []
    b_list: list[int] = []
    for _ in range(pairs_per_type):
        ai = int(torch.randint(0, top_idx_cpu.numel(), (1,), generator=generator).item())
        bi = int(torch.randint(0, valid_idx_cpu.numel(), (1,), generator=generator).item())
        a = int(top_idx_cpu[ai].item())
        b = int(valid_idx_cpu[bi].item())
        if a == b:
            continue
        if bool(torch.isclose(target_cpu[a], target_cpu[b]).item()):
            continue
        a_list.append(a)
        b_list.append(b)

    if not a_list:
        return pred.new_tensor(0.0)

    a_idx = torch.tensor(a_list, dtype=torch.long, device=pred.device)
    b_idx = torch.tensor(b_list, dtype=torch.long, device=pred.device)
    sign = torch.sign(target[a_idx] - target[b_idx])
    return F.margin_ranking_loss(pred[a_idx], pred[b_idx], sign, margin=margin)


def train_model(
    train_trajectories: list[torch.Tensor],
    train_boundaries: list[tuple[int, int]],
    workload: TypedQueryWorkload,
    model_config: ModelConfig,
    seed: int,
) -> TrainingOutputs:
    """Train typed-head model with trajectory-window ranking losses. See src/training/README.md for details."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    all_points = torch.cat(train_trajectories, dim=0)
    point_dim = 8 if model_config.model_type == "turn_aware" else 7
    points = all_points[:, :point_dim].float()

    labels, labelled_mask = compute_typed_importance_labels(
        points=all_points,
        boundaries=train_boundaries,
        typed_queries=workload.typed_queries,
        seed=seed,
    )

    scaler = FeatureScaler.fit(points, workload.query_features)
    norm_points, norm_queries = scaler.transform(points, workload.query_features)

    model_cls = TurnAwareQDSModel if model_config.model_type == "turn_aware" else TrajectoryQDSModel
    model = model_cls(
        point_dim=point_dim,
        query_dim=norm_queries.shape[1],
        embed_dim=model_config.embed_dim,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        type_embed_dim=model_config.type_embed_dim,
        query_chunk_size=model_config.query_chunk_size,
        dropout=model_config.dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    norm_points_dev = norm_points.to(device)
    norm_queries_dev = norm_queries.to(device)
    type_ids_dev = workload.type_ids.to(device)
    labels_dev = labels.to(device)
    labelled_mask_dev = labelled_mask.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=model_config.lr)
    windows_cpu = build_trajectory_windows(
        points=norm_points,
        boundaries=train_boundaries,
        window_length=model_config.window_length,
        stride=model_config.window_stride,
    )
    single_windows = [
        TrajectoryBatch(
            points=w.points.to(device),
            padding_mask=w.padding_mask.to(device),
            trajectory_ids=w.trajectory_ids,
            global_indices=w.global_indices.to(device),
        )
        for w in windows_cpu
    ]
    train_batch_size = max(1, int(getattr(model_config, "train_batch_size", 1)))
    windows = batch_windows(single_windows, train_batch_size)
    # Diagnostic pass operates on single windows so we can cheaply subsample a
    # fraction of them (batched diagnostics would waste the remaining lanes).
    diag_windows = single_windows
    diag_every = max(1, int(getattr(model_config, "diagnostic_every", 1)))
    diag_fraction = float(getattr(model_config, "diagnostic_window_fraction", 1.0))
    diag_fraction = min(1.0, max(0.05, diag_fraction))

    g = torch.Generator().manual_seed(int(seed) + 99)
    # Separate fixed-seed generator for diagnostics so the tau subsample
    # stays consistent across epochs and doesn't oscillate with training state.
    eval_g = torch.Generator().manual_seed(int(seed) + 777)
    history: list[dict[str, float]] = []

    effective_epochs = max(8, int(model_config.epochs))
    run_tag = "main"
    patience = int(getattr(model_config, "early_stopping_patience", 0) or 0)
    best_tau = float("-inf")
    best_loss = float("inf")
    epochs_no_improve = 0
    epoch_w = len(str(effective_epochs))
    epochs_trained = 0
    for epoch in range(effective_epochs):
        epoch_t0 = time.perf_counter()
        model.train()
        epoch_mix = _sample_epoch_mix(model_config.dirichlet_alpha, g)
        epoch_loss = torch.tensor(0.0, device=device)

        for w in windows:
            pred_batch = model(
                points=w.points,
                queries=norm_queries_dev,
                query_type_ids=type_ids_dev,
                padding_mask=w.padding_mask,
            )
            # pred_batch: (B, L, T).  Accumulate per-window loss terms across
            # the batch and backprop once per batch — this is what makes the
            # GPU actually saturated compared to the old batch=1 loop.
            loss_terms: list[torch.Tensor] = []
            B = pred_batch.shape[0]
            for b in range(B):
                idx = w.global_indices[b]
                valid_window = idx >= 0
                global_idx = idx[valid_window]
                pred_valid = pred_batch[b][valid_window]

                for t in range(NUM_QUERY_TYPES):
                    if float(epoch_mix[t].item()) <= 0.0:
                        continue
                    t_labels = labels_dev[global_idx, t]
                    t_mask = labelled_mask_dev[global_idx, t]
                    t_pred = pred_valid[:, t]
                    rank_loss = _ranking_loss_for_type(
                        pred=t_pred,
                        target=t_labels,
                        valid_mask=t_mask,
                        pairs_per_type=model_config.ranking_pairs_per_type,
                        top_quantile=model_config.ranking_top_quantile,
                        margin=model_config.rank_margin,
                        generator=g,
                    )
                    if bool(t_mask.any().item()):
                        mse_term = F.mse_loss(t_pred[t_mask], t_labels[t_mask])
                    else:
                        mse_term = t_pred.new_tensor(0.0)
                    # MSE term keeps scores anchored near label magnitude so the
                    # ranking loss operates on a well-scaled output range rather than
                    # drifting to arbitrary scale.  The small weight (0.05) ensures it
                    # never overrides the ranking signal; the primary objective is
                    # correct intra-trajectory ordering, not absolute score value.
                    loss_terms.append(rank_loss + 0.05 * mse_term)

            if loss_terms:
                loss = (
                    torch.stack(loss_terms).sum() / float(B)
                    + model_config.l2_score_weight * (pred_batch ** 2).mean()
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss = epoch_loss + loss.detach()

        # Diagnostic pass only on selected epochs (every `diag_every` epochs and
        # the final epoch).  Subsample windows by `diag_fraction` to further cut
        # cost: pred_std and tau are statistical aggregates and noise from a
        # ~20% sample is tiny compared to the training noise we're measuring.
        is_last_epoch = (epoch + 1) == effective_epochs
        is_diag_epoch = ((epoch + 1) % diag_every == 0) or is_last_epoch or epoch == 0
        if is_diag_epoch:
            if diag_fraction < 1.0 and len(diag_windows) > 8:
                k = max(8, int(len(diag_windows) * diag_fraction))
                perm = torch.randperm(len(diag_windows), generator=eval_g)[:k].tolist()
                sample_windows = [diag_windows[i] for i in perm]
            else:
                sample_windows = diag_windows

            model.eval()
            with torch.no_grad():
                all_pred = norm_points_dev.new_zeros((norm_points_dev.shape[0], NUM_QUERY_TYPES))
                pred_count = norm_points_dev.new_zeros((norm_points_dev.shape[0],))
                for w in sample_windows:
                    wp = model(
                        points=w.points,
                        queries=norm_queries_dev,
                        query_type_ids=type_ids_dev,
                        padding_mask=w.padding_mask,
                    )[0]
                    widx = w.global_indices[0]
                    valid = widx >= 0
                    all_pred[widx[valid]] = all_pred[widx[valid]] + wp[valid]
                    pred_count[widx[valid]] = pred_count[widx[valid]] + 1.0
                covered_mask = pred_count > 0
                pred_count = pred_count.clamp(min=1.0)
                full_pred = all_pred / pred_count.unsqueeze(1)

            stats: dict[str, float] = {
                "epoch": float(epoch),
                "loss": float(epoch_loss.item() / max(1, len(windows))),
                "pred_std": float(full_pred[covered_mask].std().item()) if bool(covered_mask.any().item()) else 0.0,
            }
            for t in range(NUM_QUERY_TYPES):
                pt = full_pred[:, t]
                stats[f"pred_p50_t{t}"] = float(torch.quantile(pt, 0.50).item())
                stats[f"pred_p90_t{t}"] = float(torch.quantile(pt, 0.90).item())
                stats[f"pred_p99_t{t}"] = float(torch.quantile(pt, 0.99).item())
                eval_mask = labelled_mask_dev[:, t] & covered_mask
                if bool(eval_mask.any().item()):
                    # Reset eval_g to the same state each epoch so the diagnostic
                    # subsample is identical across epochs, giving stable tau trends.
                    eval_g.manual_seed(int(seed) + 777)
                    p_sample, y_sample = _discriminative_sample(
                        pt[eval_mask].detach().cpu(),
                        labels_dev[eval_mask, t].detach().cpu(),
                        n_each=100,
                        generator=eval_g,
                    )
                    stats[f"kendall_tau_t{t}"] = _kendall_tau(p_sample, y_sample)
                else:
                    stats[f"kendall_tau_t{t}"] = 0.0

            if stats["pred_std"] < 1e-3:
                stats["collapse_warning"] = 1.0
        else:
            # Skip diagnostics this epoch; log only loss.  Patience counters
            # are only updated on diagnostic epochs below.
            stats = {
                "epoch": float(epoch),
                "loss": float(epoch_loss.item() / max(1, len(windows))),
            }

        history.append(stats)

        epoch_dt = time.perf_counter() - epoch_t0
        epochs_trained = epoch + 1

        if is_diag_epoch:
            tau_vals = [stats[f"kendall_tau_t{t}"] for t in range(NUM_QUERY_TYPES)]
            avg_tau = sum(tau_vals) / max(1, len(tau_vals))
            collapse = "  COLLAPSE" if stats.get("collapse_warning") else ""
            is_new_best_tau = avg_tau > best_tau + 1e-4
            is_new_best_loss = stats["loss"] < best_loss - 1e-8
            markers = []
            if epoch > 0 and is_new_best_tau:
                markers.append("*** NEW BEST TAU ***")
            if epoch > 0 and is_new_best_loss:
                markers.append("*** NEW BEST LOSS ***")
            best_marker = ("  " + "  ".join(markers)) if markers else ""
            print(
                f"  [{run_tag}] epoch {epoch + 1:0{epoch_w}d}/{effective_epochs}  "
                f"loss={stats['loss']:.8f}  avg_tau={avg_tau:+.3f}  "
                f"pred_std={stats['pred_std']:.3f}  ({epoch_dt:.2f}s){collapse}{best_marker}",
                flush=True,
            )

            if is_new_best_loss:
                best_loss = stats["loss"]

            if patience > 0:
                if is_new_best_tau:
                    best_tau = avg_tau
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(
                            f"  [{run_tag}] early stopping at epoch {epoch + 1:0{epoch_w}d}: "
                            f"avg_tau did not improve over {patience} diag epochs "
                            f"(best_tau={best_tau:+.3f}, best_loss={best_loss:.8f})",
                            flush=True,
                        )
                        break
            else:
                if is_new_best_tau:
                    best_tau = avg_tau
        else:
            # Non-diagnostic epoch: log loss only, no tau / early-stopping update.
            is_new_best_loss = stats["loss"] < best_loss - 1e-8
            if is_new_best_loss:
                best_loss = stats["loss"]
            best_marker = "  *** NEW BEST LOSS ***" if (epoch > 0 and is_new_best_loss) else ""
            print(
                f"  [{run_tag}] epoch {epoch + 1:0{epoch_w}d}/{effective_epochs}  "
                f"loss={stats['loss']:.8f}  (no-diag)  ({epoch_dt:.2f}s){best_marker}",
                flush=True,
            )

    model = model.to("cpu")
    return TrainingOutputs(
        model=model,
        scaler=scaler,
        labels=labels,
        labelled_mask=labelled_mask,
        history=history,
        epochs_trained=epochs_trained,
    )
