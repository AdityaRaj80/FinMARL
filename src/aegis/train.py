"""Training loop shared by both forecasters, plus prediction and seed ensembling."""

import copy
from types import SimpleNamespace

import numpy as np
import torch

from aegis.deep import joint_loss

EVAL_BATCH = 4096


def _chunks(d, s, size):
    for i in range(0, len(d), size):
        yield d[i:i + size], s[i:i + size]


def _forward(model, panel, d, s, device):
    x, static, y, rv, base = panel.batch(d, s, device)
    return model(x, static, base), y, rv


@torch.no_grad()
def stop_loss(model, panel, idx, config, device):
    """Mean joint loss over the early-stopping rows."""
    model.eval()
    d, s = torch.from_numpy(idx.stop_d), torch.from_numpy(idx.stop_s)
    total = 0.0
    for dd, ss in _chunks(d, s, EVAL_BATCH):
        (mu, log_sigma, nu, log_rv), y, rv = _forward(model, panel, dd, ss, device)
        total += joint_loss(y, rv, mu, log_sigma, nu, log_rv, config["variance_loss_weight"]).item() * len(dd)
    return total / len(d)


def fit_model(build, panel, idx, config, seed, device):
    """Train with early stopping. Returns the model at its best epoch and the loss after each epoch.

    build is a function that creates the model, so the seed also controls its starting weights.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = build().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    d, s = torch.from_numpy(idx.fit_d), torch.from_numpy(idx.fit_s)
    best, best_state, stale, history = np.inf, None, 0, []
    for _ in range(config["max_epochs"]):
        model.train()
        order = torch.randperm(len(d), generator=generator)
        for dd, ss in _chunks(d[order], s[order], config["batch_size"]):
            (mu, log_sigma, nu, log_rv), y, rv = _forward(model, panel, dd, ss, device)
            loss = joint_loss(y, rv, mu, log_sigma, nu, log_rv, config["variance_loss_weight"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss with seed {seed}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            opt.step()
        history.append(stop_loss(model, panel, idx, config, device))
        if history[-1] < best:
            best, stale = history[-1], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, panel, d, s, device):
    model.eval()
    out = dict(mu=[], sigma=[], nu=[], rv=[])
    for dd, ss in _chunks(torch.from_numpy(d), torch.from_numpy(s), EVAL_BATCH):
        (mu, log_sigma, nu, log_rv), _, _ = _forward(model, panel, dd, ss, device)
        out["mu"].append(mu.cpu().numpy())
        out["sigma"].append(log_sigma.exp().cpu().numpy())
        out["nu"].append(np.full(len(dd), nu.item(), np.float32))
        out["rv"].append(log_rv.exp().cpu().numpy())
    return SimpleNamespace(**{k: np.concatenate(v) for k, v in out.items()})


def ensemble(runs):
    """Combine seeds: average means, variances and degrees of freedom."""
    stack = {k: np.stack([r[k] for r in runs]) for k in ("mu", "sigma", "nu", "rv")}
    return dict(mu=stack["mu"].mean(0), sigma=np.sqrt((stack["sigma"] ** 2).mean(0)),
                nu=stack["nu"].mean(0), rv=stack["rv"].mean(0))
