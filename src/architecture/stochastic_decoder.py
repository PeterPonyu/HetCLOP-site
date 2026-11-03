"""Heteroscedastic-Gaussian stochastic decoder for Phase 2b.

Same trunk / optimizer / schedule / epochs as the deterministic MLP decoder in
experiments/phase2a_decoder_2026-07-08/decoder_eval.py (embed_dim -> 1024 ->
2048; Linear+LayerNorm+GELU+Dropout(0.1); AdamW lr=1e-3 wd=1e-4; cosine; 50
epochs; batch 256; grad-clip 1.0). ONLY the output head and loss differ:

  * two linear heads off the shared 2048-d trunk: ``mu`` and ``log_var`` per gene
  * beta-NLL loss (Seitzer et al., ICLR 2022, "On the Pitfalls of
    Heteroscedastic Uncertainty Estimation..."): the per-element Gaussian NLL
    ``0.5 * (log_var + (y - mu)^2 / exp(log_var))`` multiplied by
    ``stop_gradient(sigma^2)^beta``, mean over genes and cells. ``beta=0`` is
    plain NLL; ``beta=0.5`` (default) undoes half the ``1/sigma^2`` gradient
    scaling on ``mu`` that otherwise stalls mean learning; ``beta=1`` recovers
    fully MSE-like ``mu`` gradients.
  * log_var is clamped to [logvar_min, logvar_max] for numerical stability
    (default [-8, 3]: variance in [e^-8, e^3] ~ [3e-4, 20]).

Training fixes the documented heteroscedastic-collapse failure mode:
  1. Mean warm-up: the first ``warmup_epochs`` train MSE on ``mu`` only, so the
     log_var head sees no gradient and the mean is fit before variance is
     enabled. beta-NLL takes over for the remaining epochs.
  2. The log_var head gets a lower learning rate (``logvar_lr_scale`` x trunk).

At eval time the model exposes BOTH the mean prediction ``mu`` (for pointwise
reconstruction, which sampling necessarily degrades) and a sampled draw
``mu + exp(0.5*log_var) * eps`` (for distributional metrics).

This is a self-contained reimplementation living under HetCLOP/ per the project
rule of self-contained code."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class HeteroscedasticMLPDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_genes: int, hidden_dims=(1024, 2048),
                 dropout: float = 0.1, logvar_min: float = -8.0, logvar_max: float = 3.0):
        super().__init__()
        layers = []
        dims = [embed_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]),
                       nn.GELU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.mu_head = nn.Linear(dims[-1], num_genes)
        self.logvar_head = nn.Linear(dims[-1], num_genes)
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h).clamp(self.logvar_min, self.logvar_max)
        return mu, log_var


def gaussian_nll(y, mu, log_var, beta: float = 0.0):
    """Mean (over genes and cells) heteroscedastic Gaussian negative
    log-likelihood (dropping the constant 0.5*log(2*pi) term), optionally
    beta-weighted per Seitzer et al. (ICLR 2022).

    Per-element NLL is ``0.5 * (log_var + (y - mu)^2 / exp(log_var))``. When
    ``beta > 0`` each element is multiplied by ``stop_gradient(sigma^2)^beta``
    (with ``sigma^2 = exp(log_var)``), which cancels part of the ``1/sigma^2``
    factor the NLL gradient applies to ``mu`` -- the factor that lets the mean
    stop learning wherever the model predicts high variance. ``beta=0`` is plain
    NLL; ``beta=1`` fully cancels it, recovering MSE-like ``mu`` gradients."""
    per_elem = 0.5 * (log_var + (y - mu) ** 2 / torch.exp(log_var))
    if beta > 0.0:
        per_elem = per_elem * torch.exp(log_var).detach() ** beta
    return per_elem.mean()


def train_stochastic_decoder(
    Z_train, Y_train, Z_val, Y_val, Z_test,
    hidden_dims=(1024, 2048), epochs=50, batch_size=256, lr=1e-3,
    weight_decay=1e-4, seed=42, device="cuda",
    logvar_min=-8.0, logvar_max=3.0,
    beta=0.5, warmup_epochs=15, logvar_lr_scale=0.1,
):
    """Train the heteroscedastic decoder Z->(mu,log_var) with a mean warm-up and
    beta-NLL; early-stop on val loss.

    Schedule:
      * epochs ``[0, warmup_epochs)``: MSE on ``mu`` only. The log_var head is
        excluded from the loss, so it receives no gradient and stays at its
        (near-zero) initialisation while the mean is fit.
      * epochs ``[warmup_epochs, epochs)``: beta-NLL on both heads.
    The log_var head is optimised at ``logvar_lr_scale`` x the base lr.

    Checkpoint selection uses val beta-NLL and only considers post-warmup epochs
    (the log_var head must be trained for the sampled/distributional metrics to
    be meaningful); if ``warmup_epochs >= epochs`` it falls back to the last
    epoch.

    Returns (mu_test, logvar_test, best_val, history, n_params), where mu_test /
    logvar_test are numpy float32 arrays over the test rows. Sampling is left to
    the caller so both the mean and a sampled draw can be evaluated."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    embed_dim = Z_train.shape[1]
    num_genes = Y_train.shape[1]

    Xtr = torch.from_numpy(Z_train).float()
    Ytr = torch.from_numpy(Y_train).float()
    Xva = torch.from_numpy(Z_val).float().to(dev)
    Yva = torch.from_numpy(Y_val).float().to(dev)

    dl = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True,
                    drop_last=True, num_workers=0)
    model = HeteroscedasticMLPDecoder(
        embed_dim, num_genes, hidden_dims=hidden_dims,
        logvar_min=logvar_min, logvar_max=logvar_max).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    # Separate (lower) learning rate for the log_var head; trunk + mu head at lr.
    logvar_params = set(model.logvar_head.parameters())
    trunk_mu_params = [p for p in model.parameters() if p not in logvar_params]
    opt = torch.optim.AdamW(
        [{"params": trunk_mu_params, "lr": lr},
         {"params": list(logvar_params), "lr": lr * logvar_lr_scale}],
        weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = []
    for ep in range(epochs):
        warm = ep < warmup_epochs
        model.train()
        losses = []
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            mu, log_var = model(xb)
            if warm:
                loss = mse(mu, yb)  # log_var head not in loss -> no gradient
            else:
                loss = gaussian_nll(yb, mu, log_var, beta=beta)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            mu_v, lv_v = model(Xva)
            vmse = mse(mu_v, Yva).item()
            vnll = gaussian_nll(Yva, mu_v, lv_v, beta=beta).item()
        vloss = vmse if warm else vnll
        history.append({"epoch": ep + 1, "warmup": bool(warm),
                        "train": float(np.mean(losses)), "val": vloss,
                        "val_mse": vmse, "val_nll": vnll})
        # only checkpoint post-warmup epochs (log_var head must be trained)
        if not warm and vnll < best_val:
            best_val = vnll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is None:  # warmup_epochs >= epochs: no NLL phase
        best_val = history[-1]["val"] if history else float("nan")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    mus, lvs = [], []
    with torch.no_grad():
        for i in range(0, len(Z_test), 4096):
            xb = torch.from_numpy(Z_test[i:i + 4096]).float().to(dev)
            mu, log_var = model(xb)
            mus.append(mu.cpu().numpy())
            lvs.append(log_var.cpu().numpy())
    mu_test = np.concatenate(mus, axis=0).astype(np.float32)
    logvar_test = np.concatenate(lvs, axis=0).astype(np.float32)
    return mu_test, logvar_test, float(best_val), history, int(n_params)


def sample_from(mu: np.ndarray, log_var: np.ndarray, seed: int = 0) -> np.ndarray:
    """Draw one heteroscedastic Gaussian sample per cell/gene:
    ``mu + exp(0.5*log_var) * eps``, eps ~ N(0, I)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(size=mu.shape).astype(np.float32)
    return (mu + np.exp(0.5 * log_var) * eps).astype(np.float32)


# ============================================================================
# Negative-binomial count decoder (Phase 2c robustness — change of likelihood
# family). Trained on RAW integer counts with a per-cell library-size offset, so
# the Gaussian-on-log1p verdict of Phase 2b can be re-tested under a count model.
# ============================================================================

class NegativeBinomialDecoder(nn.Module):
    """Same shared trunk as ``HeteroscedasticMLPDecoder`` (embed_dim -> 1024 ->
    2048; Linear+LayerNorm+GELU+Dropout), with TWO linear heads off the trunk:

      * ``log_mu``    -- per (cell, gene) log expression RATE (library-size free)
      * ``log_theta`` -- per (cell, gene) log inverse-dispersion, clamped to
        ``[logtheta_min, logtheta_max]`` for numerical stability (exactly as
        ``log_var`` is clamped in the Gaussian decoder).

    The NB mean for a cell is recovered by the caller with a library-size offset:
    ``mu_ij = exp(log_mu_ij) * libsize_i / mean_libsize`` (see ``train_nb_decoder``).
    ``theta`` is the inverse-dispersion (scVI convention): NB variance is
    ``mu + mu^2 / theta``, so LARGE ``theta`` -> Poisson-like, SMALL ``theta`` ->
    heavy overdispersion. Default clamp ``[-4, 8]`` -> theta in ~[0.018, 2981]."""

    def __init__(self, embed_dim: int, num_genes: int, hidden_dims=(1024, 2048),
                 dropout: float = 0.1, logtheta_min: float = -4.0, logtheta_max: float = 8.0):
        super().__init__()
        layers = []
        dims = [embed_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]),
                       nn.GELU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.log_mu_head = nn.Linear(dims[-1], num_genes)
        self.log_theta_head = nn.Linear(dims[-1], num_genes)
        self.logtheta_min = logtheta_min
        self.logtheta_max = logtheta_max

    def forward(self, x):
        h = self.trunk(x)
        log_mu = self.log_mu_head(h)
        log_theta = self.log_theta_head(h).clamp(self.logtheta_min, self.logtheta_max)
        return log_mu, log_theta


def nb_nll(y, mu, theta, eps: float = 1e-8):
    """Mean (over cells and genes) negative log-likelihood of counts ``y`` under a
    negative-binomial with mean ``mu`` (> 0) and inverse-dispersion ``theta``
    (> 0), scVI ``log_nb_positive`` parameterisation.

    log p(y) = lgamma(y + theta) - lgamma(theta) - lgamma(y + 1)
             + theta * (log theta - log(theta + mu))
             + y     * (log mu    - log(theta + mu)).

    Exactly the NB whose variance is ``mu + mu^2/theta`` and whose mean is ``mu``;
    as ``theta -> inf`` it reduces to Poisson. Returns the mean NLL (a scalar
    tensor); dropping no constants (the lgamma(y+1) normaliser is kept so the
    value is a true NLL and is minimised at the data-generating (mu, theta))."""
    log_theta_mu = torch.log(theta + mu + eps)
    ll = (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + y * (torch.log(mu + eps) - log_theta_mu)
    )
    return -ll.mean()


def train_nb_decoder(
    Z_train, Y_train, lib_train, Z_val, Y_val, lib_val, Z_test, lib_test,
    mean_libsize: float,
    hidden_dims=(1024, 2048), epochs=50, batch_size=256, lr=1e-3,
    weight_decay=1e-4, seed=42, device="cuda",
    logtheta_min=-4.0, logtheta_max=8.0,
    warmup_epochs=15, logtheta_lr_scale=0.1,
):
    """Train the NB decoder Z->(log_mu, log_theta) on RAW counts with a per-cell
    library-size offset, mirroring the Gaussian decoder's schedule/optimiser.

    ``Y_train`` / ``Y_val`` are RAW integer counts; ``lib_*`` are per-cell total
    counts on the panel; ``mean_libsize`` is the mean train library size. The NB
    mean is ``mu_ij = exp(log_mu_ij) * lib_i / mean_libsize``.

    Schedule (parallels the Gaussian arm so the acceptance gate is comparable):
      * epochs ``[0, warmup_epochs)``: fit the MEAN only, via Poisson NLL on
        ``mu`` (theta head excluded from the loss -> no gradient, stays near
        init). Poisson is the ``theta -> inf`` limit of NB and gives a clean,
        dispersion-free mean warm-up.
      * epochs ``[warmup_epochs, epochs)``: full NB NLL on both heads.
    The log_theta head is optimised at ``logtheta_lr_scale`` x base lr.

    Checkpoint on val NB NLL over post-warmup epochs. Returns
    (mu_test, theta_test, best_val, history, n_params) with mu_test already
    including the test library-size offset (i.e. the per-cell NB mean in COUNT
    space), theta_test the per (cell, gene) inverse-dispersion."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    embed_dim = Z_train.shape[1]
    num_genes = Y_train.shape[1]

    Xtr = torch.from_numpy(Z_train).float()
    Ytr = torch.from_numpy(Y_train).float()
    Otr = torch.from_numpy((lib_train / mean_libsize).astype(np.float32)).float().view(-1, 1)
    Xva = torch.from_numpy(Z_val).float().to(dev)
    Yva = torch.from_numpy(Y_val).float().to(dev)
    Ova = torch.from_numpy((lib_val / mean_libsize).astype(np.float32)).float().view(-1, 1).to(dev)

    dl = DataLoader(TensorDataset(Xtr, Ytr, Otr), batch_size=batch_size, shuffle=True,
                    drop_last=True, num_workers=0)
    model = NegativeBinomialDecoder(
        embed_dim, num_genes, hidden_dims=hidden_dims,
        logtheta_min=logtheta_min, logtheta_max=logtheta_max).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    logtheta_params = set(model.log_theta_head.parameters())
    trunk_mu_params = [p for p in model.parameters() if p not in logtheta_params]
    opt = torch.optim.AdamW(
        [{"params": trunk_mu_params, "lr": lr},
         {"params": list(logtheta_params), "lr": lr * logtheta_lr_scale}],
        weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def poisson_nll(y, mu, eps=1e-8):
        # Poisson NLL (theta -> inf limit), dropping lgamma(y+1) constant is fine
        # for a warm-up objective; keep it for a comparable magnitude.
        return (mu - y * torch.log(mu + eps) + torch.lgamma(y + 1.0)).mean()

    best_val = float("inf")
    best_state = None
    history = []
    for ep in range(epochs):
        warm = ep < warmup_epochs
        model.train()
        losses = []
        for xb, yb, ob in dl:
            xb, yb, ob = xb.to(dev), yb.to(dev), ob.to(dev)
            log_mu, log_theta = model(xb)
            mu = torch.exp(log_mu) * ob
            if warm:
                loss = poisson_nll(yb, mu)  # theta head not in loss -> no gradient
            else:
                theta = torch.exp(log_theta)
                loss = nb_nll(yb, mu, theta)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            lmu_v, lth_v = model(Xva)
            mu_v = torch.exp(lmu_v) * Ova
            vpois = poisson_nll(Yva, mu_v).item()
            vnb = nb_nll(Yva, mu_v, torch.exp(lth_v)).item()
        vloss = vpois if warm else vnb
        history.append({"epoch": ep + 1, "warmup": bool(warm),
                        "train": float(np.mean(losses)), "val": vloss,
                        "val_poisson": vpois, "val_nb": vnb})
        if not warm and vnb < best_val:
            best_val = vnb
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_val = history[-1]["val"] if history else float("nan")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    mus, thetas = [], []
    with torch.no_grad():
        for i in range(0, len(Z_test), 4096):
            xb = torch.from_numpy(Z_test[i:i + 4096]).float().to(dev)
            ob = torch.from_numpy(
                (lib_test[i:i + 4096] / mean_libsize).astype(np.float32)).float().view(-1, 1).to(dev)
            lmu, lth = model(xb)
            mus.append((torch.exp(lmu) * ob).cpu().numpy())
            thetas.append(torch.exp(lth).cpu().numpy())
    mu_test = np.concatenate(mus, axis=0).astype(np.float32)
    theta_test = np.concatenate(thetas, axis=0).astype(np.float32)
    return mu_test, theta_test, float(best_val), history, int(n_params)


def sample_nb(mu: np.ndarray, theta: np.ndarray, seed: int = 0) -> np.ndarray:
    """Draw one NB count per cell/gene via the Gamma-Poisson mixture:
    ``lam ~ Gamma(shape=theta, scale=mu/theta)``, ``y ~ Poisson(lam)``. This is
    exactly NB(mean=mu, inverse-dispersion=theta): E[y]=mu, Var[y]=mu+mu^2/theta.
    Returns float32 counts (>= 0)."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    theta = np.clip(theta, 1e-8, None)
    scale = np.clip(mu, 0.0, None) / theta
    lam = rng.gamma(shape=theta, scale=scale)
    return rng.poisson(lam).astype(np.float32)


def sample_nb_fixed_theta(mu: np.ndarray, theta: float, seed: int = 0) -> np.ndarray:
    """Draw NB counts around mean ``mu`` with a single IMPOSED, scalar
    inverse-dispersion ``theta`` applied to every (cell, gene) -- i.e. holding
    the mean fixed and overriding the learned per-gene dispersion with a constant.

    Used by the Phase 2d theta-sweep to test the over-inflation mechanism: with
    ``mu`` held at NB's own fitted count mean, only ``theta`` varies. NB variance
    is ``mu + mu^2/theta`` so LARGE theta -> Poisson-like (variance -> mu), SMALL
    theta -> heavy overdispersion. ``theta = inf`` (or non-finite) is the exact
    Poisson limit and is sampled directly as ``Poisson(mu)`` (the Gamma-Poisson
    mixture is singular there). Returns float32 counts (>= 0)."""
    rng = np.random.default_rng(seed)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 0.0, None)
    if not np.isfinite(theta):
        return rng.poisson(mu).astype(np.float32)
    if theta <= 0.0:
        raise ValueError("theta must be > 0 (or inf for the Poisson limit)")
    scale = mu / float(theta)          # elementwise Gamma scale = mu/theta
    lam = rng.gamma(shape=float(theta), scale=scale)  # E[lam]=mu, Var=mu^2/theta
    return rng.poisson(lam).astype(np.float32)


# ============================================================================
# Zero-inflated negative-binomial (ZINB) count decoder (Phase 2i G5 -- one extra
# likelihood family). Same trunk / offset / schedule as the NB decoder, plus a
# LEARNED per-gene zero-inflation logit (a single vector, NOT conditioned on the
# latent -- "learned per-gene zero-inflation logit"). Tests whether droplet
# scRNA-seq benefits from explicit zero inflation on top of NB (Svensson 2020
# argues it does not; a null here corroborates that).
# ============================================================================

class ZeroInflatedNBDecoder(nn.Module):
    """``NegativeBinomialDecoder``'s shared trunk + ``log_mu`` and ``log_theta``
    heads, plus a single LEARNED per-gene zero-inflation logit ``zi_logit`` (a
    length-``num_genes`` parameter, shared across cells -- the standard scVI ZINB
    parameterisation where the dropout probability is a per-gene property, not a
    per-cell one).

    ``pi_g = sigmoid(zi_logit_g)`` is the probability that gene ``g``'s count is an
    inflated (structural) zero; with prob ``1 - pi_g`` the count is drawn from
    NB(mu, theta). ``zi_logit`` is initialised at ``zi_init`` (default -2.0 ->
    pi ~ 0.12) so training starts near the NB (low-inflation) regime and the data
    must actively pull inflation up if it helps."""

    def __init__(self, embed_dim: int, num_genes: int, hidden_dims=(1024, 2048),
                 dropout: float = 0.1, logtheta_min: float = -4.0, logtheta_max: float = 8.0,
                 zi_init: float = -2.0):
        super().__init__()
        layers = []
        dims = [embed_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]),
                       nn.GELU(), nn.Dropout(dropout)]
        self.trunk = nn.Sequential(*layers)
        self.log_mu_head = nn.Linear(dims[-1], num_genes)
        self.log_theta_head = nn.Linear(dims[-1], num_genes)
        self.zi_logit = nn.Parameter(torch.full((num_genes,), float(zi_init)))
        self.logtheta_min = logtheta_min
        self.logtheta_max = logtheta_max

    def forward(self, x):
        h = self.trunk(x)
        log_mu = self.log_mu_head(h)
        log_theta = self.log_theta_head(h).clamp(self.logtheta_min, self.logtheta_max)
        return log_mu, log_theta, self.zi_logit


def zinb_nll(y, mu, theta, zi_logit, eps: float = 1e-8):
    """Mean (over cells and genes) negative log-likelihood of counts ``y`` under a
    zero-inflated negative binomial with NB mean ``mu`` (>0), inverse-dispersion
    ``theta`` (>0), and per-gene zero-inflation logit ``zi_logit`` (broadcast over
    cells).

    With ``pi = sigmoid(zi_logit)`` the mixture density is
      p(y=0) = pi + (1 - pi) * NB(0; mu, theta),
      p(y>0) = (1 - pi) * NB(y; mu, theta),
    where ``log NB(0) = theta*(log theta - log(theta + mu))``. Written with the
    numerically stable ``log sigmoid`` identities
    ``log pi = -softplus(-zi)`` and ``log(1-pi) = -softplus(zi)`` and a
    ``logaddexp`` for the zero branch. As ``zi_logit -> -inf`` (pi -> 0) this
    reduces EXACTLY to ``nb_nll`` (verified in tests)."""
    softplus = torch.nn.functional.softplus
    log_theta_mu = torch.log(theta + mu + eps)
    log_nb_zero = theta * (torch.log(theta + eps) - log_theta_mu)
    nb_ll = (
        torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + y * (torch.log(mu + eps) - log_theta_mu)
    )
    log_pi = -softplus(-zi_logit)     # log sigmoid(zi)   -> log pi
    log_1mpi = -softplus(zi_logit)    # log(1 - sigmoid(zi)) -> log(1 - pi)
    zero_case = torch.logaddexp(log_pi + torch.zeros_like(nb_ll),
                                log_1mpi + log_nb_zero)
    nonzero_case = log_1mpi + nb_ll
    ll = torch.where(y < 0.5, zero_case, nonzero_case)
    return -ll.mean()


def train_zinb_decoder(
    Z_train, Y_train, lib_train, Z_val, Y_val, lib_val, Z_test, lib_test,
    mean_libsize: float,
    hidden_dims=(1024, 2048), epochs=50, batch_size=256, lr=1e-3,
    weight_decay=1e-4, seed=42, device="cuda",
    logtheta_min=-4.0, logtheta_max=8.0,
    warmup_epochs=15, logtheta_lr_scale=0.1, zi_init=-2.0,
):
    """Train the ZINB decoder Z->(log_mu, log_theta, zi_logit) on RAW counts with a
    per-cell library-size offset, mirroring ``train_nb_decoder`` exactly so the
    acceptance gate and every metric stay comparable.

    Schedule: epochs ``[0, warmup_epochs)`` fit the MEAN only via Poisson NLL on
    ``mu`` (theta head AND the zi logit excluded from the loss -> no gradient);
    epochs ``[warmup_epochs, epochs)`` full ZINB NLL on all parameters. The
    ``log_theta`` head and the ``zi_logit`` parameter are optimised at
    ``logtheta_lr_scale`` x base lr. Checkpoint on val ZINB NLL over post-warmup
    epochs. Returns (mu_test, theta_test, zi_logit, best_val, history, n_params)
    with ``mu_test`` already including the test library-size offset (per-cell NB
    mean in COUNT space) and ``zi_logit`` the length-num_genes learned logit."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    embed_dim = Z_train.shape[1]
    num_genes = Y_train.shape[1]

    Xtr = torch.from_numpy(Z_train).float()
    Ytr = torch.from_numpy(Y_train).float()
    Otr = torch.from_numpy((lib_train / mean_libsize).astype(np.float32)).float().view(-1, 1)
    Xva = torch.from_numpy(Z_val).float().to(dev)
    Yva = torch.from_numpy(Y_val).float().to(dev)
    Ova = torch.from_numpy((lib_val / mean_libsize).astype(np.float32)).float().view(-1, 1).to(dev)

    dl = DataLoader(TensorDataset(Xtr, Ytr, Otr), batch_size=batch_size, shuffle=True,
                    drop_last=True, num_workers=0)
    model = ZeroInflatedNBDecoder(
        embed_dim, num_genes, hidden_dims=hidden_dims,
        logtheta_min=logtheta_min, logtheta_max=logtheta_max, zi_init=zi_init).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    slow_params = set(model.log_theta_head.parameters()) | {model.zi_logit}
    fast_params = [p for p in model.parameters() if p not in slow_params]
    opt = torch.optim.AdamW(
        [{"params": fast_params, "lr": lr},
         {"params": list(slow_params), "lr": lr * logtheta_lr_scale}],
        weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def poisson_nll(y, mu, eps=1e-8):
        return (mu - y * torch.log(mu + eps) + torch.lgamma(y + 1.0)).mean()

    best_val = float("inf")
    best_state = None
    history = []
    for ep in range(epochs):
        warm = ep < warmup_epochs
        model.train()
        losses = []
        for xb, yb, ob in dl:
            xb, yb, ob = xb.to(dev), yb.to(dev), ob.to(dev)
            log_mu, log_theta, zi = model(xb)
            mu = torch.exp(log_mu) * ob
            if warm:
                loss = poisson_nll(yb, mu)  # theta head + zi logit excluded -> no gradient
            else:
                loss = zinb_nll(yb, mu, torch.exp(log_theta), zi)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            lmu_v, lth_v, zi_v = model(Xva)
            mu_v = torch.exp(lmu_v) * Ova
            vpois = poisson_nll(Yva, mu_v).item()
            vzinb = zinb_nll(Yva, mu_v, torch.exp(lth_v), zi_v).item()
        vloss = vpois if warm else vzinb
        history.append({"epoch": ep + 1, "warmup": bool(warm),
                        "train": float(np.mean(losses)), "val": vloss,
                        "val_poisson": vpois, "val_zinb": vzinb})
        if not warm and vzinb < best_val:
            best_val = vzinb
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_val = history[-1]["val"] if history else float("nan")
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    mus, thetas = [], []
    with torch.no_grad():
        zi_logit = model.zi_logit.detach().cpu().numpy().astype(np.float32)
        for i in range(0, len(Z_test), 4096):
            xb = torch.from_numpy(Z_test[i:i + 4096]).float().to(dev)
            ob = torch.from_numpy(
                (lib_test[i:i + 4096] / mean_libsize).astype(np.float32)).float().view(-1, 1).to(dev)
            lmu, lth, _ = model(xb)
            mus.append((torch.exp(lmu) * ob).cpu().numpy())
            thetas.append(torch.exp(lth).cpu().numpy())
    mu_test = np.concatenate(mus, axis=0).astype(np.float32)
    theta_test = np.concatenate(thetas, axis=0).astype(np.float32)
    return mu_test, theta_test, zi_logit, float(best_val), history, int(n_params)


def sample_zinb(mu: np.ndarray, theta: np.ndarray, zi_logit: np.ndarray,
                seed: int = 0) -> np.ndarray:
    """Draw one ZINB count per cell/gene: sample NB(mu, theta), then independently
    zero out gene ``g``'s value with probability ``pi_g = sigmoid(zi_logit_g)``.
    ``zi_logit`` is a per-gene vector broadcast over cells. Returns float32 counts
    (>= 0). As ``zi_logit -> -inf`` (pi -> 0) this is exactly ``sample_nb``."""
    y = sample_nb(mu, theta, seed=seed)
    zi = np.asarray(zi_logit, dtype=np.float64)
    pi = 1.0 / (1.0 + np.exp(-zi))
    rng = np.random.default_rng(seed + 1)
    drop = rng.random(size=y.shape) < pi[None, :]
    y = y.copy()
    y[drop] = 0.0
    return y.astype(np.float32)


def zi_logit_stats(zi_logit: np.ndarray) -> dict:
    """Summary of the learned per-gene zero-inflation: the implied dropout
    probabilities ``pi = sigmoid(zi_logit)`` (min/median/max) and the fraction of
    genes with pi > 0.5. A ZINB that reduces to NB has all pi ~ 0."""
    zi = np.asarray(zi_logit, dtype=np.float64)
    pi = 1.0 / (1.0 + np.exp(-zi))
    return {
        "zi_logit_min": float(zi.min()), "zi_logit_median": float(np.median(zi)),
        "zi_logit_max": float(zi.max()),
        "pi_min": float(pi.min()), "pi_median": float(np.median(pi)),
        "pi_max": float(pi.max()),
        "frac_genes_pi_gt_0.5": float(np.mean(pi > 0.5)),
        "frac_genes_pi_gt_0.1": float(np.mean(pi > 0.1)),
    }


def log_theta_stats(log_theta: np.ndarray, logtheta_min=-4.0, logtheta_max=8.0) -> dict:
    """min/median/max of log_theta plus the fraction pinned at each clamp
    boundary (saturation diagnostic, exactly as ``logvar_stats`` does for the
    Gaussian arm)."""
    lt = np.asarray(log_theta, dtype=np.float64)
    return {
        "min": float(lt.min()), "median": float(np.median(lt)), "max": float(lt.max()),
        "frac_at_min_clamp": float(np.mean(lt <= logtheta_min + 1e-4)),
        "frac_at_max_clamp": float(np.mean(lt >= logtheta_max - 1e-4)),
        "theta_median": float(np.exp(np.median(lt))),
    }
