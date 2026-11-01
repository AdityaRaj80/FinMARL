"""TFT and PatchTST forecasters with one shared dual-output head, so they differ only in the encoder.

Both read a window of past features plus stock and sector identity, and forecast the 5-day return
distribution and 5-day realised variance. Forecasts are corrections to an EWMA variance base, so an
untrained model starts from that simple forecast.

TFT here keeps the published core (static contexts, variable selection, gated LSTM, interpretable causal
attention) without a decoder, because every input is a past observation and the horizon is single.
PatchTST keeps patching, channel independence and its transformer; channels are then pooled by attention
so that one stock-level forecast comes out. RevIN is left out because inputs are already scaled.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

MEAN_SCALE = 0.01    # mean head output is in units of 1% over the horizon
NU_START = 6.0
VARIANCE_FLOOR = 1e-8


def student_t_nll(y, mu, log_sigma, nu):
    z2 = ((y - mu) * torch.exp(-log_sigma)) ** 2
    return (torch.lgamma(nu / 2) - torch.lgamma((nu + 1) / 2) + 0.5 * torch.log(nu * math.pi)
            + log_sigma + (nu + 1) / 2 * torch.log1p(z2 / nu)).mean()


def qlike_loss(y, log_h):
    log_ratio = torch.log(y.clamp(min=VARIANCE_FLOOR)) - log_h
    return (torch.exp(log_ratio) - log_ratio - 1).mean()


def joint_loss(y, rv, mu, log_sigma, nu, log_rv, weight):
    return student_t_nll(y, mu, log_sigma, nu) + weight * qlike_loss(rv, log_rv)


class ForecastHead(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Dropout(dropout), nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.raw_nu = nn.Parameter(torch.tensor(math.log(math.expm1(NU_START - 2))))

    def forward(self, h, base):
        out = self.net(h)
        nu = 2 + F.softplus(self.raw_nu)
        return MEAN_SCALE * out[:, 0], base / 2 + out[:, 1], nu, base + out[:, 2]


class GRN(nn.Module):
    """Gated residual network from the TFT paper."""

    def __init__(self, n_in, hidden, n_out, dropout, context=0):
        super().__init__()
        self.fc1 = nn.Linear(n_in + context, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden, 2 * n_out)
        self.skip = nn.Linear(n_in, n_out) if n_in != n_out else nn.Identity()
        self.norm = nn.LayerNorm(n_out)

    def forward(self, x, context=None):
        inp = x if context is None else torch.cat([x, context], dim=-1)
        h = self.drop(self.fc2(F.elu(self.fc1(inp))))
        return self.norm(self.skip(x) + F.glu(self.gate(h), dim=-1))


class GatedAddNorm(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden, 2 * hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, skip):
        return self.norm(skip + F.glu(self.gate(self.drop(x)), dim=-1))


class CausalAttention(nn.Module):
    """Interpretable multi-head attention: heads share one value projection, and no step sees a later step."""

    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.heads, self.d = heads, hidden // heads
        self.q = nn.Linear(hidden, heads * self.d)
        self.k = nn.Linear(hidden, heads * self.d)
        self.v = nn.Linear(hidden, self.d)
        self.out = nn.Linear(self.d, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.heads, self.d).transpose(1, 2)
        k = self.k(x).view(b, t, self.heads, self.d).transpose(1, 2)
        score = q @ k.transpose(-1, -2) / math.sqrt(self.d)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), 1)
        weights = self.drop(score.masked_fill(mask, -torch.inf).softmax(dim=-1)).mean(dim=1)
        return self.out(weights @ self.v(x))


class TFT(nn.Module):
    def __init__(self, n_features, n_tickers, n_sectors, hidden, heads, dropout):
        super().__init__()
        self.ticker = nn.Embedding(n_tickers, hidden)
        self.sector = nn.Embedding(n_sectors, hidden)
        self.static = GRN(2 * hidden, hidden, hidden, dropout)
        self.ctx_select, self.ctx_h, self.ctx_c, self.ctx_enrich = (GRN(hidden, hidden, hidden, dropout) for _ in range(4))
        self.embed_w = nn.Parameter(torch.randn(n_features, hidden) * 0.1)
        self.embed_b = nn.Parameter(torch.zeros(n_features, hidden))
        self.var_grn = GRN(hidden, hidden, hidden, dropout)
        self.selector = GRN(n_features * hidden, hidden, n_features, dropout, context=hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.post_lstm = GatedAddNorm(hidden, dropout)
        self.enrich = GRN(hidden, hidden, hidden, dropout, context=hidden)
        self.attention = CausalAttention(hidden, heads, dropout)
        self.post_attention = GatedAddNorm(hidden, dropout)
        self.position = GRN(hidden, hidden, hidden, dropout)
        self.post_position = GatedAddNorm(hidden, dropout)
        self.head = ForecastHead(hidden, dropout)

    def _context(self, static):
        return self.static(torch.cat([self.ticker(static[:, 0]), self.sector(static[:, 1])], dim=-1))

    def _select(self, x, context):
        embedded = x[..., None] * self.embed_w + self.embed_b
        b, t, _, _ = embedded.shape
        ctx = self.ctx_select(context)[:, None].expand(b, t, -1)
        weights = self.selector(embedded.flatten(2), ctx).softmax(dim=-1)
        return embedded, weights

    def variable_weights(self, x, static):
        """Share of attention each input receives at each step, for explaining a forecast."""
        return self._select(x, self._context(static))[1]

    def encode(self, x, static):
        c = self._context(static)
        embedded, weights = self._select(x, c)
        selected = (self.var_grn(embedded) * weights[..., None]).sum(dim=2)
        state = (self.ctx_h(c)[None], self.ctx_c(c)[None])
        lstm_out, _ = self.lstm(selected, state)
        h = self.post_lstm(lstm_out, selected)
        enriched = self.enrich(h, self.ctx_enrich(c)[:, None].expand(-1, h.shape[1], -1))
        attended = self.post_attention(self.attention(enriched), enriched)
        return self.post_position(self.position(attended), h)

    def forward(self, x, static, base):
        return self.head(self.encode(x, static)[:, -1], base)


class PatchTST(nn.Module):
    def __init__(self, n_features, n_tickers, n_sectors, lookback, hidden, heads, dropout, patch, stride, layers):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.n_patches = (lookback - patch) // stride + 1
        self.project = nn.Linear(patch, hidden)
        self.position = nn.Parameter(torch.randn(self.n_patches, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.flatten = nn.Linear(self.n_patches * hidden, hidden)
        self.channel = nn.Parameter(torch.randn(n_features, hidden) * 0.02)
        self.ticker = nn.Embedding(n_tickers, hidden)
        self.sector = nn.Embedding(n_sectors, hidden)
        self.query = nn.Linear(2 * hidden, hidden)
        self.fuse = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ELU(), nn.Dropout(dropout))
        self.head = ForecastHead(hidden, dropout)

    def forward(self, x, static, base):
        b, _, f = x.shape
        patches = x.transpose(1, 2).unfold(2, self.patch, self.stride)
        z = self.project(patches.reshape(b * f, self.n_patches, self.patch)) + self.position
        z = self.flatten(self.encoder(z).flatten(1)).view(b, f, -1) + self.channel
        identity = torch.cat([self.ticker(static[:, 0]), self.sector(static[:, 1])], dim=-1)
        score = (z * self.query(identity)[:, None]).sum(dim=-1) / math.sqrt(z.shape[-1])
        pooled = (score.softmax(dim=-1)[..., None] * z).sum(dim=1)
        return self.head(self.fuse(torch.cat([pooled, identity], dim=-1)), base)
