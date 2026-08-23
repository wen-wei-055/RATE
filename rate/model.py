"""The RATE network.

Waveforms and station coordinates are embedded per station, a transformer mixes
information across stations, and every station's slot is decoded into a mixture
of Gaussians over its log10 peak ground acceleration.  When a historical event
is retrieved it simply extends the station axis: the transformer sees
``stations * 2`` slots and attends across both blocks, and the second block is
dropped again when the predictions are read out.

Sub-module attributes are CapWords (``self.TotalEmbedding`` and friends)
because their names are the keys of the checkpoints published with the paper.
Renaming them would make every existing checkpoint unloadable, so they stay.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class MLP(nn.Module):
    def __init__(self, input_shape, dims):
        super().__init__()
        self.linears = nn.ModuleList()
        widths = [input_shape[0], *dims]
        for a, b in zip(widths[:-1], widths[1:]):
            self.linears.append(nn.Linear(in_features=a, out_features=b, bias=True))
        for layer in self.linears:
            nn.init.xavier_uniform_(layer.weight.data)
            nn.init.constant_(layer.bias.data, 0)

    def forward(self, x):
        for layer in self.linears:
            x = F.relu(layer(x))
        return x


class MixtureOutput(nn.Module):
    """Head producing ``n`` Gaussians per station: weight, mean, std."""

    def __init__(self, input_shape, n, d=1, eps=1e-4, bias_mu=1.8, bias_sigma=0.2):
        super().__init__()
        self.eps = eps
        self.n = n
        self.d = d
        self.alpha_linear = nn.Linear(input_shape[-1], n)
        self.mu_linear = nn.Linear(input_shape[-1], n * d)
        self.sigma_linear = nn.Linear(input_shape[-1], n * d)

        for layer, bias in (
            (self.alpha_linear, 0.0),
            (self.mu_linear, bias_mu),
            (self.sigma_linear, bias_sigma),
        ):
            nn.init.xavier_uniform_(layer.weight.data)
            nn.init.constant_(layer.bias.data, bias)

    def forward(self, x):
        alpha = F.softmax(self.alpha_linear(x), dim=-1).reshape(-1, self.n, 1)
        mu = self.mu_linear(x).reshape(-1, self.n, self.d)
        # Epsilon keeps the density finite for a collapsed component.
        sigma = (F.relu(self.sigma_linear(x)) + self.eps).reshape(-1, self.n, self.d)
        return torch.cat((alpha, mu, sigma), 2)


class NormalizedScaleEmbedding(nn.Module):
    """Embeds one station's waveform, scale-free plus its scale as one number."""

    def __init__(self, input_shape, mlp_dims, downsample=1, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.inp_shape = input_shape
        channels = input_shape[-1]
        self.Conv2D_1 = nn.Conv2d(1, 8, kernel_size=(downsample, 1), stride=(downsample, 1))
        self.Conv2D_2 = nn.Conv2d(8, 32, kernel_size=(16, 3), stride=(1, 3))
        self.Conv1D_1 = nn.Conv1d(32 * channels // 3, 64, kernel_size=16)
        self.Conv1D_2 = nn.Conv1d(64, 128, kernel_size=16)
        self.Conv1D_3 = nn.Conv1d(128, 32, kernel_size=8)
        self.Conv1D_4 = nn.Conv1d(32, 32, kernel_size=8)
        self.Conv1D_5 = nn.Conv1d(32, 16, kernel_size=4)
        self.pool1D_1 = nn.MaxPool1d(kernel_size=2)
        self.pool1D_2 = nn.MaxPool1d(kernel_size=2)
        self.pool1D_3 = nn.MaxPool1d(kernel_size=2)

        # Width of the convolution stack, measured rather than hard coded.
        with torch.no_grad():
            width = self._convolve(torch.zeros(1, *input_shape)).shape[-1]
        self.MLP = MLP(input_shape=(width + 1,), dims=mlp_dims)

        for conv in (self.Conv2D_1, self.Conv2D_2, self.Conv1D_1, self.Conv1D_2,
                     self.Conv1D_3, self.Conv1D_4, self.Conv1D_5):
            nn.init.xavier_uniform_(conv.weight.data)
            nn.init.constant_(conv.bias.data, 0)

    def _convolve(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.relu(self.Conv2D_1(x))
        x = F.relu(self.Conv2D_2(x))
        x = torch.reshape(x, (x.shape[0], 32 * self.inp_shape[-1] // 3, -1))
        x = self.pool1D_1(F.relu(self.Conv1D_1(x)))
        x = self.pool1D_2(F.relu(self.Conv1D_2(x)))
        x = self.pool1D_3(F.relu(self.Conv1D_3(x)))
        x = F.relu(self.Conv1D_4(x))
        x = F.relu(self.Conv1D_5(x))
        return torch.flatten(x, 1)

    def forward(self, x):
        peak = torch.amax(torch.abs(x), dim=(1, 2), keepdims=True) + self.eps
        scale = torch.log(torch.squeeze(peak, 2)) / 100
        x = self._convolve(x / peak)
        return self.MLP(torch.cat((x, scale), -1))


class TimeDistributed(nn.Module):
    """Applies a module to every station of a batch independently."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        if x.dim() == 4:
            y = self.module(x.contiguous().view(-1, x.size(-2), x.size(-1)))
        elif x.dim() == 3:
            y = self.module(x.contiguous().view(-1, x.size(-1)))
        else:
            raise ValueError(f"TimeDistributed expects a 3D or 4D input, got {tuple(x.shape)}")

        if y.dim() == 3:
            return y.contiguous().view(x.size(0), x.size(1), -1, y.size(-1))
        if y.dim() == 2:
            return y.contiguous().view(x.size(0), -1, y.size(-1))
        raise ValueError(f"TimeDistributed cannot reshape a {y.dim()}D output")


class LayerNorm(nn.Module):
    """Kept instead of ``nn.LayerNorm``: the parameter names are checkpoint keys."""

    def __init__(self, input_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(input_shape[-1:]))
        self.gamma = nn.Parameter(torch.ones(input_shape[-1:]))

    def forward(self, x):
        m = x.mean(-1, keepdims=True)
        s = torch.mean(torch.square(x - m), dim=-1, keepdims=True)
        return self.gamma * (x - m) / torch.sqrt(s + self.eps) + self.beta


class PositionEmbedding(nn.Module):
    """Sinusoidal embedding of a station's position on the globe.

    Latitude, longitude and depth each get their own geometric series of
    wavelengths, interleaved into one vector so that nearby stations get
    similar embeddings at every scale.
    """

    def __init__(self, wavelength, emb_dim, borehole=False, rotation=None, rotation_anchor=None):
        super().__init__()
        self.emb_dim = emb_dim
        self.borehole = borehole
        self.rotation = rotation
        self.rotation_anchor = rotation_anchor
        if rotation is not None and rotation_anchor is None:
            raise ValueError("Rotations in the positional embedding require a rotation anchor")
        if emb_dim % 10 != 0 or (borehole and emb_dim % 20 != 0):
            raise ValueError("emb_dim must be a multiple of 10, or of 20 for borehole data")

        matrix = None
        if rotation is not None:
            c, s = np.cos(rotation), np.sin(rotation)
            matrix = torch.tensor(np.array(((c, -s), (s, c)), dtype="float32"))
        # Buffers, so device placement follows the module, but non-persistent
        # so they stay out of the checkpoint.
        self.register_buffer("rotation_matrix", matrix, persistent=False)

        (min_lat, max_lat), (min_lon, max_lon), (min_depth, max_depth) = wavelength
        lat_dim = lon_dim = emb_dim // 5
        depth_dim = emb_dim // 20 if borehole else emb_dim // 10

        def coefficients(low, high, dim):
            series = (low / high) ** (np.arange(dim) / dim)
            return torch.from_numpy((2 * np.pi / low * series).astype("float32"))

        self.register_buffer("lat_coeff", coefficients(min_lat, max_lat, lat_dim), persistent=False)
        self.register_buffer("lon_coeff", coefficients(min_lon, max_lon, lon_dim), persistent=False)
        self.register_buffer("depth_coeff", coefficients(min_depth, max_depth, depth_dim), persistent=False)

        # Interleave: sin/cos of lat, lon and depth repeat every 5 (10 with a
        # second, borehole depth) positions of the embedding.
        slots = np.arange(emb_dim)
        gather = np.zeros(emb_dim)
        gather[slots % 5 == 0] = np.arange(lat_dim)
        gather[slots % 5 == 1] = lat_dim + np.arange(lat_dim)
        gather[slots % 5 == 2] = 2 * lat_dim + np.arange(lon_dim)
        gather[slots % 5 == 3] = 2 * lat_dim + lon_dim + np.arange(lon_dim)
        if borehole:
            depth_dim *= 2
        gather[slots % 10 == 4] = 2 * lat_dim + 2 * lon_dim + np.arange(depth_dim)
        gather[slots % 10 == 9] = 2 * lat_dim + 2 * lon_dim + depth_dim + np.arange(depth_dim)
        self.register_buffer(
            "gather_index", torch.LongTensor(gather.astype("int64"))[None, None], persistent=False
        )

    def forward(self, x):
        if self.rotation is not None:
            lat_base = x[:, :, 0]
            lon_base = x[:, :, 1] * torch.cos(lat_base * np.pi / 180)
            lat_base = lat_base - self.rotation_anchor[0]
            lon_base = lon_base - self.rotation_anchor[1] * np.cos(self.rotation_anchor[0] * np.pi / 180)
            rotated = torch.stack([lat_base, lon_base], axis=-1) @ self.rotation_matrix
            lat_base = rotated[:, :, 0:1] * self.lat_coeff
            lon_base = rotated[:, :, 1:2] * self.lon_coeff
        else:
            lat_base = x[:, :, 0:1] * self.lat_coeff
            lon_base = x[:, :, 1:2] * self.lon_coeff
        depth_base = x[:, :, 2:3] * self.depth_coeff

        parts = [lat_base, lon_base, depth_base]
        if self.borehole:
            if x.shape[-1] == 3:
                # No borehole column: put the station depth in the second slot.
                parts[2] = depth_base * 0
                parts.append(x[:, :, 2:3] * self.depth_coeff)
            else:
                parts.append(x[:, :, 3:4] * self.depth_coeff)

        output = torch.cat([f(p) for p in parts for f in (torch.sin, torch.cos)], -1)
        return torch.gather(output, -1, self.gather_index.expand(x.shape[0], x.shape[1], self.emb_dim))


class WaveformsEmbedding(nn.Module):
    def __init__(self, input_shape, downsample, mlp_dims):
        super().__init__()
        self.NormalizedScaleEmbedding = NormalizedScaleEmbedding(
            input_shape=input_shape, downsample=downsample, mlp_dims=mlp_dims
        )
        # Same module under a second name: both paths are checkpoint keys.
        self.TimeDistributed = TimeDistributed(self.NormalizedScaleEmbedding)
        self.LayerNorm = LayerNorm(mlp_dims)

    def forward(self, x, recording):
        x = x * torch.unsqueeze(recording, -1)
        return self.LayerNorm(self.TimeDistributed(x))


class TotalEmbedding(nn.Module):
    """One vector per station slot: what it recorded plus where it stands."""

    def __init__(self, input_shape, downsample, mlp_dims, wavelength, borehole=False,
                 rotation=None, rotation_anchor=None, alternative_coords_embedding=False):
        super().__init__()
        self.alternative_coords_embedding = alternative_coords_embedding
        self.Station_emb = PositionEmbedding(
            wavelength=wavelength, emb_dim=mlp_dims[-1], borehole=borehole,
            rotation=rotation, rotation_anchor=rotation_anchor,
        )
        self.WaveformsEmbedding = WaveformsEmbedding(
            input_shape=input_shape, downsample=downsample, mlp_dims=mlp_dims
        )

    def forward(self, waveforms, coords):
        recording = torch.any(torch.any(waveforms != 0, -1), -1).unsqueeze(-1)
        placed = torch.any(coords != 0, -1).unsqueeze(-1)

        embedded = self.WaveformsEmbedding(waveforms, recording)
        position = self.Station_emb(coords)
        if self.alternative_coords_embedding:
            return torch.cat((embedded, position), -1), recording
        return embedded + position * placed, recording


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, total_station, n_heads, mlp_dims, att_dropout, initializer_range,
                 tie_qkv=False, infinity=1e6):
        super().__init__()
        self.n_heads = n_heads
        self.infinity = infinity
        self.d_model = mlp_dims[-1]
        self.stations = total_station
        self.d_key = self.d_model // n_heads
        if self.d_model % n_heads:
            raise ValueError(f"{self.d_model} model dimensions do not split over {n_heads} heads")
        self.dropout = nn.Dropout(p=att_dropout) if att_dropout > 0 else nn.Identity()

        def uniform(shape):
            return torch.distributions.Uniform(-initializer_range, initializer_range).sample(shape)

        projection = (self.d_model, self.d_key * n_heads)
        self.WQ = nn.Parameter(uniform(projection))
        if tie_qkv:
            # Published behaviour: queries, keys and values share one weight
            # matrix.  See README, "Known deviations".
            self.WK = self.WV = self.WQ
        else:
            self.WK = nn.Parameter(uniform(projection))
            self.WV = nn.Parameter(uniform(projection))
        self.WO = nn.Parameter(uniform((self.d_key * n_heads, self.d_model)))

    def _heads(self, x, weight):
        projected = torch.matmul(x, weight)
        projected = torch.reshape(projected, (-1, self.stations, self.d_key, self.n_heads))
        return projected.permute((0, 3, 1, 2))

    def forward(self, x, recording):
        q = self._heads(x, self.WQ)
        k = self._heads(x, self.WK).permute((0, 1, 3, 2))
        v = self._heads(x, self.WV)

        score = torch.matmul(q, k) / np.sqrt(self.d_key)
        # Stations with no waveform are removed as attention targets.
        silent = torch.unsqueeze(torch.logical_not(recording), -1).permute((0, 2, 3, 1))
        score = F.softmax(score - silent * self.infinity, dim=-1)
        score = self.dropout(score)

        o = torch.matmul(score, v).permute((0, 2, 1, 3))
        o = torch.reshape(o, (-1, self.stations, self.n_heads * self.d_key))
        return torch.matmul(o, self.WO)


class PointwiseFeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.kernel1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(emb_dim, hidden_dim), gain=1))
        self.kernel2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(hidden_dim, emb_dim), gain=1))
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        self.bias2 = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        x = F.gelu(torch.matmul(x, self.kernel1) + self.bias1)
        return torch.matmul(x, self.kernel2) + self.bias2


class TransformerLayer(nn.Module):
    def __init__(self, total_station, mlp_dims, n_heads, attention_dropout, initializer_range,
                 ffn_hidden_dim, hidden_dropout, tie_qkv):
        super().__init__()
        self.MultiHeadSelfAttention = MultiHeadSelfAttention(
            total_station=total_station, mlp_dims=mlp_dims, n_heads=n_heads,
            att_dropout=attention_dropout, initializer_range=initializer_range, tie_qkv=tie_qkv,
        )
        self.PointwiseFeedForward = PointwiseFeedForward(
            emb_dim=mlp_dims[-1], hidden_dim=ffn_hidden_dim
        )
        self.LayerNorm1 = LayerNorm(input_shape=mlp_dims)
        self.LayerNorm2 = LayerNorm(input_shape=mlp_dims)
        self.dropout = nn.Dropout(hidden_dropout) if hidden_dropout > 0 else nn.Identity()

    def forward(self, x, recording):
        x = self.LayerNorm1(x + self.dropout(self.MultiHeadSelfAttention(x, recording)))
        return self.LayerNorm2(x + self.dropout(self.PointwiseFeedForward(x)))


class Transformer(nn.Module):
    def __init__(self, layers, **layer_params):
        super().__init__()
        prototype = TransformerLayer(**layer_params)
        self.layers = nn.ModuleList([copy.deepcopy(prototype) for _ in range(layers)])

    def forward(self, x, recording):
        for layer in self.layers:
            x = layer(x, recording)
        return x


class To_GaussianDistribution(nn.Module):
    def __init__(self, emb_dim, output_mlp_dims, pga_mixture):
        super().__init__()
        self.MLP = MLP((emb_dim,), output_mlp_dims)
        self.TimeDistributed1 = TimeDistributed(self.MLP)
        self.MixtureOutput = MixtureOutput((output_mlp_dims[-1],), pga_mixture, bias_mu=-5, bias_sigma=1)
        self.TimeDistributed2 = TimeDistributed(self.MixtureOutput)

    def forward(self, x):
        return self.TimeDistributed2(self.TimeDistributed1(x))


class FullModel(nn.Module):
    """Waveforms and coordinates in, a PGA distribution per station slot out.

    Shapes: ``(batch, slots, samples, channels)`` and ``(batch, slots, 4)`` in,
    ``(batch, slots, mixture, 3)`` out, where ``slots`` is ``stations`` during
    pre-training and ``2 * stations`` once an event is retrieved.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        input_shape = (config.trace_length, config.channels)
        self.TotalEmbedding = TotalEmbedding(
            input_shape=input_shape,
            downsample=config.downsample,
            mlp_dims=config.mlp_dims,
            wavelength=config.wavelength,
            borehole=config.borehole,
            rotation=config.rotation,
            rotation_anchor=config.rotation_anchor,
            alternative_coords_embedding=config.alternative_coords_embedding,
        )
        self.Transformer = Transformer(
            layers=config.transformer_layers,
            total_station=config.total_stations,
            mlp_dims=config.mlp_dims,
            n_heads=config.n_heads,
            attention_dropout=config.attention_dropout,
            initializer_range=config.initializer_range,
            ffn_hidden_dim=config.ffn_hidden_dim,
            hidden_dropout=config.hidden_dropout,
            tie_qkv=config.tie_qkv,
        )
        self.To_GaussianDistribution = To_GaussianDistribution(
            emb_dim=config.mlp_dims[-1],
            output_mlp_dims=config.output_mlp_dims,
            pga_mixture=config.pga_mixture,
        )

    def forward(self, waveforms, coords):
        x, recording = self.TotalEmbedding(waveforms, coords)
        x = self.Transformer(x, recording)
        return self.To_GaussianDistribution(x)

    def predict_current(self, waveforms, coords):
        """Predictions for the current event only, dropping retrieved slots."""
        return self(waveforms, coords)[:, : self.config.stations]
