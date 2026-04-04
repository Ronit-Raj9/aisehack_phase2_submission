import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


def _to_b_t_c_h_w(x: torch.Tensor) -> torch.Tensor:
	if x.ndim != 5:
		raise ValueError(f"Expected 5D tensor, got shape {tuple(x.shape)}")

	# Supported layouts:
	# (B, T, C, H, W)
	# (B, T, H, W, C)
	if x.shape[2] <= 64 and x.shape[3] > 64 and x.shape[4] > 64:
		return x
	if x.shape[-1] <= 64 and x.shape[2] > 64 and x.shape[3] > 64:
		return x.permute(0, 1, 4, 2, 3)
	raise ValueError(
		f"Could not infer input layout from shape {tuple(x.shape)}. "
		"Use (B,T,C,H,W) or (B,T,H,W,C)."
	)


def _safe_feature_index(feature_names: Sequence[str]) -> Dict[str, int]:
	mapping = {name: idx for idx, name in enumerate(feature_names)}
	required = [
		"cpm25",
		"q2",
		"t2",
		"u10",
		"v10",
		"swdown",
		"pblh",
		"psfc",
		"rain",
		"PM25",
		"NH3",
		"SO2",
		"NOx",
		"NMVOC_e",
		"NMVOC_finn",
		"bio",
	]
	missing = [name for name in required if name not in mapping]
	if missing:
		raise KeyError(f"Missing required feature names: {missing}")
	return mapping


def _spatial_divergence(u10: torch.Tensor, v10: torch.Tensor) -> torch.Tensor:
	# u10, v10: (B, T, H, W)
	if u10.ndim != 4 or v10.ndim != 4:
		raise ValueError(f"Expected u10,v10 shape (B,T,H,W), got {tuple(u10.shape)} and {tuple(v10.shape)}")
	du_dx = F.pad(u10[..., :, 1:] - u10[..., :, :-1], (0, 1, 0, 0), mode="replicate")
	dv_dy = F.pad(v10[..., 1:, :] - v10[..., :-1, :], (0, 0, 0, 1), mode="replicate")
	return du_dx + dv_dy


class SEBlock(nn.Module):
	def __init__(self, channels: int, reduction: int = 8):
		super().__init__()
		hidden = max(4, channels // reduction)
		self.fc1 = nn.Conv2d(channels, hidden, 1)
		self.fc2 = nn.Conv2d(hidden, channels, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		w = F.adaptive_avg_pool2d(x, 1)
		w = F.gelu(self.fc1(w))
		w = torch.sigmoid(self.fc2(w))
		return x * w


class ChannelAttention(nn.Module):
	def __init__(self, channels: int, reduction: int = 8):
		super().__init__()
		hidden = max(4, channels // reduction)
		self.mlp = nn.Sequential(
			nn.Conv2d(channels, hidden, 1, bias=False),
			nn.GELU(),
			nn.Conv2d(hidden, channels, 1, bias=False),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
		mx = self.mlp(F.adaptive_max_pool2d(x, 1))
		w = torch.sigmoid(avg + mx)
		return x * w


class SpatialAttention(nn.Module):
	def __init__(self, kernel_size: int = 7):
		super().__init__()
		padding = kernel_size // 2
		self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		avg = torch.mean(x, dim=1, keepdim=True)
		mx, _ = torch.max(x, dim=1, keepdim=True)
		w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
		return x * w


class CBAM(nn.Module):
	def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
		super().__init__()
		self.ca = ChannelAttention(channels, reduction)
		self.sa = SpatialAttention(spatial_kernel)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.ca(x)
		x = self.sa(x)
		return x


class SpectralConv2d(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
		super().__init__()
		self.modes1 = modes1
		self.modes2 = modes2
		scale = 1.0 / math.sqrt(in_channels * out_channels)
		self.weights = nn.Parameter(
			scale * torch.randn(in_channels, out_channels, modes1, modes2, 2, dtype=torch.float32)
		)

	@staticmethod
	def _as_complex_weight_4d(weight: torch.Tensor) -> torch.Tensor:
		# Accept current canonical shape: (Cin, Cout, M1, M2) complex
		if weight.ndim == 4 and torch.is_complex(weight):
			return weight

		# Preferred internal storage: (Cin, Cout, M1, M2, 2) real/imag
		if weight.ndim == 5 and weight.shape[-1] == 2:
			return torch.view_as_complex(weight.contiguous())

		# Legacy packed formats occasionally found in older checkpoints/code:
		# 1) (2, Cin, Cout, M1, M2) where [0]=real, [1]=imag
		# 2) (Cin, Cout, M1, M2, 2) where [...,0]=real, [...,1]=imag
		if weight.ndim == 5:
			if weight.shape[0] == 2:
				return torch.complex(weight[0], weight[1])

		raise ValueError(
			"Unsupported spectral weight shape/dtype. Expected complex (Cin,Cout,M1,M2) "
			f"or legacy packed 5D real/imag format, got shape={tuple(weight.shape)} dtype={weight.dtype}."
		)

	@staticmethod
	def compl_mul2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
		if x.ndim != 4 or w.ndim != 4:
			raise ValueError(f"Expected x,w to be 4D for spectral multiply, got {tuple(x.shape)} and {tuple(w.shape)}")
		return torch.einsum("bixy,ioxy->boxy", x, w)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		b, _, h, w = x.shape
		orig_dtype = x.dtype

		# FFT kernels on T4 do not support fp16 for these spatial sizes.
		# Keep the spectral branch in fp32 even under AMP, then cast the result back.
		x = x.float()
		weights = self._as_complex_weight_4d(self.weights).to(torch.complex64)
		x_ft = torch.fft.rfft2(x, dim=(-2, -1))
		out_ft = torch.zeros(
			b,
			weights.shape[1],
			h,
			w // 2 + 1,
			dtype=torch.complex64,
			device=x.device,
		)
		m1 = min(self.modes1, h)
		m2 = min(self.modes2, w // 2 + 1)
		out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], weights[:, :, :m1, :m2])
		out = torch.fft.irfft2(out_ft, s=(h, w))
		return out.to(orig_dtype)


class FNOBlock(nn.Module):
	def __init__(self, width: int, modes1: int, modes2: int):
		super().__init__()
		self.spec = SpectralConv2d(width, width, modes1, modes2)
		self.pw = nn.Conv2d(width, width, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return F.gelu(self.spec(x) + self.pw(x))


class ConvLSTMCell(nn.Module):
	def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 5):
		super().__init__()
		padding = kernel_size // 2
		self.hidden_channels = hidden_channels
		self.conv = nn.Conv2d(
			input_channels + hidden_channels,
			4 * hidden_channels,
			kernel_size=kernel_size,
			padding=padding,
			bias=True,
		)

	def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		combined = torch.cat([x_t, h_prev], dim=1)
		gates = self.conv(combined)
		i, f, o, g = torch.chunk(gates, 4, dim=1)
		i = torch.sigmoid(i)
		f = torch.sigmoid(f)
		o = torch.sigmoid(o)
		g = torch.tanh(g)
		c = f * c_prev + i * g
		h = o * torch.tanh(c)
		return h, c


class ConvLSTMStack(nn.Module):
	def __init__(self, input_channels: int, hidden_channels: int = 64, n_layers: int = 3, kernel_size: int = 5):
		super().__init__()
		cells = []
		for i in range(n_layers):
			in_ch = input_channels if i == 0 else hidden_channels
			cells.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size=kernel_size))
		self.cells = nn.ModuleList(cells)
		self.hidden_channels = hidden_channels

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, T, C, H, W)
		b, t, _, h, w = x.shape
		hs = [torch.zeros(b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype) for _ in self.cells]
		cs = [torch.zeros(b, self.hidden_channels, h, w, device=x.device, dtype=x.dtype) for _ in self.cells]

		for ti in range(t):
			x_t = x[:, ti]
			for li, cell in enumerate(self.cells):
				h_prev, c_prev = hs[li], cs[li]
				h_new, c_new = cell(x_t, h_prev, c_prev)
				hs[li], cs[li] = h_new, c_new
				x_t = h_new
		return hs[-1]


class HybridFNOConvLSTMEnsemble(nn.Module):
	def __init__(
		self,
		feature_names: Sequence[str],
		time_in: int = 10,
		time_out: int = 16,
		width: int = 64,
		modes1: int = 24,
		modes2: int = 24,
		convlstm_hidden: int = 64,
	):
		super().__init__()
		self.feature_names = list(feature_names)
		self.feature_idx = _safe_feature_index(self.feature_names)
		self.time_in = time_in
		self.time_out = time_out
		self.width = width

		self.static_names = ["NMVOC_e", "SO2", "NH3", "NOx", "PM25", "NMVOC_finn", "bio", "psfc"]
		self.dynamic_names = ["cpm25", "u10", "v10", "q2", "t2", "pblh", "rain", "swdown"]

		dynamic_channels = len(self.dynamic_names) + 2 + 2  # + div_wind + pblh_inv_emit + positional grid
		static_channels = len(self.static_names)
		season_channels = 4
		all_channels_per_t = dynamic_channels + static_channels + season_channels

		self.in_lift = nn.Conv2d(time_in * all_channels_per_t + 2, width, 1)
		self.se_lift = SEBlock(width, reduction=8)

		self.convlstm = ConvLSTMStack(dynamic_channels, hidden_channels=convlstm_hidden, n_layers=3, kernel_size=5)

		self.fusion_conv = nn.Conv2d(width + convlstm_hidden, width, 3, padding=1)
		self.clstm_gate_proj = nn.Conv2d(width, convlstm_hidden, 1)
		self.fno_from_clstm = nn.Conv2d(convlstm_hidden, width, 1)
		self.clstm_from_fno = nn.Conv2d(width, convlstm_hidden, 1)

		self.fno_block1 = FNOBlock(width, modes1, modes2)
		self.fno_block2 = FNOBlock(width, modes1, modes2)
		self.fno_block3 = FNOBlock(width, modes1, modes2)

		self.cbam_h2 = CBAM(width, reduction=8, spatial_kernel=7)

		self.dec1 = nn.Conv2d(width * 2, width, 3, padding=1)
		self.dec2 = nn.Conv2d(width * 2, width, 3, padding=1)
		self.cbam_dec = CBAM(width, reduction=8, spatial_kernel=7)
		self.fno_out = nn.Conv2d(width, time_out, 1)

		self.clstm_dec1 = nn.Conv2d(convlstm_hidden, 128, 3, padding=1)
		self.clstm_sa = SpatialAttention(kernel_size=7)
		self.clstm_out = nn.Conv2d(128, time_out, 1)

		self.ep_head = nn.Sequential(
			nn.Conv2d(width, 32, 3, padding=1),
			nn.GELU(),
			nn.Conv2d(32, 1, 1),
		)

	@staticmethod
	def _grid(b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
		gx = torch.linspace(0, 1, h, device=device, dtype=dtype).view(1, 1, h, 1).repeat(b, 1, 1, w)
		gy = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w).repeat(b, 1, h, 1)
		return torch.cat([gx, gy], dim=1)

	def _gather(self, x: torch.Tensor, names: Sequence[str]) -> torch.Tensor:
		idxs = [self.feature_idx[n] for n in names]
		return x[:, :, idxs]

	def _season_onehot(self, batch: int, season_idx: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
		if season_idx is None:
			oh = torch.zeros(batch, 4, device=device, dtype=dtype)
		else:
			if season_idx.ndim == 0:
				season_idx = season_idx.expand(batch)
			oh = F.one_hot(season_idx.long().clamp(min=0, max=3), num_classes=4).to(dtype=dtype)
		return oh

	def _build_features(
		self,
		x_b_t_c_h_w: torch.Tensor,
		season_idx: Optional[torch.Tensor],
	) -> Tuple[torch.Tensor, torch.Tensor]:
		# Returns:
		# all_for_fno: (B, T, C_all, H, W)
		# dynamic_for_clstm: (B, T, C_dyn, H, W)
		b, t, _, h, w = x_b_t_c_h_w.shape

		dyn = self._gather(x_b_t_c_h_w, self.dynamic_names)  # (B,T,7,H,W)
		u10 = x_b_t_c_h_w[:, :, self.feature_idx["u10"]]
		v10 = x_b_t_c_h_w[:, :, self.feature_idx["v10"]]
		pblh = x_b_t_c_h_w[:, :, self.feature_idx["pblh"]]
		pm25_emit = x_b_t_c_h_w[:, :, self.feature_idx["PM25"]]

		div_wind = _spatial_divergence(u10, v10).unsqueeze(2)
		pblh_inv_emit = (pm25_emit / torch.clamp(pblh, min=EPS)).unsqueeze(2)
		# Add positional grid channels for ConvLSTM branch
		grid = self._grid(b, h, w, x_b_t_c_h_w.device, x_b_t_c_h_w.dtype)
		grid_t = grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
		dynamic_for_clstm = torch.cat([dyn, div_wind, pblh_inv_emit, grid_t], dim=2)

		static = self._gather(x_b_t_c_h_w, self.static_names).mean(dim=1)  # (B,8,H,W)
		static_tiled = static.unsqueeze(1).repeat(1, t, 1, 1, 1)

		season_oh = self._season_onehot(b, season_idx, x_b_t_c_h_w.device, x_b_t_c_h_w.dtype)
		season_map = season_oh[:, None, :, None, None].repeat(1, t, 1, h, w)

		all_for_fno = torch.cat([dynamic_for_clstm, static_tiled, season_map], dim=2)
		return all_for_fno, dynamic_for_clstm

	def forward(
		self,
		x: torch.Tensor,
		season_idx: Optional[torch.Tensor] = None,
		return_aux: bool = False,
	):
		x = _to_b_t_c_h_w(x)
		b, t, _, h, w = x.shape
		if t != self.time_in:
			raise ValueError(f"Expected time_in={self.time_in}, got {t}")

		x_all, x_dyn = self._build_features(x, season_idx)

		# FNO input flatten across time
		x_fno = x_all.permute(0, 1, 3, 4, 2).reshape(b, t * x_all.shape[2], h, w)
		x_fno = torch.cat([x_fno, self._grid(b, h, w, x.device, x.dtype)], dim=1)
		x_fno = self.in_lift(x_fno)
		x_fno = self.se_lift(x_fno)

		# ConvLSTM branch
		h_last = self.convlstm(x_dyn)

		# Cross-branch interaction / gating
		fusion = torch.cat([x_fno, h_last], dim=1)
		gate = torch.sigmoid(self.fusion_conv(fusion))
		clstm_gate = torch.sigmoid(self.clstm_gate_proj(gate))
		fno_gated = x_fno * gate
		clstm_gated = h_last * (1.0 - clstm_gate)

		x_fno = fno_gated + self.fno_from_clstm(clstm_gated)
		h_ctx = clstm_gated + self.clstm_from_fno(fno_gated)
		h_ctx = h_ctx.contiguous()

		# FNO-UNet branch
		h0 = self.fno_block1(x_fno)
		h1 = self.fno_block2(h0)
		h2 = self.fno_block3(h1)
		h2_attn = self.cbam_h2(h2)

		ep_prob = torch.sigmoid(self.ep_head(h2_attn))  # (B,1,H,W)

		d1 = F.gelu(self.dec1(torch.cat([h2_attn, h1], dim=1)))
		d2 = F.gelu(self.dec2(torch.cat([d1, h0], dim=1)))
		d2 = self.cbam_dec(d2)
		pred_fno = self.fno_out(d2)  # (B,T_out,H,W)

		# ConvLSTM decoder
		c = self.clstm_dec1(h_ctx)
		c = F.gelu(c)
		c = c.contiguous()
		c = self.clstm_sa(c)
		pred_clstm = self.clstm_out(c)  # (B,T_out,H,W)

		ep = ep_prob.repeat(1, self.time_out, 1, 1)
		pred = (1.0 - ep) * pred_fno + ep * pred_clstm
		pred = pred.permute(0, 2, 3, 1)  # (B,H,W,T_out)

		if return_aux:
			return pred, {
				"pred_fno": pred_fno,
				"pred_clstm": pred_clstm,
				"ep_prob": ep_prob,
			}
		return pred


def smape_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	return (2.0 * torch.abs(pred - target) / (torch.abs(pred) + torch.abs(target) + eps)).mean()


def episode_weighted_smape_loss(
	pred: torch.Tensor,
	target: torch.Tensor,
	episode_mask: torch.Tensor,
	alpha: float = 5.0,
	eps: float = 1e-8,
) -> torch.Tensor:
	# pred/target expected (B,H,W,T) or (B,T,H,W)
	if pred.shape != target.shape:
		raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")

	# Normalize mask layout to match pred/target
	if episode_mask.ndim == 2:
		episode_mask = episode_mask.unsqueeze(0).unsqueeze(-1).expand(pred.shape[0], -1, -1, pred.shape[-1])
	elif episode_mask.ndim == 3:
		episode_mask = episode_mask.unsqueeze(-1).expand(-1, -1, -1, pred.shape[-1])
	elif episode_mask.ndim == 4 and episode_mask.shape[1] == 1:
		# (B,1,H,W) -> (B,H,W,1) -> expand over horizon
		episode_mask = episode_mask.permute(0, 2, 3, 1).expand(-1, -1, -1, pred.shape[-1])
	elif episode_mask.ndim == 4 and episode_mask.shape[-1] == pred.shape[-1]:
		pass
	else:
		raise ValueError(
			"episode_mask must be one of (H,W), (B,H,W), (B,1,H,W), (B,H,W,T). "
			f"Got {tuple(episode_mask.shape)}"
		)

	sm = 2.0 * torch.abs(pred - target) / (torch.abs(pred) + torch.abs(target) + eps)
	w = 1.0 + (alpha - 1.0) * episode_mask.float()
	return (sm * w).mean()


def hybrid_total_loss(
	pred: torch.Tensor,
	target: torch.Tensor,
	ep_prob: torch.Tensor,
	episode_mask: torch.Tensor,
	alpha: float = 5.0,
	ep_head_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
	# Main loss on (B,H,W,T)
	main = episode_weighted_smape_loss(pred, target, episode_mask, alpha=alpha)

	# Convert episode mask to 2D probability target for episode head: any episode over horizon
	if episode_mask.ndim == 2:
		ep_target_2d = episode_mask.float().unsqueeze(0).unsqueeze(0).expand(pred.shape[0], 1, -1, -1)
	elif episode_mask.ndim == 3:
		ep_target_2d = episode_mask.float().unsqueeze(1)
	elif episode_mask.ndim == 4 and episode_mask.shape[-1] == target.shape[-1]:
		ep_target_2d = episode_mask.float().amax(dim=-1, keepdim=True).permute(0, 3, 1, 2)
	elif episode_mask.ndim == 4 and episode_mask.shape[1] == 1:
		ep_target_2d = episode_mask.float()
	else:
		raise ValueError(
			"episode_mask must be one of (H,W), (B,H,W), (B,H,W,T), (B,1,H,W). "
			f"Got {tuple(episode_mask.shape)}"
		)

	# BCE is unsafe under autocast on this setup; force fp32 for this term.
	with torch.amp.autocast(device_type="cuda", enabled=False):
		ep_bce = F.binary_cross_entropy(ep_prob.float(), ep_target_2d.float())
	total = main + ep_head_weight * ep_bce
	return {"total": total, "main": main, "ep_bce": ep_bce}

