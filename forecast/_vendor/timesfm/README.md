# Private TimesFM 2.5 inference source

Copied from `E:/pycharmproject/epf/TF/src/timesfm` on 2026-10-30.
Upstream project: https://github.com/google-research/timesfm (Google LLC, Apache-2.0).
`UPSTREAM.json` records SHA-256 hashes of the exact original files; the run's
`metadata.json/code_sha256` records hashes of the adapted Python files.
The included LICENSE is the standard Apache-2.0 license text. Original copyright
headers are preserved. No weights are included.

Local modifications: retain PyTorch univariate inference only; remove XReg entry
point and remote hub loading; replace import-time shared model allocation with
instance construction; select an explicit device after preflight; load local
safetensors without torch.compile. The numerical architecture and decode logic
are unchanged. No public `timesfm` import or global sys.path override is used.

TimesFM1 update: restored the upstream `forecast_with_covariates` method and
`utils/xreg_lib.py` from the same local source. XReg is imported lazily, and its
original source hash is included in `UPSTREAM.json`. The pure-price adapter keeps
`return_backcast=False`; the new XReg adapter uses True. Local-only loading and
instance-owned torch models remain unchanged.
