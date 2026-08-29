# freetoken-ox-boost

GLM-5.3-Flash adaptation + single-GPU offload performance patch set for
[FreeToken](https://github.com/FlashML-org/FreeToken) v0.1.2, shipped as a
patch plugin (no upstream code is redistributed).

Running GLM-5.3-Flash-NVFP4 (181 GB of expert weights, host-memory offload,
PCIe Gen5 x16) on a single RTX PRO 6000 Blackwell 96 GB: single-stream decode
went from **17.8 to 35 tok/s (~2x)**, 2-way concurrent aggregate 44.9
(warm topics), 4-way aggregate ~49. Zero quality loss; 48/48 steps match the
HF reference implementation token for token.

Why concurrency does not scale here: GLM-5.3-Flash routing is extremely flat
(the 8 experts each token activates barely overlap between tokens), so
concurrent requests' expert-fetch bytes add up linearly per stream while also
competing for the cache — and the PCIe link is already saturated. Batching
cannot amortize the one thing this machine actually bills for: bytes.

## Measured metrics

Speed is sensitive to context length and cache warmth; each row states its
measurement conditions.

| Metric | Measured | Conditions |
|---|---|---|
| Single-stream decode | **35.0 tok/s** (33.8-36.3) | 2-slot 256K pool, warm cache |
| 2-way concurrent aggregate | 44.9 tok/s (warm) / 36-37 (sustained cold topics) | 22.4 / ~18 per stream |
| 4-way concurrent aggregate | ~49 tok/s | ~12 per stream |
| MoE cache hit rate | **~84%** (miss 0.157) | 8 resident layers + P=4 prefetch @ 2079 slots; now 2600 slots, slightly better |
| TTFT (10-token prompt) | **0.59 s** | on-demand prefill path (whole-layer streaming is flat ~2.1 s) |
| TTFT (40 tokens) | 0.95 s | same |
| TTFT (2.7K tokens) | 3.12 s (prefill 852 tok/s) | streaming prefill |
| 232K long-context prefill | 232 s (~1030 tok/s) | needles at 10%/50%/95% all recalled |
| Decode after 232K context | 25.1 tok/s | measured on the 1-slot 256K config, near-lossless |

See `examples/serve_full.sh` for a launch reference (adjust paths and
configuration for your machine).

## Layout

```
patches/    39 per-file unified diffs (against v0.1.2, git apply -p1)
overlay/    17 new files (the models/glm5_next directory + 4 triton kernels + prefetch/LFU + gpu_select)
install.sh  version check -> dry-run -> apply -> compileall
examples/   production launch script (2-slot 256K, all optimizations on by default)
MANIFEST.md feature -> files -> switches -> measured numbers
```

## License

Apache-2.0, same as upstream. This repository contains only our patches and
new files, never the upstream code itself.
