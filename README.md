# freetoken-ox-boost

GLM-5.3-Flash adaptation + single-GPU offload performance patch set for
[FreeToken](https://github.com/FlashML-org/FreeToken) v0.1.2, shipped as a
patch plugin (no upstream code is redistributed).

Running GLM-5.3-Flash-NVFP4 (181 GB of expert weights, host-memory offload,
PCIe Gen5 x16) on a single RTX PRO 6000 Blackwell 96 GB: single-stream decode
went from **17.8 to 36.7 tok/s (~2x)**, 2-way concurrent aggregate 44.9
(warm topics), 4-way aggregate ~49. Zero quality loss; 48/48 steps match the
HF reference implementation token for token. Vision (image input) is supported
end-to-end — see below.

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
| Single-stream decode | **36.7 tok/s** | 1-slot 256K pool (current default), warm cache |
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

## Vision support (multimodal GLM-5.3-Flash)

The checkpoint's full `Glm5NextVisionModel` tower (0.6B ViT: Conv3d patch
embed, 24 attention blocks, 2x2 spatial-merge downsample, clamped-swiglu
merger) is ported and wired end-to-end through the serving stack:

- `overlay: models/glm5_next/vision.py` — faithful eager port, validated
  stage-by-stage against the HF reference on the real weights: patch embed and
  rotary tables bitwise-equal; residual per-layer drift is within the BF16
  kernel-noise envelope measured between HF's own sdpa and eager backends.
- `overlay: models/glm5_next/image_process.py` — vendored preprocessing
  (smart resize -> bicubic -> pad -> CLIP normalize -> patchify), bitwise-equal
  `pixel_values` vs HF `Glm5NextImageProcessorPil` on test images (the runtime
  env pins transformers 5.15.x, which has no glm5_next classes). Needs `pillow`.
- Intake patches (`server_* / tokenizer_* / message_* / scheduler_scheduler`) —
  images accepted on **both APIs** (OpenAI `image_url` data URI or http URL,
  Anthropic base64 `image` blocks), chat-template placeholder expansion in the
  tokenize worker, pixel transport over the ZMQ msgpack codec, and vision
  encoding at request admission inside the scheduler process (which owns the
  GPU).

Enable with `FREETOKEN_GLM5_VISION=1` (default off; the BF16 tower costs
~1.2 GB VRAM, text-only behavior is byte-identical when off). Multi-image
prompts work; an image prompt must fit in a single prefill chunk. Verified
end-to-end on the production server: shape/color/position description, object
counting, two-image comparison, and text-in-image reading through both
endpoints.

**Video input** is supported on the OpenAI endpoint (`{"type": "video_url"}`
content parts, data URI or http URL; decoded via PyAV): frames are sampled at
`FREETOKEN_GLM5_VIDEO_FPS` (2), paired into temporal patches, and expanded
into per-second frame blocks with timestamps exactly like the HF processor.
Budgets: `FREETOKEN_GLM5_MAX_VIDEO_TOKENS` (8000 patches for the whole clip),
`FREETOKEN_GLM5_MAX_VIDEO_FRAMES` (64). Verified: motion direction,
start/end positions, and colors on a synthetic clip.

**Prefix caching works for image and video requests** (this patch set makes it
content-safe): each image span's placeholder ids are replaced by a per-image
pixel-content hash in the radix cache keys, so identical text+media prefixes
hit while different images diverge at the span's first token. Measured on the
production box: repeating the same image request drops TTFT 3.8 -> 1.0 s;
different images never false-hit; a follow-up turn in the same image
conversation drops 2.8 -> 1.1 s (with `--cache-type radix` and
`FREETOKEN_PREFILL_ONDEMAND_TOKENS=128`, both now the example defaults).
Vision-specific overhead is otherwise negligible: a 220-token image request
has the same TTFT as a 220-token text prompt (tower + preprocessing < 20 ms;
the vendored preprocessor is 14 ms/image).

## Layout

```
patches/    53 per-file unified diffs (against v0.1.2, git apply -p1)
overlay/    19 new files (the models/glm5_next directory incl. vision + 4 triton kernels + prefetch/LFU + gpu_select)
install.sh  version check -> dry-run -> apply -> compileall
examples/   production launch script (1-slot 256K, all optimizations on by default)
MANIFEST.md feature -> files -> switches -> measured numbers
```

## License

Apache-2.0, same as upstream. This repository contains only our patches and
new files, never the upstream code itself.
