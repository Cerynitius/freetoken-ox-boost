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

## Vision, video and caching

The checkpoint's 0.6B ViT tower is ported and wired end to end. Images work on both the OpenAI and Anthropic-style endpoints, video on the OpenAI endpoint via PyAV at 2 fps by default. Enable with FREETOKEN_GLM5_VISION=1, off by default, costs about 1.2 GB VRAM, and the text-only path is byte-identical when off. Preprocessing is bitwise-equal to the HF image processor and tower drift stays within the BF16 noise between HF's own sdpa and eager backends. Verified on the production server for description, counting, two-image comparison, text-in-image, and motion on video. A 220-token image request has the same TTFT as a 220-token text prompt.

Prefix caching works for media requests. Image placeholder ids are replaced by a pixel-content hash in the radix cache key, so identical text plus media prefixes hit and different images never false-hit. Repeating an image request drops TTFT from 3.8 to 1.0 s, and a follow-up turn in the same image conversation from 2.8 to 1.1 s. Under production-shaped load, multi-turn agent conversations with interleaved images hold about 1 s TTFT per turn, shared system prompts and long documents hit across conversations, and LRU eviction past the 262K pool keeps the newest entries. One known edge: a re-query landing within one decode step of an identical request can miss the not-yet-inserted prefix and pay one cold prefill.

**Known issue (2026-08-30, under investigation):** on this hybrid (KDA) model,
radix cache-HIT requests co-batched with in-flight decode corrupt ~8% of the
time under concurrent load — blank visual grounding, empty or garbled answers,
text and image hits alike. Cold prefill is clean, and sequential single-stream
use showed no corruption in extended testing. The trigger is the upstream
hybrid hit path under concurrency, not the media keying (reproduced with
pure-text prompts; forensics confirm payloads and keys correct on every
failure). The example default is `naive` until this is fixed; opt in with
`GLM5_CACHE_TYPE=radix` for single-stream use.

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
