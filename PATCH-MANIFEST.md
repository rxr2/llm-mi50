# PATCH MANIFEST — MI50 / gfx906 / Qwen3.8-27B Q4_0

Base repo: `https://github.com/ggml-org/llama.cpp` @ **`42916d83f4a225e56709f873aa8050ac11f5b6a4`** (master, 2026-09-2x)
Reference fork: `alex4300/llama.cpp-gfx906-opt` branch `gfx906` @ **`f9616ce7212cf6eceaa7cacf33e13fdd7b20d38f`**
(merge-base with upstream: `e107984b`, 2026-09-03; 47 fork commits, upstream is 351 commits ahead)

Patch series: `patches/000N-*.patch` (`git format-patch upstream/master..HEAD`), applied with `git am`.
Result SHA after applying 0001–0003 on 42916d83: **`844e42b4b`** (built locally, see BUILD-VERIFICATION.md).

Every patch can be turned off (env var or CMake option). "Stage" = the stage where it becomes ACTIVE by default.

| ID | PATCH | SOURCE REPO | SHA (orig.) | FILES | DEPS | CONFLICTS | BENEFIT (measured, source) | RISK | STAGE | TOGGLE |
|---|---|---|---|---|---|---|---|---|---|---|
| P00 | upstream base | ggml-org/llama.cpp | 42916d83 | — | — | — | contains #27841 GCN-MMQ, #28475 mmid race fix, #28846 BF16→F32, pos0 API, qwen4exp fusions | new upstream code not tested on gfx906 by anyone in this chain | 0 | — |
| P01 | wide GEMV Q4_0 n=1..8 (y in LDS, dp4a) + breit2 (n=5) | alex4300 | c0593614, 28170aaf | mmvq.cu | P03 (dispatch) | none on upstream (new code) | Q4_0 4096×14336: n1 77→54 µs, n4 100→60, n8 181→104 (alex, 225 W) | occupancy/VGPR under a different compiler (see BUILD-VERIFICATION §3) | 1 | `GGML_MMVQ_Q4_BREIT_GFX906=0`, `GGML_MMVQ_Q4_BREIT1_GFX906=0` (n=1 only) |
| P02 | wide GEMV Q5_K / Q6_K / Q4_1 n=2..8 | alex4300 | 0f27b7cd | mmvq.cu, test-backend-ops | P03 | none | Q5_K n8 884→135 µs (ssm_out, 48 tensors); Q6_K n8 341→162 (output head); Q4_1 n8 170→101 (8× ffn_down) | same as P01 | 1 | `GGML_MMVQ_Q5K_BREIT_GFX906=0`, `..._Q6K_...=0`, `..._Q41_...=0` |
| P03 | VEGA20 MMVQ/MMQ cutover + rows/block per type | alex4300 | b19fed0a | mmvq.cu | — | **textual** conflict with upstream Orin/Volta table → resolved "keep both" | Q4_K → MMQ from n=6 (MMVQ 419–538 µs vs MMQ 411) | only affects VEGA20 | 1 | code (policy table, see CUTOVER.md) |
| P04 | gfx906 n=1 GEMVs Q4_K/Q5_K/Q6_K (+fused gate/GLU), SWIGLU_CLAMP fix | alex4300 | 4d161380 (part), 72049338 | mmvq.cu, vecdotq.cuh | — | **textual** conflict vecdotq.cuh with upstream #26705 (Q4_K scales, branchless) → **upstream version taken** | K-quants only; **not used by Q4_0 model except Q5_K ssm_out n=1 / Q6_K output n=1** | small | 1 | `GGML_MMVQ_Q4K_GFX906=0`, `..._Q5K_...=0`, `..._Q6K_...=0` |
| P05 | MMQ profile gcn5 (alex) vs upstream #27841 GCN profile | alex4300 / ggml-org | 4d161380 (part) / c8edceb0 | mmq.cuh, mmq-config-gcn5.cuh, mmq-load-tiles.cuh, mmq-vec-dot.cuh | — | **SILENT CONFLICT**: upstream's `GCN` branch comes first → alex gcn5 becomes dead code. Fixed by new switch | alex: Q4_0/Q4_K I=128 (I=64 costs +2.7%/+11.9%); upstream Q4_0 I=64. Unknown which wins on 42916d83 → **A/B mandatory** | prefill only | 1 (=alex, default) / A/B | CMake `-DGGML_MMQ_GCN_PROFILE=1\|0` (two builds) |
| P06 | fattn-tile head-256 gfx906 values | alex4300 | c583c038 | fattn-tile.cuh | — | none (upstream did not touch it since base) | per layer n=1 207→191 µs, n=8 451→350; prefill batch 23.8→21.6 ms | small | 1 | code only (revert patch hunk) |
| P07 | DPP warp reductions GCN (float/int sum, max) | y-morgunov/furnace (sixvolts) | 32f28424 | common.cuh, ggml-hip/CMakeLists.txt | — | SAME idea as PR #26466 → take **only one** (this one). Local fix: `=&v` early-clobber on xor4 | tg +2.9…4.7% on other models (furnace); #26466: +3% on 27B Q6_K | inline asm hazards (s_nop), untested on this model | 3 | CMake `-DGGML_HIP_GCN_DPP=ON` (default OFF) |
| P08 | server: per-request `speculative.n_max`, `--spec-draft-n-max-default`, `--spec-draft-n-max-ngram` | alex4300 | db3ede92, c4f3d550, 4b3173a8 | common/arg.cpp, common.h, speculative.{h,cpp}, server-context.cpp, server-schema.cpp | — | **textual** conflict with upstream `n_past → pos0` rename → resolved | allows per-workload MTP depth without restart; ngram deep + MTP shallow (Ornith 60→147 t/s on file work) | touches server; default off = upstream behaviour | 2 | options unset = off |
| P09 | toggles for P05/P07 | this stack | 844e42b4 | mmq.cuh, common.cuh, ggml-hip/CMakeLists.txt | P05, P07 | — | makes A/B possible | none | 0 | — |
| P10 | Q8_1 activation cache (per graph compute) | alex4300 / mx-llama | ea9489cf+00f8b13f / mx q8_1 reuse | common.cuh, ggml-cuda.cu, mmvq.cu | HIP graphs interaction | alex version **crashes with HIP graphs**; mx version resets per `graph_compute` | alex: +1.1% only | crash | **not active** (code present, alex default OFF) | `GGML_CUDA_Q8_1_CACHE=1` (do NOT use with graphs) |
| P11 | fattn-tile GQA6 (ncols2=6) | alex4300 | ac66ecb6 | fattn-tile.cuh | — | — | **negative** (n=3 947 vs 408 µs) | — | never | `GGML_FATTN_GQA6_GFX906=1` (leave unset) |
| P12 | sched_group_barrier in Q4_0-breit | alex4300 | 2c3b4e2f | mmvq.cu | — | — | negative | — | never | compile macro, off |
| P13 | CMake per-file flags `GGML_MMVQ_FLAGS`, `GGML_MMQ_Q4_0_FLAGS` | alex4300 | a7f0f414, 287c931e | ggml-hip/CMakeLists.txt | — | — | measurement only | none | 0 (empty) | CMake var empty |
| — | graph-time printf `LLAMA_GRAPH_TIME` | alex4300 | (in 4d161380) | src/llama-context.cpp | — | — | debug only | — | **dropped** | not ported |
| — | mx snapshot ring | mxxm-t/mx-llama | 7c5afc12 | 16 files (+677/−132) GDN, llama-graph, memory-recurrent | — | overlaps upstream `n_rs_seq` rollback, which is already here | measured only on 4×MI50 MoE | HIGH | 5 (analysis only) | — |
| — | furnace K-quant repack | furnace | (repack series) | — | — | — | **no Q4_0 repack exists** → irrelevant for this model | — | 4 (only if you switch to K-quants) | — |
| — | mx TP/AllReduce | mx-llama | — | — | — | — | needs P2P; this host has `hipDeviceCanAccessPeer=0` and 1 card | — | never (1 GPU) | — |

## What changed during the forward-port (exact resolutions)

| File | Conflict | Resolution |
|---|---|---|
| ggml-cuda.cu | alex GCN BF16→F16 fallback vs upstream #28846 (general BF16→F32 rule for AMD n>32) | **upstream kept**; alex rule dropped (not relevant for a Q4_0 model; BF16 only in mmproj) |
| mmq.cuh `ntiles_x` | upstream `args.ncols_opt` (MoE opt) vs alex `ncols_picker` (MoE typical width) | `min(ncols_opt, ncols_picker)` — identical for dense models (both = ncols_max) |
| mmq.cuh config selection | silent: upstream `GCN` first | new `GGML_MMQ_GCN_PROFILE` switch, host+device path |
| mmvq.cu `should_use_mmvq` | upstream Orin/Volta table vs alex VEGA20 table | both kept; VEGA20 block after NVIDIA blocks (no overlap) |
| vecdotq.cuh Q4_K scales | upstream branchless (#26705) vs alex probe macros | upstream branchless kept; alex `GGML_MMVQ_Q4K_WIDE_PROBE` (measurement-only, wrong results by design) left with `#endif` only |
| speculative.h / server-context.cpp | `n_past` → `pos0` | alex field `n_max_ngram` added before `pos0`; initializer uses `pos0 = slot.prompt.tokens.pos_next()` |

## Pairwise overlap (no summing of percentages)

| | P01 wide Q4_0 | P02 wide K/Q4_1 | P03 cutover | P05 MMQ prof. | P06 fattn | P07 DPP | P08 server | P10 q8_1 cache | HIP graphs | MTP depth |
|---|---|---|---|---|---|---|---|---|---|---|
| **P01** | — | ORTHOGONAL (different tensors) | PARTIAL (P03 decides whether P01 runs at n=6..8) | ORTHOGONAL (n≤8 vs n>8) | ORTHOGONAL | PARTIAL (P01 uses `warp_reduce_sum<TPR>` → DPP changes its reduction; small share) | ORTHOGONAL | PARTIAL (both cut y/quantize traffic; P01 already reads y once per WG → cache mostly redundant) | ORTHOGONAL | PARTIAL (gain of P01 grows with verify width) |
| **P02** | | — | PARTIAL | ORTHOGONAL | ORTHOGONAL | PARTIAL | ORTHOGONAL | PARTIAL | ORTHOGONAL | PARTIAL |
| **P03** | | | — | PARTIAL (defines where MMQ starts) | ORTHOGONAL | ORTHOGONAL | ORTHOGONAL | ORTHOGONAL | ORTHOGONAL | PARTIAL |
| **P05** | | | | — | ORTHOGONAL | ORTHOGONAL | ORTHOGONAL | ORTHOGONAL | ORTHOGONAL | ORTHOGONAL (prefill) |
| **P06** | | | | | — | PARTIAL (fattn uses warp reductions) | ORTHOGONAL | ORTHOGONAL | ORTHOGONAL | PARTIAL |
| **P07 vs #26466** | | | | | | **SAME** — choose one | | | | |
| **P10 alex vs mx** | | | | | | | | **SAME** — choose one (mx if ever) | **CONFLICTING** (alex version crashes) | |
| **P11 GQA6 vs P06** | | | | | CONFLICTING (same kernel, P11 slower) | | | | | |

## Blacklist (not in any stage)

- `v_dot8_i32_i4` as a "magic speedup" (alex measured: no gain; Q4_0 nibble order needs repacking)
- wide loads that break locality (alex: Q4_K dwordx3 scales −26%)
- lowering VGPR at the cost of LDS bank conflicts; stride-32 LDS layouts (P01 uses a padded 20-dword stride on purpose)
- deeper MTP without an acceptance analysis (see MTP-POLICY in RUNTIME-CONFIG.md)
- Q8_0 KV with MTP without a test (q8_0 KV at n>1 = full-cache dequant each step, −7…10%)
- DFlash2 before n≈8 verify gets faster
- a full GDN rewrite
- a new multi-column GEMV while P01/P02 work
- both DPP sources at once (P07 + #26466)
- alex Q8_1 cache together with HIP graphs
- `HSA_OVERRIDE_GFX_VERSION` (gfx906 is a native target in both ROCm variants)
- amdgpu-dkms on this host (GPU hangs 2026-09-08) → inbox kernel driver
- overclocking / raising power above stock 225 W
