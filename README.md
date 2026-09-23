# MI50-QWEN38 stack — what to build today

**1× MI50 32 GB (gfx906, wave64) · Xeon E5-2680v2 · 128 GB DDR3 · Debian 13 / kernel 6.12 · Qwen3.8-27B Q4_0 (unsloth) + MTP**

> **Answer.** Build **upstream `ggml-org/llama.cpp` @ `42916d83`** plus the 3 patches in `patches/`. That gives **`844e42b4b`**, and the SHA is reproducible.
> Install STABLE **ROCm 7.1.1** (inbox driver, no dkms, → `/opt/rocm-7.1.1`, the same path `build.sh` defaults to) and run `run-golden.sh`. Use f16 KV, `draft-mtp,ngram-mod` and a **fixed** `--spec-draft-n-max 3`.
> Day-1 flow, zero manual edits: install ROCm (`./install-rocm.sh stable`) → `./build.sh golden` → **`./preflight.sh`** → `./benchmark-mi50.sh …` (T0 first; see `PRE-FLIGHT-CHECKLIST.md`). Strict preflight needs the built binaries, so the build comes before it.
> On the first day on the server, run the A/B against **`alex4300 @ f9616ce`** exactly as built (Plan B) with `benchmark-mi50.sh`.
> No new kernel. It works and compiles for gfx906, but **it has not been measured on the card yet**. The server isn't available, so every tok/s figure in this doc comes from alex4300's measurements, not from this build.

## Files

| File | What it is |
|---|---|
| `patches/0001…0003` | the patch series (git am) on 42916d83 |
| `PATCH-MANIFEST.md` | PATCH/SOURCE/SHA/FILES/DEPS/CONFLICTS/BENEFIT/RISK/STAGE/TOGGLE, conflict resolutions, pairwise overlap, blacklist |
| `preflight.sh` | **the single pre-flight gate**: ends `READY_FOR_MI50_BENCHMARK` or `NOT_READY:` + reasons (gfx906, /dev/kfd, model sha256, binaries, ROCm + rocm-smi/rocprofv3/hipconfig paths via `rocm-env.sh`, HSA override, PCIe link recording, power cap, NUMA facts) |
| `PRE-FLIGHT-CHECKLIST.md` | BEFORE SERVER ARRIVES / FIRST BOOT WITH MI50 / FIRST 30 MINUTES / FIRST 2 HOURS |
| `BUILD-VERIFICATION.md` | what is verified without hardware, ISA artifacts, gates, harness tests |
| `build.sh` | reproducible build of 5 variants (golden, golden-upmmq, golden-dpp, alex-exact, upstream), ROCm via shared `rocm-env.sh` (prefers `/opt/rocm-7.1.1`, then `/opt/rocm`, `/opt/therock`; explicit `ROCM_PATH` must be valid), writes build-info + CMakeCache + checksums, runs `verify-isa.sh`, **strict test gate incl. `MUL_MAT_VEC_FUSION` (nonzero exit or FAIL aborts — no `|| true`)** |
| `rocm-env.sh` | the ONE ROCm resolution helper sourced by `preflight.sh` + `build.sh` + `benchmark-mi50.sh`: same `ROCM_PATH`/`PATH`/`LD_LIBRARY_PATH` and same `rocm-smi`/`rocprofv3`/`hipconfig` paths (preflight READY ⇒ benchmark tools are present) |
| `verify-isa.sh` + `isa_verify.py` | gfx906 ISA verification after build: extracts ALL AMDGPU ELF objects, **groups symbols per ELF** (never the last loop file), verifies each required default (q4_0_breit n=1 base+fused / n=4 / n=8, q5_K_breit default, q6_K_breit default) against the TSV, **`e_flags` mach == 36 is a PASS condition**, scratch=0, disassembly from each symbol's own ELF → `build-verification/disassembly/` + `isa-report.json` (TSV is COPY'd into Docker) |
| `install-rocm.sh` | STABLE (ROCm 7.1.1 → `/opt/rocm-7.1.1`) / MODERN (TheRock 10.1 gfx906) / `check` |
| `model-provenance.sh` | exact GGUF provenance: HF repo URL, filename, sha256, size (single source of truth) |
| `Dockerfile` | the same stack in a container, `--build-arg ROCM=stable\|modern`, `VARIANT=` |
| `run-golden.sh` | **MI50-QWEN38-GOLDEN-BASELINE** runtime config; finds `~/mi50-builds/golden/build/bin` automatically or takes `BIN=` explicitly (never defaults to a missing `./bin`) |
| `power.sh` | status / reset to stock 225 W / 1 s monitor with a **>95 °C abort** (writes `FILE.INVALID_THERMAL`) |
| `benchmark-mi50.sh` + `summarize.py` + `spec_metrics.py` | T0–T10 with strict preflight first; **run isolation** (existing run dir with data FAILs, `ALLOW_APPEND=1` to override); power cap recorded per test group and **FAILs outside 220–226 W** (`ALLOW_POWER_MISMATCH=1` to override); thermal watchdog (`INVALID_THERMAL`) wraps suites **and** PROFILE/T0/T1/T3/T9-native/T10-native; `PROFILE=1` rocprofv3 mode; `NUMA_MODE=auto\|off`; warm-up never touches `requests.csv` and takes no `/metrics` snapshots; each measured request stores exact `/metrics` counter deltas (29-col `requests.csv`: `verification_steps`, `draft_tokens`, `accepted_tokens`, `mean_target_width`, `survival_per_pos`, `metrics_label`); `summarize.py` renders survival-based P(A=k) distributions (never chain-multiplied); results in `results/YYYY-MM-DD/<run>/` (dry run verified: T0–T10 in ~1 s) |
| `prompts/` | CODING-1 (Python), CODING-2 (C++), EDIT, JSON, AGENT, PROSE |
| `tests/static-tests.sh` | offline static battery: bash -n, py_compile, Prometheus fixture, survival math, run-name rejection, verify-isa multi-ELF, record_cap, contract greps |
| `build-verification/` | VGPR/LDS/scratch table of every gfx906 kernel (the TSV `verify-isa` defaults); `disassembly/` (generated) |

---

## 1. Code lineage

```
ggml-org/llama.cpp master ─────────────●e107984b (09-03) ─────── 351 commits ───────●42916d83 (today)
                                        │                         incl. #27841 GCN-MMQ (c8edceb0),     │
                                        │                         #28475 mmid race, #28846 BF16,        │
                                        │                         #26705 Q4_K scales, pos0 API          │
                                        │                                                               │
 alex4300/llama.cpp-gfx906-opt  gfx906 ─┴─ 47 commits ──●f9616ce ── ggml/ + server diff ──► forward-port ─►●844e42b4b  ◄ GOLDEN
   (Unsloth qwen4exp #27742 inside)          │  wide GEMV Q4_0/Q5_K/Q6_K/Q4_1, gcn5 MMQ,            (0001 ggml, 0002 server,
   ▲                                         │  fattn head256, VEGA20 cutover, server n_max            0003 toggles + DPP)
   │ templates taken, default OFF            │                                                          ▲
 iacopPBK/llama.cpp-gfx906 125db33d ─────────┘  (Q8_0/Q4_0/IQ4_NL GEMV → alex measured: no gain)        │
   DPP + q8 cache (SAME idea as furnace/#26466/mx)                                                      │
                                                                                                        │
 y-morgunov/llamacpp-gfx906-furnace 016dde3b ─ DPP 32f28424 (sixvolts) ─────── cherry-pick + =&v fix ────┘ (opt-in, Stage 3)
   └ K-quant repack (no Q4_0 repack → not relevant)
 mxxm-t/mx-llama.cpp eefc4e73 ─ TP/AllReduce (needs P2P: host has none), q8_1 reuse (SAME as alex P10),
   └ snapshot ring 7c5afc12 (Stage 5, analysis only; upstream n_rs_seq rollback already covers 1 sequence)
 PR #26466 (DRAFT, not merged) ─ DPP ← maximumbusdatatype #16291: SAME as furnace → not taken
 exabit / hipfire / arte-fact turbo ─ not used (turbo: inactive since 05-12, −O1 miscompile)
```

| Patch family | Where it exists | Status |
|---|---|---|
| wide multi-column GEMV (y in LDS) | alex only | **fork-only** → ported (P01/P02) |
| GCN MMQ profile | alex gcn5 **and** upstream #27841 | **duplicate** → one switch, A/B (P05) |
| DPP reductions | furnace, #26466, iacop, mx | **4× the same idea** → take only furnace (P07), opt-in |
| Q8_1 activation cache | alex, mx, iacop, milpster | duplicate; low gain (+1.1%) → not active |
| SWIGLU_CLAMP, mmid race | alex / upstream #28475 | upstream already has the race fix; alex's GLU fix is in P04 |
| BF16 fallback | alex / upstream #28846 | **already upstream** → alex version dropped |
| server n_max per request, ngram cap | alex only | fork-only → ported (P08) |
| rollback of recurrent state (MTP) | upstream `n_rs_seq` | **already upstream**, also in alex |
| snapshot ring | mx only | fork-only, HIGH, Stage 5 |

## 2. Variant A vs B → choice: **A′** ("alex patches as a series on current upstream")

After the forward-port it turned out A and B converge to **the same code**. What differs is how it's maintained.

| Criterion | A: alex branch + merge upstream | B: upstream + backport | **A′: upstream + patch series (chosen)** |
|---|---|---|---|
| Conflicts at the 1st sync | 6 files (trial merge) | same hunks | 6 hunks, **resolved and recorded** (MANIFEST) |
| Next sync | a merge commit over 47 commits of history, German docs, experiments | a manual job again | `git rebase` of 3 patches; conflicts only in mmvq.cu dispatch + mmq.cuh |
| Dead/experimental code | carries everything (GQA6, sched_group_barrier, probe macros, graph-time printf) | you pick | printf dropped; negative experiments remain behind toggles (default OFF) |
| Silent-conflict risk (#27841 vs gcn5) | **dead code without anyone noticing** | same | switch `GGML_MMQ_GCN_PROFILE` + A/B |
| Unit of review | 47 commits | — | 3 patches, 20 files, +4907/−435 |
| Coupling to one person | full (fork) | none | only the kernel code; the base is upstream |

A′ is B in form ("upstream + selected patches"). Its content is A: all of alex's measured kernels 1:1, no redesign.

## 3. Wide-GEMV audit (alex P01/P02)

**Files:** `ggml/src/ggml-cuda/mmvq.cu`

| Piece | Location in 844e42b4b |
|---|---|
| Q4_0 kernel | ~L1352 `mul_mat_vec_q4_0_breit_gfx906` |
| breit2 | ~L1600 |
| launcher | ~L1760–1830 |
| dispatch | ~L3380 |
| Q5_K | ~L2513/2688 |
| Q6_K | ~L2845 |
| Q4_1 | ~L2877/3007 |

| | Q4_0 `breit` | Q4_0 `breit2` (n=5 only) | Q5_K / Q6_K `breit` | Q4_1 `breit` |
|---|---|---|---|---|
| template | `<ncols 1..8, TPR 8\|16, KC 32\|64\|128, has_fusion>` | `<ncols, 16, KC>` | `<ncols 2..8, 16, KC 4\|8>` (KC in superblocks) | `<ncols, 16, 32\|64>` |
| launch_bounds | (256, 2) | (256, 2) | (256, …) | (256, …) |
| rows / WG | 256/TPR = 16 (TPR 16) or 32 (TPR 8, n=5,7) | 32 (2 rows/thread) | 16 | 16 |
| threads | 256 = 4 waves64 | 256 | 256 | 256 |
| default | TPR 16 (8 at n=5,7), KC 64 (32 at n>5) | KC 64 | KC 8 (4 at n>5) | KC 64/32 |
| LDS (measured in the TheRock build) | n·KC/2·(80+16) B: n1 KC64 3 KB … n8 KC32 12 KB | n5 KC64 15 KB | Q5_K n8 KC4 12 KB | n8 KC32 12 KB |
| VGPR (default instances, TheRock clang 23) | 36–66, **scratch 0** | 62–102; n7 KC64 = 128 + **72 B scratch** (not the default: n7 uses breit) | 50–84 | 31–55 |
| layout y (LDS) | `int4 y_qs[n][KC/2][5]`: 2 Q8_1 blocks = 4×int4 + **1 int4 padding** (20-dword stride against bank conflicts) + `float4 y_ds` (d0,s0,d1,s1) | same | per-superblock scales in LDS | same |
| layout x | 2 Q4_0 blocks = 36 B read as 9 dwords, `alignbit` for the 2-byte offset, nibbles unpacked once into `vl[8]`, `vh[8]` | | | |
| dot | `v_dot4_i32_i8` (dp4a), 16 per column per block pair | | | |
| epilogue | `d4·(sumi·d8 − 8·s8)` in FP32 | | | |
| reduction | `warp_reduce_sum<TPR>` (shfl_xor, sub-wave groups of 16/8 lanes) — **this is where DPP (P07) would apply** | | | |
| fusion | gate+GLU+bias only at n=1 (replaces the generic n=1 path) | – | – | – |
| host conditions | VEGA20, n 1..8, no ids, `ncols_x % 64 == 0`, even stride, no x_scale | | Q5_K/Q6_K: `ncols_x % 256 == 0` | |

**Why it's faster** (split out; alex measured on 4096×14336):

1. **Activation reuse (main lever).** The generic MMVQ reads y (Q8_1) from L2 again for every row block. At n=8 that's 134 MB against 29 MB of L2 traffic. Here y goes to LDS once per WG and all 16–32 rows of the WG reuse it.
2. **Weight reuse.** Each Q4_0 block pair is read and unpacked **once** and used for n columns. The generic kernel repeats the unpack per column.
3. **Dequant reuse.** The nibble masks and `d4` are computed once per block pair, not per column.
4. **L2.** Fewer y re-reads free L2 for weights. The weights stay a stream (no wide loads that would break locality).
5. **LDS.** The padded stride removes bank conflicts. KC drops at n>5 so that 2 WG/CU still fit (KC 64 at n=8 → 140 instead of 103 µs).
6. **Launches.** Same count (1 kernel per matmul). There is no gain from launches.

**Comparison with upstream since the base (e107984b → 42916d83):** upstream has **no** multi-column LDS GEMV for GCN.

| Question | Answer |
|---|---|
| 1:1 port? | **yes** (0001, dispatch unchanged) |
| Redesign dispatch? | **no.** Only the VEGA20 table next to Orin/Volta |
| Conflict with MMQ? | no: n≤8 → MMVQ, n>8 → MMQ. #27841 changes only MMQ (n>8) |
| Redundant? | **no** |

**New fact from the build under TheRock/clang 23:** the `KC=128` instances hit VGPR 157–199 and occupancy 1. Linker warning: `failed to meet occupancy target`. They are **only reachable** via `GGML_MMVQ_Q4_BREIT_KC=128` (a measurement knob), so defaults are unaffected. Still, if **clang 23 vs ROCm 7.1 clang** changes VGPR for the default instances, T1 will show it. The table is in `build-verification/`.

## 4. MMVQ/MMQ cutover (VEGA20, dispatch policy only)

Source: `ggml_cuda_should_use_mmvq`, VEGA20 block (alex b19fed0a), plus the wide dispatch.

| quant \ n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9–16 |
|---|---|---|---|---|---|---|---|---|---|
| **Q4_0** (93% of the model's weights) | breit (+fused gate/GLU) | breit | breit | breit | **breit2** | breit KC32 | breit TPR8 KC32 | breit KC32 | MMQ (gcn5 I=128 **or** #27841 I=64 → A/B) |
| **Q4_1** (8× ffn_down) | generic MMVQ | breit | breit | breit | breit | breit | breit | breit | MMQ |
| **Q5_K** (48× ssm_out) | gfx906 n=1 GEMV | breit | breit | breit | breit | breit | breit | breit | MMQ |
| **Q6_K** (output head 5120×248320) | gfx906 n=1 GEMV | breit | breit | breit | breit | breit | breit | breit | MMQ |
| **Q8_0** (MTP eh_proj) | generic MMVQ | generic | generic | generic | generic | generic | generic | generic | MMQ |
| Q4_K (other models) | gfx906 GEMV | generic | generic | generic | generic | **MMQ** | MMQ | MMQ | MMQ |
| IQ4_XS (other models) | generic | … | | | | | | ≤8 | MMQ |

The verify batch for MTP n-max 3 is 4 columns (1 + 3). ngram-mod can push it up to 8 or more. Above 8 it falls into the **MMQ valley** (~400 µs per 4096×14336 against 104 µs at n=8). That is why `--spec-draft-n-max` above 7 makes no sense on this model without a separate ngram cap. **T3-SYNTHETIC-NCOL** (test id T3) measures the n-column cost curve — it characterizes GGML/GEMV/MMQ width cost, it does **not** measure MTP verify latency.

## 5. Dispatch map tensor → dispatch → kernel (Qwen3.8-27B Q4_0, decode/verify n ≤ 8)

Read from the GGUF header (866 tensors). There are 64 blocks: 48 DeltaNet + 16 attention, plus the MTP block.

| Tensor (count) | Type | k×m | n=1 | n=2..8 (MTP verify) |
|---|---|---|---|---|
| ffn_gate + ffn_up (64+64) | Q4_0 | 5120×17408 | **fused**: `breit<1,16,64,true>` gate+up+SwiGLU in 1 kernel (`ggml_cuda_should_fuse_mul_mat`) | 2× `breit<n,…>` + separate GLU (no fusion at n>1) |
| ffn_down (56) | Q4_0 | 17408×5120 | `breit<1>` | `breit<n>` |
| ffn_down (8) | **Q4_1** | 17408×5120 | generic MMVQ | `q4_1_breit<n>` |
| attn_qkv (48, DeltaNet) | Q4_0 | 5120×10240 | `breit<1>` | `breit<n>` |
| attn_gate (48, DeltaNet) | Q4_0 | 5120×6144 | `breit<1>` | `breit<n>` |
| ssm_out (48) | **Q5_K** | 6144×5120 | `q5_K_gfx906` | `q5_K_breit<n>` |
| ssm_alpha/beta (48+48) | F32 | 5120×48 | MMVF (f32) | MMVF / rocBLAS |
| attn_q (16) | Q4_0 | 5120×12288 | `breit<1>` | `breit<n>` |
| attn_k, attn_v (16+16) | Q4_0 | 5120×1024 | `breit<1>` | `breit<n>` |
| attn_output (16) | Q4_0 | 6144×5120 | `breit<1>` | `breit<n>` |
| output (1) | **Q6_K** | 5120×248320 | `q6_K_gfx906` | `q6_K_breit<n>` (16% of the 8-token step) |
| token_embd | Q4_0 | — | get_rows | get_rows |
| MTP: attn_q/k/v/o, ffn_gate/up/down (1 each) | Q4_0 | as above | breit (drafter runs n=1 per draft step) | — |
| MTP: nextn.eh_proj | **Q8_0** | 10240×5120 | generic MMVQ (`GGML_MMVQ_Q8_GFX906=1` = iacop GEMV, measured: no gain) | generic |
| attention (16 layers) | f16 KV | head 256, GQA 6 | fattn-tile, alex head256 config (P06) | fattn-tile |
| DeltaNet (48 layers) | F32 state | S 128, H_v 48 | GATED_DELTA_NET AR | GDN chunked, K = n_rs_seq + 1 snapshots (upstream) |

## 6. PR #27841 — status and A/B

The PR is merged upstream (c8edceb0) and is part of the base. For GCN it adds its own wave64 profile. Its reported gains are measured against old upstream, **not against alex gcn5**:

| Quant | #27841 gain vs old upstream |
|---|---|
| Q4_0 | +37% |
| Q4_K | +5.7% |
| Q8_0 | +46% |
| IQ4_XS | +18.5% |

**Silent conflict:** upstream's GCN branch comes first, so alex gcn5 would be dead code. The `GGML_MMQ_GCN_PROFILE` switch fixes that.

**A/B (T2 + T1, builds `golden` vs `golden-upmmq`):**

| Quant | Tests | Decision |
|---|---|---|
| Q4_0 | pp512, pp2048, T8 prefill 8k/32k | profile with higher median pp over 5 reps; tie within ±1.5% → keep **upstream (0)** for maintainability |
| Q4_K, Q8_0, IQ4_XS | `test-backend-ops perf -o MUL_MAT` n=16..512 (other models, no download needed) | per quant; the model only needs Q4_0 plus the n=9..16 cutover |
| cutover | T1: n=8 (MMVQ) vs n=9..16 (MMQ) per profile | unchanged unless MMQ n=9 < MMVQ n=8 |

If upstream (0) wins or ties, drop alex's gcn5 table from the series at the next rebase. That's −353 lines of maintenance.

## 7. DPP (P07)

| | furnace 32f28424 (taken) | PR #26466 | iacopPBK |
|---|---|---|---|
| correctness | test-backend-ops 12367/12367 (their build) | draft, own bench | — |
| wave64 | xor1/2 quad_perm, xor4 = 2× masked row_shl/shr, xor8 row_ror:8, xor16 ds_swizzle SWAP16, xor32 = shfl (bpermute) | GFX9 path added, since row_share/xmask are GFX10+ | similar |
| s_nop | s_nop 4 / 1 before DPP (VALU→DPP hazard; the compiler can't see across asm boundaries) | yes | ? |
| bug | **xor4: missing early-clobber `=&v`** → the compiler may assign the same VGPR to in and out, and the first `v_mov` destroys the input. **Fixed in 0003.** | — | — |
| gain | tg +2.9…4.7% (gemma 31B, qwen 0.8B) | tg +3% (27B Q6_K) | — |

The reduction is a small share of the wide GEMV. Expect **≤ +3% effective**, and only in Stage 3 with the T0 gate plus A/B. It's compiled and verified in both states (OFF and ON) under TheRock.

## 8. Q8_1 activation cache

- **alex (ea9489cf/00f8b13f):** per-graph cache in `common.cuh`/`ggml-cuda.cu`, used from `mmvq.cu`. Default OFF: with HIP graphs the pointer is invalid after capture, which crashes. Gain +1.1%.
- **mx:** MAX_ENTRIES 2, keyed on src1 ptr/ne/strides, `q8_1_cache_reset()` at the start of every `graph_compute`, thread-local `g_cuda_outer_capture`, reset and direct eval on a graph failure. Memory: 2 × n × k × 36/32 B (< 1 MB).
- **Decision:** not in Stage 1–2. Wide GEMV already reads y only once per WG, so the cache would only save `quantize_q8_1` (a few µs per matmul). If ever needed, use mx's version with the per-compute reset, Stage 3+.

## 9. Weight repack

furnace repacks **only K-quants** and there is **no Q4_0 repack** anywhere in the lineage. Q4_0 is already a simple 18-byte block, and wide GEMV reads it as 9 dwords per pair with `alignbit`. **Stage 1 has no repack.** Repack is Stage 4 and only applies if you move to Q4_K_M/UD-Q4_K_XL for quality.

## 10. f16 KV cache (16 attention layers × 4 KV heads × 256 × 2 (K+V) × 2 B = 64 KiB/token)

| ctx | f16 KV | + weights 16.06 GB + compute/MTP ~1.5 GB | fits 32 GB? |
|---|---|---|---|
| 2k | 0.13 GB | ~17.7 | yes |
| 8k | 0.54 GB | ~18.1 | yes |
| 32k | 2.15 GB | ~19.7 | yes |
| 64k | 4.29 GB | ~21.9 | yes (**golden**) |
| 128k | 8.59 GB | ~26.2 | yes, but check compute buffers (ubatch 512) in T8 |

The DeltaNet state is constant: 48 × (128×128×48 + conv) × 4 B ≈ 150 MB per sequence, × (n_rs_seq+1) snapshots.

## 11. Snapshot ring (mx 7c5afc12) — complexity **HIGH**, Stage 5

The change touches 16 files, +677/−132: GDN kernels, llama-graph, memory-recurrent, qwen4exp. It handles 1 sequence.

Upstream already has the same core feature, `n_rs_seq` rollback (K snapshots written by the GDN op, `seq_rm` via `rs_idx`), and this build uses it (`need_n_rs_seq()` → draft.n_max). mx measured its gain only on 4×MI50 MoE. For a dense model at n-max 3, the rollback cost is small.

**Do not port.** Only analyse it once the T3/T6 profile shows > 5% of step time in copying recurrent state.

## 12. HIP graphs

Graphs are ON at build time (`-DGGML_HIP_GRAPHS=ON`). To turn them off at runtime, set `GRAPHS=0` in run-golden, which sets `GGML_CUDA_DISABLE_GRAPHS=1`.

They're required OFF for rocprofv3 (SIGSEGV otherwise, per alex). T9 measures ON vs OFF. Graphs are not the main lever.

## 13. ROCm

| | STABLE (default) | MODERN |
|---|---|---|
| version | ROCm **7.1.1** userspace (alex measured "ROCm 7.1") | TheRock **10.1.0a20260822** gfx906 tarball (2.1 GB, sha256 6ae3c683…) |
| driver | **inbox amdgpu** of kernel 6.12 (no dkms) | inbox amdgpu |
| LLVM | ROCm 7.1 amdclang | AMD clang **23.0.0git** (0bace190) |
| rocBLAS/Tensile | the 7.x package can omit gfx906 → `install-rocm.sh` shows a fix (Docker image or build rocBLAS `-a gfx906`) | **210 gfx906 files included** (verified) |
| rocprofiler | rocprofv3 + libdw1t64 workaround (Debian 13) | rocprofv3, rocprof-compute included |
| target | `gfx906` (the MI50 runs xnack− / sramecc+; `AMDGPU_TARGETS=gfx906` builds a generic target that loads on both sramecc settings — do not force `gfx906:sramecc+:xnack-` unless rocminfo shows a mismatch) | same |
| HSA_OVERRIDE | **no** | **no** |
| official support | MI50 deprecated but works | ROCm 10 system requirements mark MI50 ❌; the TheRock nightly still builds gfx906 |
| **verified here** | not built (no apt) | **built and linked**: llama-server + llama-bench, 844e42b4b, both DPP states |

MODERN is the fallback path and the long-term one. STABLE gives comparability with alex's numbers. Compiler differences go through the same T1/T2 before switching.

## 14. Power / clocks

Stock is **225 W**; no OC and no clock or voltage changes.

1. Before every run: `./power.sh status` → cap must be 225 W. If not: `sudo ./power.sh stock`. `./preflight.sh` fails outside 220–226 W.
2. During a run: 1 s monitor (`benchmark-mi50.sh` runs it by itself), saved as `power_*.csv`. **The actual cap is recorded before every test group** into `power_caps.csv` — and a real benchmark now **FAILs (aborts)** if the cap is unreadable or outside `POWER_CAP_MIN..MAX` (220–226 W); the only override is an explicit `ALLOW_POWER_MISMATCH=1` (never the default). `summarize.py` never aggregates runs with different caps (separate sections + warning).
3. Numbers are comparable only at the same cap. alex measured that 125→225 W gives +22% decode, so a run at a different cap is invalid.
4. **Abort if Tj > 95 °C — implemented, not just commented:** the benchmark-side watchdog samples junction temp every second; on breach it terminates the server AND the benchmark, writes `INVALID_THERMAL` into the run dir, and `summarize.py` marks the run partial/invalid. `power.sh monitor` has the same abort (`FILE.INVALID_THERMAL`, exit 95). Threshold: `TEMP_MAX_C` (default 95). Note that at 225 W decode sits at ~220 W / 1701 MHz, so the card is power-limited.

## 15. Stages (each with its own benchmark; stop as soon as the target is met)

| Stage | Contents | Build | Bench | Exit criterion |
|---|---|---|---|---|
| **0** | Plan B references: `alex-exact` (f9616ce) and `upstream` (42916d83, no patches) | build.sh alex-exact / upstream | T0, T2, T3, T6 | numbers match alex (native ~30, MTP3 ~53 code); upstream shows how much the patches add |
| **1** | **GOLDEN** = 0001+0003 (wide GEMV, gcn5, fattn head256, cutover), graphs ON | `golden` | T0–T6, T9 | effective ≥ alex-exact − 2% on CODING-1/2, EDIT, AGENT; test-backend-ops 100% |
| 1b | MMQ profile A/B | `golden-upmmq` | T2, T1, T8 | §6 |
| **2** | 0002 server: n_max per request / separate ngram cap | (same build) | T7 | e.g. `mtp2_ngram64` beats `mtp3` on EDIT/AGENT (lots of repetition) without losing on CODING |
| **3** | DPP (P07) | `golden-dpp` | T0, T2, T6 | ≥ +1.5% effective on coding, 0 test FAILs, T10 determinism OK |
| 4 | only when changing to K-quants: furnace repack + Q4_K cutover | — | T1, T2 | n/a for Q4_0 |
| 5 | snapshot ring analysis, Q8_1 cache (mx), DFlash2 once n≈8 verify gets faster | — | `PROFILE=1` trace | only if a profile shows > 5% there |

## 16. MTP policy

- **Start fixed:** n-max **3** (golden). No adaptive 3..12.
- **Depth by acceptance:** measured per workload in T5/T6/T7.

| Acceptance at depth 3 | Depth |
|---|---|
| < 0.50 | 2 |
| 0.50–0.75 | 3 |
| > 0.75 on ≥ 2 coding workloads, and T7 mtp4 wins by more than the rep spread | 4 |

  Real-code acceptance so far: median 0.59, so depth 3. For reference, alex measured at depth 2/3/4: 50.0/53.1/52.9 on code, 47.3 on prose.
- **Logging:** the `-lv 4` server trace gives `draft acceptance`, `mean len`, and `acc per pos` for each slot. It goes to `acc_per_pos.csv`. The API `timings.draft_n` / `draft_n_accepted` goes into each request row.
- **T3 is NOT the MTP verification measurement.** T3 = **T3-SYNTHETIC-NCOL**: `llama-bench -p N -n 0` characterizes the synthetic n-column GGML/GEMV/MMQ cost for n=1..16 (width cost curve). Keep it — it is useful — but never report it as actual MTP verification latency.
- **Real MTP verification measurement (T5/T6/T7):** per request the harness records `draft_n`, `draft_n_accepted`, acceptance, total `predicted_ms`, wall time and TTFT; per config it records acceptance per position (server trace) and the target batch width distribution. If the exact target verification time is not directly exposed by the server, derived values (`verify_steps_EST`, `ms_per_verify_EST`, batch-width distribution) are labeled **ESTIMATE** — `predicted_ms / verify_steps_EST` is never called "measured verify latency" (it contains draft + verify + overhead). With `PROFILE=1` the kernel trace/profile of a short fixed workload is saved alongside (`profile/`).
- **Effective TPS** = predicted_n / predicted_ms. This is the only success metric.

| Level | Effective tok/s |
|---|---|
| Native | ~30 |
| MTP target | 45–55 |
| Stretch | > 55 |

  The ceiling at forced acceptance is 87.5.
- **Depth > 4** only with T7 data. A draft with ngram cap > 7 falls into the MMQ valley (§4).

## 17. A/B test plan (day 1 on the server)

Order matters: correctness, then references, then golden, then variants. Same power cap, same model sha, 3 seeds, 768 tokens, temp 0.6.

```bash
sudo ./power.sh stock && ./power.sh status          # 225 W
./install-rocm.sh check
./build.sh alex-exact && ./build.sh upstream && ./build.sh golden && ./build.sh golden-upmmq && ./build.sh golden-dpp
./preflight.sh                                      # must end READY_FOR_MI50_BENCHMARK (needs the binaries)
M=/models/Qwen3.8-27B-Q4_0.gguf; B=~/mi50-builds
./benchmark-mi50.sh A0-alex-exact  $B/alex-exact/build   $M T0,T2,T3,T4,T6
./benchmark-mi50.sh A0-upstream    $B/upstream/build     $M T0,T2,T3,T4,T6
./benchmark-mi50.sh A1-golden      $B/golden/build       $M            # all T0-T10
./benchmark-mi50.sh A1b-upmmq      $B/golden-upmmq/build $M T0,T1,T2,T8
./benchmark-mi50.sh A3-dpp         $B/golden-dpp/build   $M T0,T2,T6,T10
```

Every `benchmark-mi50.sh` invocation re-runs the strict preflight first (no file edits needed: `ALLOW_MODEL_MISMATCH=1` is the sanctioned model-sha override; `ALLOW_APPEND=1` explicitly allows appending to an existing run dir — the default is FAIL; `ALLOW_POWER_MISMATCH=1` explicitly allows a non-stock power cap — the default is FAIL; `NUMA_MODE=auto|off` picks the binding; `PROFILE=1` adds the short rocprofv3 workload).

| Comparison | Metric | Decision |
|---|---|---|
| golden vs alex-exact | coding-median effective tok/s T6 | golden ≥ alex − 2% → golden is the base. Otherwise: **Plan B = run alex-exact** in production and bisect the difference (0001 vs 42916d83 upstream changes: qwen4exp fusions, GDN) |
| golden vs upstream | T4/T6 | shows the value of the patches; if < 5%, rethink whether keeping the fork is worth it |
| golden vs golden-upmmq | pp512/pp2048/T8 | §6 |
| golden vs golden-dpp | T6 coding + T10 | ≥ +1.5% and determinism OK → Stage 3 enabled |
| T5/T6/T7 | acceptance per workload | §16, the depth per workload |
| T9 | graphs ON/OFF | ON unless OFF ≥ ON (then investigate) |
| T8 | f16 vs q8_0 KV at 32k/64k | f16 stays; q8_0 only as a test arm |

## 18. Final fork architecture (A–L)

| | Element | Decision |
|---|---|---|
| A | base | upstream ggml-org 42916d83 |
| B | patch transport | `git format-patch` series of 3, rebased on every upstream sync; SHA verified by build.sh |
| C | decode kernels | alex wide GEMV Q4_0/Q4_1/Q5_K/Q6_K 1:1 |
| D | cutover | alex VEGA20 table (Q4_K→MMQ n>5, others n>8) |
| E | prefill | MMQ, gcn5 **or** #27841 (A/B, then drop one) |
| F | attention | fattn-tile with alex head256 config, f16 KV |
| G | DeltaNet / rollback | upstream GDN + `n_rs_seq` (no snapshot ring) |
| H | speculation | upstream draft-mtp + ngram-mod, alex server caps (0002) |
| I | reductions | generic; DPP opt-in (Stage 3) |
| J | graphs | HIP graphs ON, env off |
| K | toolchain | ROCm 7.1.1 STABLE / TheRock 10.1 MODERN, gfx906 native, inbox driver |
| L | measurement | benchmark-mi50.sh T0–T10 + summarize.py, results in git |

## 19. Plan B

If golden turns out slower than alex-exact on day 1 (> 2% on coding), run **alex-exact (f9616ce) as built** in production. It's fully measured: 50–53 t/s MTP on code.

Then bisect: take 42916d83 upstream without 0001, check whether the qwen4exp/GDN changes caused it, and apply the patches one at a time.

The upstream arm with no wide GEMV shows the patches' real value on day 1.

## 20. Build verification (in this environment, no GPU)

Full write-up: **`BUILD-VERIFICATION.md`**. ISA disassembly artifacts: **`build-verification/disassembly/`** (generated by `verify-isa.sh` at the end of every `./build.sh`; q4_0_breit n=1/4/8, q5_K_breit, q6_K_breit; confirms gfx906 ISA + scratch=0 for defaults; no performance claims without hardware).

- **Forward-port and SHA:** `git am` of the 3 patches on 42916d83 gives **844e42b4b37a…** every time (checked twice).
- **CPU compile check:** the host code for the server/speculative changes compiles (GCC 14.2).
- **TheRock 10.1 gfx906 build:** full build of `llama-server` and `llama-bench` in 11.5 min with 2 cores, 0 errors. `--help` shows `--spec-draft-n-max-default`, `--spec-draft-n-max-ngram`.
- **DPP=ON:** `norm.cu`, `softmax.cu`, `mmvq.cu` compiled for gfx906 without errors.
- **Kernel resources:** `build-verification/mmvq-gfx906-kernel-resources.tsv` (124 kernels). Default instances have 0 scratch. The only scratch case is `breit2<7,16,64>` (not the default).
- **MMQ profile switch works:** `mmq-instance-q4_0.cu` with `GGML_MMQ_GCN_PROFILE=1` vs `=0` produces different gfx906 code objects (different md5; `mul_mat_q<Q4_0,J=64>` VGPR 79 vs 80, J=128 156 vs 157). The upstream-profile arm is therefore not dead code.
- **Not verified:** `test-backend-ops` and any tok/s. That needs the MI50.
