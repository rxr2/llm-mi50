# BUILD VERIFICATION — MI50 / gfx906 / Qwen3.8-27B Q4_0

What has been verified **without** the MI50 hardware, what is verified
**automatically on the server**, and how to regenerate every artifact.
Referenced by `PATCH-MANIFEST.md` (golden SHA) and `PRE-FLIGHT-CHECKLIST.md`.

---

## 1. Patch series reproducibility (no GPU)

| Check | Result |
|---|---|
| `git am` of `patches/0001-0003` on upstream `42916d83f4a225e56709f873aa8050ac11f5b6a4` | golden SHA **`844e42b4b37a6717d03393155e9914332f891c34`**, deterministic (verified twice: fixed committer identity/date, `--committer-date-is-author-date`) |
| `build.sh` enforcement | `build.sh` **fails** if `git rev-parse HEAD != 844e42b4b…` after applying the series — a wrong base or a corrupted patch cannot produce a build |
| Clean tree | `build.sh` fails if the tree is dirty after checkout (`git status --porcelain --untracked-files=no`) |
| Host (CPU) compile of server/speculative changes | OK, GCC 14.2 |

## 2. Device build (no GPU, ROCm toolchain only)

| Check | Result |
|---|---|
| TheRock 10.1 gfx906 build (`ROCM_PATH=/opt/therock`) | full build of `llama-server` + `llama-bench` from `844e42b4b`: **0 errors**, 11.5 min on 2 cores |
| `--help` exposes the P08 server knobs | `--spec-draft-n-max-default`, `--spec-draft-n-max-ngram` present |
| DPP (P07) both states | `norm.cu`, `softmax.cu`, `mmvq.cu` compile for gfx906 with `GGML_HIP_GCN_DPP=ON` and `=OFF` |
| MMQ profile switch (P05) | `mmq-instance-q4_0.cu` with `GGML_MMQ_GCN_PROFILE=1` vs `=0` produces **different** gfx906 code objects (different md5; `mul_mat_q<Q4_0,J=64>` VGPR 79 vs 80, J=128 156 vs 157) → the upstream-profile arm is not dead code |
| Kernel resources | `build-verification/mmvq-gfx906-kernel-resources.tsv` — 124 kernels: VGPR/SGPR/LDS/scratch/max-wg per symbol. **Defaults: scratch = 0.** The single scratch case is `breit2<7,16,64>` (72 B), which is *not* a default (n=7 dispatches `breit`, `breit2` is n=5 only). `KC=128` measurement-knob instances hit VGPR 157–199 / occupancy warnings — also not defaults |

## 3. ISA verification artifacts (item: post-build, gfx906, no GPU)

`build.sh` runs `verify-isa.sh <build-dir>` after every build (skip:
`SKIP_ISA=1`). It extracts gfx906 device code objects from the build tree
(clang offload bundles + embedded AMDGPU ELF), then, using `roc-objdump` or
`llvm-objdump`:

* writes per-kernel disassembly for **at least**:
  * `q4_0_breit` n=1, n=4, n=8
  * `q5_K_breit`
  * `q6_K_breit`
* confirms each kernel's gfx906 ISA is actually present;
* confirms `private_segment_fixed_size` (**scratch**) = 0 for every default
  instantiation, cross-checked against `mmvq-gfx906-kernel-resources.tsv`;
* writes everything to **`build-verification/disassembly/`**:
  * `isa-verify-report.txt` (ends `ISA_VERIFY: PASS`/`FAIL` — build fails on FAIL)
  * `<kernel>/disassembly.txt` per required kernel
  * `.code-objects/` — the extracted gfx906 ELFs
  * `INDEX.txt`

Regenerate by hand:

```bash
./verify-isa.sh ~/mi50-builds/golden/build          # -> build-verification/disassembly/
```

**No performance interpretation is made from the artifacts** — ISA presence and
scratch-freedom are structural facts; speed needs the MI50 (T0/T1).

## 4. Correctness gate (requires the MI50)

Runs automatically inside `build.sh` (unless `SKIP_TESTS=1`) and again as
benchmark `T0`:

```
test-backend-ops test -o MUL_MAT / MUL_MAT_ID / FLASH_ATTN_EXT / GATED_DELTA_NET  -b ROCm0
test-backend-ops test -o MUL_MAT_VEC_FUSION -b ROCm0   (fused gate/GLU path, Q4_0 n=1)
```

Any `FAIL` **or any nonzero exit** aborts the build/benchmark
(`do not benchmark this build`). There is no `|| true` escape on
`MUL_MAT_VEC_FUSION` — the gate must fail the build loudly.

## 5. Harness verification (no GPU)

| Check | Command |
|---|---|
| static test battery (no GPU) | `./tests/static-tests.sh` — bash -n, py_compile (`summarize.py spec_metrics.py isa_verify.py`), Prometheus parser fixture, MTP survival-math fixture (no chain rule), repeated run-name rejection (`ALLOW_APPEND`), verify-isa multi-ELF (synthesized ELFs + fake objdump, e_flags PASS condition), record_cap FAIL/`ALLOW_POWER_MISMATCH`, contract greps |
| shell syntax | `for f in *.sh; do bash -n "$f"; done` |
| python syntax | `python3 -m py_compile summarize.py spec_metrics.py isa_verify.py` |
| full harness dry run | `DRY_RUN=1 ./benchmark-mi50.sh dry <build> <model> T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10` — must finish in seconds (no `sleep 5` in `srv_stop`, monitor/server never launch) |
| preflight | `./preflight.sh` ends `READY_FOR_MI50_BENCHMARK` or `NOT_READY:` + reasons |

## 6. Explicitly NOT verified without hardware

* any tok/s number (native ~30 / MTP 45–55 targets are alex's measurements)
* `test-backend-ops` pass/fail on the actual card
* power/thermal behaviour, PCIe link width actually negotiated under load
* T3-SYNTHETIC-NCOL costs and the real T5/T6/T7 MTP verification measurements
* T8 long-context behaviour at 8k/32k/64k

That is exactly what `./build.sh golden` → `./preflight.sh` (READY) →
`./benchmark-mi50.sh …` is for — with **zero manual file edits**
(strict preflight needs the built binaries, so the build comes first).
