# PRE-FLIGHT CHECKLIST — MI50 / gfx906 / Qwen3.8-27B Q4_0 benchmark stack

Four phases: **BEFORE SERVER ARRIVES** · **FIRST BOOT WITH MI50** · **FIRST 30
MINUTES** · **FIRST 2 HOURS**. Nothing here requires editing a file by hand —
every step is a command.

The gate for everything after phase 2 is one command:

```bash
./preflight.sh
```

It ends with exactly `READY_FOR_MI50_BENCHMARK` or
`NOT_READY:` + specific reasons. **Do not benchmark until it is READY.**

---

## BEFORE SERVER ARRIVES

Do all of this on any dev machine (no GPU needed):

- [ ] Clone/copy this repository; verify the working tree is clean:
      `git status`
- [ ] Confirm the executable bits survived the transfer (zip/git preserve them;
      this repo ships +x on all scripts):
      ```bash
      ls -l build.sh install-rocm.sh run-golden.sh power.sh \
            benchmark-mi50.sh summarize.py preflight.sh verify-isa.sh
      # every line must show -x (rwxr-xr-x)
      ```
- [ ] Run the static tests:
      ```bash
      for f in *.sh; do bash -n "$f"; done
      python3 -m py_compile summarize.py
      ```
- [ ] Run the harness dry run (finishes in seconds, needs no hardware):
      ```bash
      DRY_RUN=1 ./benchmark-mi50.sh dryrun ./prompts /dev/null T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10
      ```
- [ ] Read `README.md`, `PATCH-MANIFEST.md`, `BUILD-VERIFICATION.md`.
- [ ] Stage the model on a disk that will be moved to the server (exact
      provenance — record it, verify it):
      * repo:    https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
      * file:    `Qwen3.8-27B-Q4_0.gguf`
      * sha256:  `ede16c7b36e578ca87a8c70e011e4b4633a32c831c0ce76d0f474582384e671d`
      * size:    `16056478688` bytes
      ```bash
      sha256sum Qwen3.8-27B-Q4_0.gguf    # must match exactly
      ```
      (preflight re-checks sha256 + size; `ALLOW_MODEL_MISMATCH=1` is the only
      sanctioned override and it is recorded.)
- [ ] Note the expected stock power cap: **225 W** (accepted range 220–226 W;
      override bounds with `POWER_CAP_MIN`/`POWER_CAP_MAX` only if you know why).
- [ ] Have `install-rocm.sh` ready: STABLE installs **ROCm 7.1.1 to
      `/opt/rocm-7.1.1`** — the same path `build.sh` defaults to. Do NOT set
      `HSA_OVERRIDE_GFX_VERSION` anywhere.
- [ ] Print/keep `PRE-FLIGHT-CHECKLIST.md` — phases 2–4 are done on the server.

---

## FIRST BOOT WITH MI50

Hardware + kernel driver only. No ROCm userspace yet.

- [ ] Seat the MI50 (x16 slot on the CPU-local PCIe root complex if possible),
      connect the PCIe power connector, verify the fans spin.
- [ ] Boot. Confirm the **inbox amdgpu** driver — amdgpu-dkms previously caused
      GPU hangs on this host and must NOT be installed:
      ```bash
      uname -r                          # expected: 6.12.x
      modinfo amdgpu | grep filename    # must NOT contain updates/dkms
      dmesg | grep -Ei 'amdgpu|vega' | tail -20
      ```
      If dkms got pulled in: `sudo apt purge amdgpu-dkms` and reboot.
- [ ] Device nodes and groups:
      ```bash
      ls -l /dev/kfd /dev/dri/         # both must exist
      id -nG                           # user must be in: render video
      # if not: sudo usermod -aG render,video $USER  && re-login
      ```
- [ ] PCI visible:
      ```bash
      lspci -nn | grep -Ei 'vega 20|MI50|66a[0-7]'
      ```
- [ ] No HSA override anywhere (check shell rc files, systemd units, docker):
      ```bash
      env | grep HSA_OVERRIDE   # must be empty
      ```
- [ ] Record PCIe link + NUMA facts (also done automatically by preflight):
      ```bash
      BDF=$(lspci -D | grep -Ei 'vega 20|MI50|66a[0-7]' | awk '{print $1}')
      cat /sys/bus/pci/devices/$BDF/current_link_speed \
           /sys/bus/pci/devices/$BDF/current_link_width \
           /sys/bus/pci/devices/$BDF/numa_node
      lscpu -e=CPU,SOCKET,NODE,CORE | head
      numactl --hardware
      lspci -tv
      ```
      On a dual-socket board the `numa_node` of the MI50 is the node
      `NUMA_MODE=auto` (the default) will bind CPU + memory to; it is saved in
      `meta.json` with every run.

---

## FIRST 30 MINUTES

ROCm userspace + build. Still no benchmark numbers.

- [ ] Install STABLE ROCm (userspace only, no dkms):
      ```bash
      sudo ./install-rocm.sh stable      # -> /opt/rocm-7.1.1
      ./install-rocm.sh check            # must show: /dev/kfd ok, gfx906, no override
      ```
      If rocBLAS ships without gfx906 Tensile files, follow the printed
      `fix_rocblas` hints (Docker image or rebuild rocBLAS `-a gfx906`).
- [ ] Power to factory stock (225 W cap, auto perf level, no OC):
      ```bash
      sudo ./power.sh stock && ./power.sh status
      ```
- [ ] Put the model where the tools look for it (or always pass the path):
      `/models/Qwen3.8-27B-Q4_0.gguf` (also probed: `~/models/`, `models/`).
- [ ] **Run the gate:**
      ```bash
      ./preflight.sh
      ```
      Must end `READY_FOR_MI50_BENCHMARK`. Every `NOT_READY:` line names the
      exact failing check (gfx906 detection, /dev/kfd, model sha256, build
      binaries, ROCm identity, HSA override, PCIe link recording, power cap).
      Common fixes:
      * build binaries missing → `./build.sh golden` (writes
        `~/mi50-builds/golden/build/bin`, which `run-golden.sh` finds
        automatically; or pass `BIN=`/`--build` explicitly)
      * power cap off → `sudo ./power.sh stock`
      * deliberate non-standard model → `ALLOW_MODEL_MISMATCH=1 ./preflight.sh`
        (recorded, never silent)
- [ ] Build (also generates the ISA verification artifacts):
      ```bash
      ./build.sh golden
      ls build-verification/disassembly/   # isa-verify-report.txt + per-kernel .s
      ```
      `build.sh` fails if the golden patch series does not reproduce
      `844e42b4b…` or if `verify-isa.sh` cannot confirm gfx906 ISA + scratch=0
      for the default breit kernels (q4_0 n=1/4/8, q5_K, q6_K).
- [ ] Re-run `./preflight.sh` after the build — now with binaries — and keep
      the `preflight-report/` folder as the machine's birth certificate.

---

## FIRST 2 HOURS

Correctness first, then short benchmarks; watch power and temperature the
whole time.

- [ ] Re-verify the gate and power in the same shell you will benchmark from:
      ```bash
      ./preflight.sh && ./power.sh status
      ```
- [ ] Harness dry run on the server itself (seconds, no GPU load):
      ```bash
      DRY_RUN=1 ./benchmark-mi50.sh dry /path/to/build /models/Qwen3.8-27B-Q4_0.gguf
      ```
- [ ] Correctness gate on hardware (test-backend-ops, fails the run on FAIL):
      ```bash
      ./benchmark-mi50.sh T0-gate /path/to/golden/build /models/Qwen3.8-27B-Q4_0.gguf T0
      ```
- [ ] First short real groups — same power cap is mandatory (each test group
      records its actual cap into `power_caps.csv`; `summarize.py` refuses to
      aggregate mixed caps):
      ```bash
      ./benchmark-mi50.sh day1-smoke ~/mi50-builds/golden/build /models/Qwen3.8-27B-Q4_0.gguf T2,T3,T4,T6
      ```
      * `T3` = **T3-SYNTHETIC-NCOL** (synthetic n-column cost n=1..16 — NOT MTP
        verify latency)
      * real MTP verification numbers come from `T5/T6/T7`
- [ ] Thermal rules while anything runs:
      * watchdog threshold: junction **> 95 °C** (`TEMP_MAX_C`, default 95)
      * on trip: server AND benchmark are terminated, the run directory gets
        `INVALID_THERMAL`, and `summarize.py` marks the run partial/invalid
      * if it trips: `sudo ./power.sh stock`, check case airflow, re-run —
        never continue an INVALID_THERMAL run
- [ ] Optional kernel profiling (short fixed workload only, HIP graphs off,
      never the full suite):
      ```bash
      PROFILE=1 ./benchmark-mi50.sh prof ~/mi50-builds/golden/build /models/Qwen3.8-27B-Q4_0.gguf T6
      # -> results/.../prof/profile/{kernel-summary.csv, raw/*.csv}
      ```
- [ ] NUMA: default `NUMA_MODE=auto` binds CPU + memory to the GPU-local node
      (saved in `meta.json` as `numa.node` / `numa.binding`); set
      `NUMA_MODE=off` to disable. Single-socket boxes bind node 0 (harmless).
- [ ] After the smoke run: inspect `results/YYYY-MM-DD/<run>/summary.md`,
      confirm power caps are uniform, no `INVALID_THERMAL`, T0 clean — then
      proceed to the full A/B plan (README §17).
