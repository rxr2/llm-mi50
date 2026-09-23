#!/usr/bin/env bash
# static-tests.sh — offline static test battery for the MI50 benchmark stack.
# No GPU, no ROCm, no network needed. Covers the review contract:
#   1  bash -n on every *.sh            6  verify-isa multi-ELF (synth ELFs + fake objdump)
#   2  py_compile (all python)          7  record_cap power-cap FAIL/override
#   3  REAL {position="N"} Prometheus   8  T0 fake exit 139 without "FAIL" rejected
#      parser fixture + zero deltas     9  ROCm mixed-stack rejection (stack purity)
#   4  MTP survival math + steps-       10 contract greps (fusion gate, 30 cols, no
#      weighted aggregation (10/100)         chain rule, position label, op_rc)
#   5  repeated run-name rejection
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PASS=0; FAIL=0
ok()  { echo "[PASS] $1"; PASS=$((PASS+1)); }
bad() { echo "[FAIL] $1"; FAIL=$((FAIL+1)); }

# ---- 1. bash -n -------------------------------------------------------------
n_ok=1
for f in *.sh tests/*.sh; do
  [ -f "$f" ] || continue
  bash -n "$f" || { n_ok=0; bad "bash -n $f"; }
done
[ "$n_ok" = 1 ] && ok "bash -n all shell scripts"

# ---- 2. py_compile ----------------------------------------------------------
if python3 -m py_compile *.py 2>/tmp/st_out; then
  ok "py_compile *.py ($(ls *.py | tr '\n' ' '))"
else
  bad "py_compile *.py"; sed 's/^/    /' /tmp/st_out
fi

# ---- 3. REAL {position="N"} Prometheus fixture + zero deltas ----------------
if python3 - <<'PY'
import sys
sys.path.insert(0, ".")
import spec_metrics as sm

# Copied from the actual llama-server /metrics exposition format of our
# exact upstream base 42916d83: the label is position="N" (NOT pos="N").
FIXTURE = """
# HELP llamacpp:spec_decode_num_drafts_total drafts
# TYPE llamacpp:spec_decode_num_drafts_total counter
llamacpp:spec_decode_num_drafts_total{slot="0"} 40
llamacpp:spec_decode_num_drafts_total{slot="1"} 60
llamacpp:spec_decode_num_draft_tokens_total{slot="0"} 100
llamacpp:spec_decode_num_draft_tokens_total{slot="1"} 200
llamacpp:spec_decode_num_accepted_tokens_total{slot="0"} 75
llamacpp:spec_decode_num_accepted_tokens_total{slot="1"} 125
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{slot="0",position="0"} 80
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{slot="0",position="1"} 50
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{slot="1",position="0"} 120
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{slot="1",position="1"} 70
llamacpp:prompt_tokens_total 999999
# TYPE llama_build counter
some_other_metric{position="1"} 12345
"""
p = sm.parse_prometheus(FIXTURE)
assert p[sm.SPEC_DRAFTS]["total"] == 100, p
assert p[sm.SPEC_DRAFT_TOKENS]["total"] == 300, p
assert p[sm.SPEC_ACCEPTED_TOKENS]["total"] == 200, p
assert p[sm.SPEC_PER_POS]["by_pos"] == {0: 200, 1: 120}, p   # REAL position= label

# legacy pos= label still accepted (optional compatibility)
legacy = sm.parse_prometheus(sm.SPEC_PER_POS + '{pos="0"} 7\n')
assert legacy[sm.SPEC_PER_POS]["by_pos"] == {0: 7}, legacy

# counter deltas over REAL position labels
before = sm.parse_prometheus(sm.SPEC_DRAFTS + ' 10\n' + sm.SPEC_DRAFT_TOKENS + ' 50\n'
                             + sm.SPEC_ACCEPTED_TOKENS + ' 30\n'
                             + sm.SPEC_PER_POS + '{position="0"} 40\n'
                             + sm.SPEC_PER_POS + '{position="1"} 10\n'
                             + sm.SPEC_PER_POS + '{position="2"} 5\n')
after = sm.parse_prometheus(sm.SPEC_DRAFTS + ' 16\n' + sm.SPEC_DRAFT_TOKENS + ' 68\n'
                            + sm.SPEC_ACCEPTED_TOKENS + ' 45\n'
                            + sm.SPEC_PER_POS + '{position="0"} 52\n'
                            + sm.SPEC_PER_POS + '{position="1"} 14\n'
                            + sm.SPEC_PER_POS + '{position="2"} 5\n')   # pos 2 UNCHANGED
d = sm.counter_delta(before, after)
assert d["verification_steps"] == 6, d
assert d["draft_tokens"] == 18, d
assert d["accepted_tokens"] == 15, d
# ZERO delta for position 2 must be PRESERVED (not dropped)
assert d["per_pos"] == {0: 12, 1: 4, 2: 0}, d
assert sm.per_pos_vector(d["per_pos"]) == [12, 4, 0]
print("prometheus position= fixture + zero deltas ok")
PY
then ok "REAL {position=} Prometheus fixture + zero-delta preservation"
else bad "REAL {position=} Prometheus fixture + zero-delta preservation"; fi

# ---- 4. MTP survival math + steps-weighted aggregation ----------------------
if python3 - <<'PY'
import sys
sys.path.insert(0, ".")
import spec_metrics as sm

# server semantics: r[i] = P(A >= i+1)  (UNCONDITIONAL — never chain-multiply)
r = [0.8, 0.5, 0.25]
dist = sm.accepted_prefix_distribution(r)
assert abs(dist[0] - (1 - 0.8)) < 1e-12, dist          # P(A=0) = 1 - r0
assert abs(dist[1] - (0.8 - 0.5)) < 1e-12, dist         # P(A=k) = r[k-1] - r[k]
assert abs(dist[2] - (0.5 - 0.25)) < 1e-12, dist
assert abs(dist[3] - 0.25) < 1e-12, dist                # P(A=K) = r[K-1]
assert abs(sum(dist) - 1.0) < 1e-12, dist
assert abs(dist[1] - 0.8 * 0.5) > 1e-9, "chain rule crept back in"

# mean target width = 1 + draft_tokens / verification_steps
m = sm.means_from_delta({"verification_steps": 10, "draft_tokens": 30,
                         "accepted_tokens": 18, "per_pos": {}})
assert abs(m["mean_draft_tokens_per_step"] - 3.0) < 1e-12
assert abs(m["mean_accepted_tokens_per_step"] - 1.8) < 1e-12
assert abs(m["mean_target_width"] - 4.0) < 1e-12
assert sm.means_from_delta({"verification_steps": 0}) is None

# survival_from_delta keeps zero-count trailing positions
s = sm.survival_from_delta({"verification_steps": 10,
                            "per_pos": {0: 8, 1: 5, 2: 0, 5: 1}})
assert s == [0.8, 0.5, 0.0], s   # zero at pos 2 kept; gap at 5 truncates

# EXACT steps-weighted aggregation (review 2):
#   request A = 10 steps, counts [8]  -> r = 0.8
#   request B = 100 steps, counts [50] -> r = 0.5
#   weighted  = (8+50)/(10+100) = 58/110 ~= 0.52727
#   equal-weight would be (0.8+0.5)/2 = 0.65 — MUST NOT be used
rw = sm.weighted_survival([[8], [50]], [10, 100])
assert len(rw) == 1
assert abs(rw[0] - 58 / 110) < 1e-12, rw
assert abs(rw[0] - 0.65) > 1e-9, "equal-weight crept back in"
# a 100-step request weighs 10x a 10-step one — sanity via scale invariance
rw2 = sm.weighted_survival([[8], [50]], [10, 100])
assert rw2 == rw
# vector fallback (survival_per_pos only) uses the same weighting
rv = sm.weighted_survival_vectors([[0.8], [0.5]], [10, 100])
assert abs(rv[0] - 58 / 110) < 1e-12, rv
# position exposure: a request without position 1 does not vote on it
rw3 = sm.weighted_survival([[8, 4], [50]], [10, 100])
assert abs(rw3[1] - 4 / 10) < 1e-12, rw3
# zero-delta-only request contributes 0 counts, full steps denominator
rw4 = sm.weighted_survival([[8, 0], [50, 0]], [10, 100])
assert abs(rw4[1] - 0.0) < 1e-12, rw4
# equal-weight helper exists ONLY for the steps-less server-trace fallback
assert sm.average_survival([[1.0], [0.0]]) == [0.5]
print("survival math + steps-weighted aggregation ok")
PY
then ok "MTP survival math + verification_steps-weighted aggregation (10/100)"
else bad "MTP survival math + verification_steps-weighted aggregation (10/100)"; fi

# ---- 5. repeated run-name rejection ----------------------------------------
RUNNAME="static-iso-$$"
cleanup() { rm -rf "$HERE/results"/2*/"$RUNNAME"* 2>/dev/null || true; }
trap cleanup EXIT
DRY=0
DRY_RUN=1 ./benchmark-mi50.sh "$RUNNAME" ./prompts /dev/null T10 >/tmp/st_out 2>&1
if [ $? -eq 0 ] && [ -f "$HERE/results/$(date +%F)/$RUNNAME/requests.csv" ]; then
  DRY=1; ok "first DRY_RUN run created data"
else
  bad "first DRY_RUN run"; tail -15 /tmp/st_out | sed 's/^/    /'
fi
if [ "$DRY" = 1 ]; then
  if DRY_RUN=1 ./benchmark-mi50.sh "$RUNNAME" ./prompts /dev/null T10 >/tmp/st_out 2>&1; then
    bad "second run with same name must FAIL (data present)"
  elif grep -q "already contains benchmark data" /tmp/st_out; then
    ok "second run with same name rejected (ALLOW_APPEND not set)"
  else
    bad "second run failed for the wrong reason"; tail -10 /tmp/st_out | sed 's/^/    /'
  fi
  if ALLOW_APPEND=1 DRY_RUN=1 ./benchmark-mi50.sh "$RUNNAME" ./prompts /dev/null T10 >/tmp/st_out 2>&1; then
    ok "ALLOW_APPEND=1 override accepted"
  else
    bad "ALLOW_APPEND=1 override accepted"; tail -10 /tmp/st_out | sed 's/^/    /'
  fi
fi

# ---- 6. verify-isa multi-ELF (synthesized ELFs + fake objdump) --------------
VDIR=$(mktemp -d /tmp/isa-static-XXXX)
if python3 - "$VDIR" <<'PY'
import os, struct, subprocess, sys, shutil
import importlib.util
spec = importlib.util.spec_from_file_location("isa_verify", "isa_verify.py")
iv = importlib.util.module_from_spec(spec); spec.loader.exec_module(iv)

d = sys.argv[1]
elfdir = os.path.join(d, "elfs"); out = os.path.join(d, "out")
os.makedirs(elfdir); os.makedirs(out)

needles = [n for _, n in iv.REQUIRED_DEFAULTS]
lds_for = {
    needles[0]: 3072, needles[1]: 3072, needles[2]: 12288,
    needles[3]: 12288, needles[4]: 12288, needles[5]: 9216,
}
tsv = os.path.abspath("build-verification/mmvq-gfx906-kernel-resources.tsv")

def make_elf(path, names, e_flags):
    text = b""
    syms = [(n, i * 64) for i, n in enumerate(names)]
    for n, off in syms:
        pad = off - len(text)
        if pad > 0: text += b"\0" * pad
        text += struct.pack("<IIII", lds_for[n], 0, 8, 0)   # lds, scratch=0, kernarg
    text += b"\0" * ((64 - len(text) % 64) % 64)
    strtab = b"\0"
    sym_entries = [b"\0" * 24]
    for n, off in syms:
        nm = len(strtab)
        strtab += n.encode() + b"\0"
        sym_entries.append(struct.pack("<IBBHQQ", nm, 0x12, 0, 1, off, 64))
    symtab = b"".join(sym_entries)
    shstr = b"\0.text\0.symtab\0.strtab\0.shstrtab\0"
    ehdr_sz, shdr_sz = 64, 64
    off_text = ehdr_sz
    off_sym = off_text + len(text)
    off_str = off_sym + len(symtab)
    off_shstr = off_str + len(strtab)
    off_shdrs = off_shstr + len(shstr)
    def shdr(name_off, typ, flags, addr, offset, size, link, info, align, entsz):
        return struct.pack("<IIQQQQIIQQ", name_off, typ, flags, addr, offset, size, link, info, align, entsz)
    shdrs = b"".join([
        b"\0" * 64,
        shdr(1, 1, 6, 0, off_text, len(text), 0, 0, 8, 0),
        shdr(7, 2, 0, 0, off_sym, len(symtab), 3, 1, 8, 24),
        shdr(15, 3, 0, 0, off_str, len(strtab), 0, 0, 1, 0),
        shdr(23, 3, 0, 0, off_shstr, len(shstr), 0, 0, 1, 0),
    ])
    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"; ehdr[4] = 2; ehdr[5] = 1; ehdr[6] = 1
    struct.pack_into("<HHIQQQIHHHHHH", ehdr, 16,
                     1, 0xE0, 1, 0, 0, off_shdrs, 0, 64, 0, 0, 64, 5, 4)
    struct.pack_into("<I", ehdr, 48, e_flags)
    with open(path, "wb") as f:
        f.write(bytes(ehdr) + text + symtab + strtab + shstr + shdrs)

make_elf(os.path.join(elfdir, "a-ok.elf"), needles[0:2], 36)
make_elf(os.path.join(elfdir, "b-ok.elf"), needles[2:6], 36)

fake = os.path.join(d, "objdump")
logf = os.path.join(d, "objdump.log")
with open(fake, "w") as f:
    f.write(f"""#!/usr/bin/env bash
echo "$@" >> {logf}
syms=""
for a in "$@"; do case "$a" in --disassemble-symbols=*) syms="${{a#*=}}";; esac; done
file="${{@: -1}}"
IFS=',' read -ra arr <<< "$syms"
for s in "${{arr[@]}}"; do echo "0000000000000000 <$s>:"; done
exit 0
""")
os.chmod(fake, 0o755)

report = os.path.join(d, "report.txt")
rc = iv.run_verifier(elfdir, out, report, tsv, fake)
assert rc == 0, "expected PASS for 2 correct ELFs, rc=%s\n%s" % (rc, open(report).read())

calls = [l.strip() for l in open(logf) if l.strip()]
assert len(calls) == 2, "expected 2 objdump calls (one per ELF), got: %r" % calls
def syms_of(call):
    for a in call.split():
        if a.startswith("--disassemble-symbols="):
            return set(a.split("=", 1)[1].split(","))
    return set()
a_call = next(c for c in calls if c.endswith("a-ok.elf"))
b_call = next(c for c in calls if c.endswith("b-ok.elf"))
assert syms_of(a_call) == set(needles[0:2]), (a_call, syms_of(a_call))
assert syms_of(b_call) == set(needles[2:6]), (b_call, syms_of(b_call))

# FAIL scenario 1: hosting ELF has wrong e_flags (mach != 36) -> must FAIL
bad_elf = os.path.join(elfdir, "c-bad-flags.elf")
make_elf(bad_elf, [needles[0]], 28)
os.rename(os.path.join(elfdir, "a-ok.elf"), os.path.join(d, "a-ok.elf.saved"))
rc3 = iv.run_verifier(elfdir, os.path.join(d, "out3"), os.path.join(d, "report3.txt"), tsv, fake)
rep3 = open(os.path.join(d, "report3.txt")).read()
assert rc3 != 0, "expected FAIL when the only host ELF has mach != 36"
assert "GFX906" in rep3 and "e_flags" in rep3, rep3

# FAIL scenario 2: missing symbol -> must FAIL
os.remove(bad_elf)
rc4 = iv.run_verifier(elfdir, os.path.join(d, "out4"), os.path.join(d, "report4.txt"), tsv, fake)
rep4 = open(os.path.join(d, "report4.txt")).read()
assert rc4 != 0 and "not found" in rep4, rep4
print("isa multi-elf ok")
PY
then ok "verify-isa multi-ELF (grouping + e_flags FAIL + missing symbol FAIL)"
else bad "verify-isa multi-ELF"; fi
rm -rf "$VDIR"

# ---- 7. record_cap power-cap FAIL/override ---------------------------------
RC_DIR=$(mktemp -d /tmp/reccap-XXXX)
cat > "$RC_DIR/fake-smi" <<'FAKE'
#!/usr/bin/env bash
echo "220"
FAKE
chmod +x "$RC_DIR/fake-smi"
sed -n '/^record_cap() {/,/^}/p' benchmark-mi50.sh > "$RC_DIR/rc.sh"
harness() { # $1 = cap value, rest = env assignments prefix
  local cap="$1"; shift
  echo "#!/usr/bin/env bash
echo \"$cap\"" > "$RC_DIR/fake-smi"
  chmod +x "$RC_DIR/fake-smi"
  env "$@" DRY_RUN=0 OUT="$RC_DIR" LOG="$RC_DIR/bench.log" \
      SMI="$RC_DIR/fake-smi" GPU=0 POWER_CAP_MIN=220 POWER_CAP_MAX=226 \
      bash -c 'log(){ echo "[T] $*"; }; source '"$RC_DIR"'/rc.sh; record_cap T1 testcfg' \
      >/tmp/st_out 2>&1
  return $?
}
if harness 225; then ok "record_cap in-range 225 W passes"
else bad "record_cap in-range 225 W passes"; tail -5 /tmp/st_out | sed 's/^/    /'; fi
if harness 99; then bad "record_cap 99 W must FAIL (outside 220-226)"
else ok "record_cap 99 W fails without override"; fi
if grep -q "outside stock range" /tmp/st_out; then ok "record_cap FAIL message names the range"
else bad "record_cap FAIL message names the range"; fi
if harness 99 ALLOW_POWER_MISMATCH=1; then ok "record_cap ALLOW_POWER_MISMATCH=1 override"
else bad "record_cap ALLOW_POWER_MISMATCH=1 override"; tail -5 /tmp/st_out | sed 's/^/    /'; fi
rm -rf "$RC_DIR"

# ---- 8. T0 fake exit 139 without "FAIL" text must be rejected --------------
# Extract the REAL T0 op loop from benchmark-mi50.sh and execute it against a
# fake test-backend-ops. The old `cmd > file; tail file` chain returned tail's
# status — exit 139 with no literal "FAIL" must still STOP the run.
T0_CODE="$(awk '/mon_start "\$OUT\/T0\/power\.csv"/{f=1} f{print} /text FAIL in op output/{exit}' benchmark-mi50.sh)"
if [ -z "$T0_CODE" ]; then
  bad "T0 exit-code test could not extract the T0 loop"
else
  T0D=$(mktemp -d /tmp/t0rc-XXXX)
  mkdir -p "$T0D/bin" "$T0D/out/T0"
  printf '#!/usr/bin/env bash\necho "op exploded silently"\nexit 139\n' > "$T0D/bin/test-backend-ops"
  chmod +x "$T0D/bin/test-backend-ops"
  t0_run() (
    set -euo pipefail
    OUT="$T0D/out"; LOG="$OUT/bench.log"; : > "$LOG"
    DRY_RUN=0; NPFX=""; BIN="$T0D/bin"
    log() { echo "[T] $*" | tee -a "$LOG" >&2; }
    mon_start() { :; }; mon_stop() { :; }; check_thermal() { :; }
    eval "$T0_CODE"
    echo T0LOOP_SURVIVED
  )
  if t0_run >"$T0D/run1.log" 2>&1; then
    bad "T0 exit 139 without 'FAIL' text must be rejected (loop survived)"
    tail -10 "$T0D/run1.log" | sed 's/^/    /'
  elif grep -q "exited rc=139" "$T0D/run1.log" && ! grep -q T0LOOP_SURVIVED "$T0D/run1.log"; then
    ok "T0 exit 139 without 'FAIL' text is rejected (rc=139 captured, run stopped)"
  else
    bad "T0 exit 139 rejection message"; tail -10 "$T0D/run1.log" | sed 's/^/    /'
  fi
  # positive control: exit 0 (no FAIL) passes the loop
  printf '#!/usr/bin/env bash\necho "all good"\nexit 0\n' > "$T0D/bin/test-backend-ops"
  chmod +x "$T0D/bin/test-backend-ops"
  if t0_run >"$T0D/run2.log" 2>&1 && grep -q T0LOOP_SURVIVED "$T0D/run2.log"; then
    ok "T0 exit 0 passes (positive control)"
  else
    bad "T0 exit 0 passes (positive control)"; tail -10 "$T0D/run2.log" | sed 's/^/    /'
  fi
  # text FAIL with exit 0 must also stop
  printf '#!/usr/bin/env bash\necho "FAIL: bad value"\nexit 0\n' > "$T0D/bin/test-backend-ops"
  chmod +x "$T0D/bin/test-backend-ops"
  if t0_run >"$T0D/run3.log" 2>&1; then
    bad "T0 text FAIL with exit 0 must be rejected"
  elif grep -q "text FAIL in op output" "$T0D/run3.log"; then
    ok "T0 text FAIL with exit 0 is rejected"
  else
    bad "T0 text FAIL rejection message"; tail -10 "$T0D/run3.log" | sed 's/^/    /'
  fi
  rm -rf "$T0D"
fi

# ---- 9. ROCm mixed-stack rejection (stack purity) --------------------------
T=$(mktemp -d /tmp/rcm-tree-XXXX); P=$(mktemp -d /tmp/rcm-path-XXXX)
mkdir -p "$T/bin" "$T/opt-tools"
printf '#!/usr/bin/env bash\necho "hipconfig v9.9"\n' > "$T/bin/hipconfig"
printf '#!/usr/bin/env bash\necho "in-tree smi"\n' > "$T/opt-tools/rocm-smi"
chmod +x "$T/bin/hipconfig" "$T/opt-tools/rocm-smi"
# foreign tool on PATH (regular file, belongs to NO stack under $T) -> reject
printf '#!/usr/bin/env bash\necho foreign\n' > "$P/rocprofv3"; chmod +x "$P/rocprofv3"
# system wrapper/symlink on PATH that resolves INTO the selected stack -> accept
ln -s "$T/opt-tools/rocm-smi" "$P/rocm-smi"
if out=$(
  . ./rocm-env.sh
  ROCM_PATH="$T" PATH="$P:$PATH" rocm_env_resolve || exit 9
  h=$(rocm_tool hipconfig) || exit 1
  [ "$h" = "$T/bin/hipconfig" ] || { echo "bad in-tree: $h"; exit 2; }
  if rocm_tool rocprofv3 >/dev/null 2>&1; then
    echo "foreign rocprofv3 was accepted"; exit 3
  fi
  [ -z "${ROCPROFV3:-}" ] || { echo "resolve leaked foreign ROCPROFV3=$ROCPROFV3"; exit 4; }
  s=$(rocm_tool rocm-smi) || { echo "wrapper not accepted"; exit 5; }
  [ "$s" = "$T/opt-tools/rocm-smi" ] || { echo "wrapper resolved to $s"; exit 6; }
  echo STACK_PURITY_OK
); then
  if printf '%s' "$out" | grep -q STACK_PURITY_OK; then
    ok "ROCm mixed-stack rejection (foreign PATH tool rejected, in-stack wrapper accepted)"
  else
    bad "ROCm mixed-stack rejection output"; printf '%s\n' "$out" | sed 's/^/    /'
  fi
else
  bad "ROCm mixed-stack rejection (rc=$?)"; printf '%s\n' "$out" | sed 's/^/    /'
fi
rm -rf "$T" "$P"

# ---- 10. contract greps -----------------------------------------------------
if grep -q 'MUL_MAT_VEC_FUSION.*|| true' build.sh; then
  bad "build.sh fusion gate must not have '|| true'"
else
  ok "build.sh fusion gate has no '|| true' escape"
fi
if grep -q 'test -o MUL_MAT_VEC_FUSION' build.sh; then
  ok "build.sh runs the fusion op as a gate"
else bad "build.sh runs the fusion op as a gate"; fi
if grep -q 'for op in MUL_MAT MUL_MAT_ID FLASH_ATTN_EXT GATED_DELTA_NET MUL_MAT_VEC_FUSION' benchmark-mi50.sh; then
  ok "benchmark T0 includes MUL_MAT_VEC_FUSION"
else bad "benchmark T0 includes MUL_MAT_VEC_FUSION"; fi
if grep -q 'op_rc=\$?' benchmark-mi50.sh; then
  ok "T0 captures test-backend-ops own exit code (op_rc=\$?)"
else bad "T0 captures test-backend-ops own exit code (op_rc=\$?)"; fi
hdr=$(grep -o 'test,config,workload[^"]*' benchmark-mi50.sh | head -1)
cols=$(printf '%s' "$hdr" | awk -F, '{print NF}')
if [ "$cols" = 30 ]; then ok "requests.csv header has 30 columns"
else bad "requests.csv header has 30 columns (got $cols)"; fi
if printf '%s' "$hdr" | grep -q 'per_pos_counts' ; then
  ok "requests.csv header carries per_pos_counts"
else bad "requests.csv header carries per_pos_counts"; fi
if grep -qE 'width_estimate\(|surv \*= |\(1 - acc_vec' summarize.py; then
  bad "summarize.py must not chain-rule-multiply acc-per-pos"
else ok "summarize.py has no chain-rule product"; fi
if grep -q 'acc_cfg\[line\[0\]\] = parse_acc_vec' summarize.py; then
  bad "acc_per_pos must not last-wins overwrite"
else ok "acc_per_pos rows are accumulated, not overwritten"; fi
if grep -q 'weighted_survival' summarize.py && grep -q 'steps-weighted' summarize.py; then
  ok "summarize aggregates with verification_steps weighting"
else bad "summarize aggregates with verification_steps weighting"; fi
if grep -q 'position' spec_metrics.py && grep -q '_POS_LABEL_LEGACY' spec_metrics.py; then
  ok "spec_metrics parses REAL position= label (plus legacy pos=)"
else bad "spec_metrics parses REAL position= label (plus legacy pos=)"; fi
if grep -q 'per_pos_counts' benchmark-mi50.sh; then
  ok "request() stores raw per_pos_counts"
else bad "request() stores raw per_pos_counts"; fi
if grep -q 'ALLOW_APPEND=1' benchmark-mi50.sh && grep -q 'ALLOW_POWER_MISMATCH=1' benchmark-mi50.sh; then
  ok "ALLOW_APPEND / ALLOW_POWER_MISMATCH overrides present"
else bad "ALLOW_APPEND / ALLOW_POWER_MISMATCH overrides present"; fi
for t in T0/power T1/power profile/power T9/power_native T10/power_native; do
  if grep -Fq "mon_start \"\$OUT/$t" benchmark-mi50.sh; then
    ok "thermal monitor wrap present: $t"
  else
    bad "thermal monitor wrap present: $t"
  fi
done
if grep -q '\. "\$HERE/rocm-env.sh"' preflight.sh && \
   grep -q '\. "\$HERE/rocm-env.sh"' build.sh && \
   grep -q '\. "\$HERE/rocm-env.sh"' benchmark-mi50.sh; then
  ok "preflight + build + benchmark all source rocm-env.sh"
else bad "preflight + build + benchmark all source rocm-env.sh"; fi
if grep -q 'rocm-smi missing in ROCm tree' preflight.sh && grep -q 'rocprofv3 does not start' preflight.sh \
   && grep -q 'hipconfig --version does not run' preflight.sh; then
  ok "preflight EXECUTES hipconfig/rocprofv3/rocm-smi (not just locate)"
else bad "preflight EXECUTES hipconfig/rocprofv3/rocm-smi (not just locate)"; fi
if grep -q 'command -v "\$n"' rocm-env.sh && grep -q 'readlink -f' rocm-env.sh; then
  ok "rocm-env validates PATH hits with readlink -f (stack purity)"
else bad "rocm-env validates PATH hits with readlink -f (stack purity)"; fi
if grep -q 'extra-libs' rocm-env.sh; then
  ok "rocm-env adds \$ROCM_PATH/extra-libs (Debian libdw) to LD_LIBRARY_PATH"
else bad "rocm-env adds \$ROCM_PATH/extra-libs (Debian libdw) to LD_LIBRARY_PATH"; fi

# ---- summary ----------------------------------------------------------------
echo
echo "static-tests: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
