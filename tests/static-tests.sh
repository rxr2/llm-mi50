#!/usr/bin/env bash
# static-tests.sh — offline static test battery for the MI50 benchmark stack.
# No GPU, no ROCm, no network needed. Covers the review contract:
#   1  bash -n on every *.sh          5  repeated run-name rejection
#   2  py_compile (all python)        6  verify-isa multi-ELF (synth ELFs + fake objdump)
#   3  Prometheus parser fixture      7  record_cap power-cap FAIL/override
#   4  MTP survival-math fixture      8  contract greps (fusion gate, 29 cols, no chain rule)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PASS=0; FAIL=0
ok()  { echo "[PASS] $1"; PASS=$((PASS+1)); }
bad() { echo "[FAIL] $1"; FAIL=$((FAIL+1)); }
run_check() { # name, cmd...
  local name="$1"; shift
  if "$@" >/tmp/st_out 2>&1; then ok "$name"; else bad "$name"; sed 's/^/    /' /tmp/st_out | tail -20; fi
}

# ---- 1. bash -n -------------------------------------------------------------
n_ok=1
for f in *.sh tests/*.sh; do
  [ -f "$f" ] || continue
  bash -n "$f" || { n_ok=0; bad "bash -n $f"; }
done
[ "$n_ok" = 1 ] && ok "bash -n all shell scripts"

# ---- 2. py_compile ----------------------------------------------------------
if python3 -m py_compile summarize.py spec_metrics.py isa_verify.py 2>/tmp/st_out; then
  ok "py_compile summarize.py spec_metrics.py isa_verify.py"
else
  bad "py_compile"; sed 's/^/    /' /tmp/st_out
fi

# ---- 3. Prometheus parser fixture ------------------------------------------
if python3 - <<'PY'
import sys
sys.path.insert(0, ".")
import spec_metrics as sm

FIXTURE = """
# HELP llamacpp:spec_decode_num_drafts_total drafts
# TYPE llamacpp:spec_decode_num_drafts_total counter
llamacpp:spec_decode_num_drafts_total{slot="0"} 40
llamacpp:spec_decode_num_drafts_total{slot="1"} 60
llamacpp:spec_decode_num_draft_tokens_total{slot="0"} 100
llamacpp:spec_decode_num_draft_tokens_total{slot="1"} 200
llamacpp:spec_decode_num_accepted_tokens_total{slot="0"} 75
llamacpp:spec_decode_num_accepted_tokens_total{slot="1"} 125
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{slot="0",pos="0"} 80
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{slot="0",pos="1"} 50
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{slot="1",pos="0"} 120
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{slot="1",pos="1"} 70
llamacpp:prompt_tokens_total 999999
# TYPE llama_build counter
some_other_metric{pos="1"} 12345
"""
p = sm.parse_prometheus(FIXTURE)
assert p[sm.SPEC_DRAFTS]["total"] == 100, p
assert p[sm.SPEC_DRAFT_TOKENS]["total"] == 300, p
assert p[sm.SPEC_ACCEPTED_TOKENS]["total"] == 200, p
assert p[sm.SPEC_PER_POS]["by_pos"] == {0: 200, 1: 120}, p
assert sm.SPEC_PER_POS not in p or "prompt" not in str(p.keys())

before = sm.parse_prometheus(sm.SPEC_DRAFTS + ' 10\n' + sm.SPEC_DRAFT_TOKENS + ' 50\n'
                             + sm.SPEC_ACCEPTED_TOKENS + ' 30\n'
                             + sm.SPEC_PER_POS + '{pos="0"} 40\n'
                             + sm.SPEC_PER_POS + '{pos="1"} 10\n')
after = sm.parse_prometheus(sm.SPEC_DRAFTS + ' 16\n' + sm.SPEC_DRAFT_TOKENS + ' 68\n'
                            + sm.SPEC_ACCEPTED_TOKENS + ' 45\n'
                            + sm.SPEC_PER_POS + '{pos="0"} 52\n'
                            + sm.SPEC_PER_POS + '{pos="1"} 14\n')
d = sm.counter_delta(before, after)
assert d["verification_steps"] == 6, d
assert d["draft_tokens"] == 18, d
assert d["accepted_tokens"] == 15, d
assert d["per_pos"] == {0: 12, 1: 4}, d
print("prometheus fixture ok")
PY
then ok "Prometheus parser + counter-delta fixture"
else bad "Prometheus parser + counter-delta fixture"; fi

# ---- 4. MTP survival math fixture ------------------------------------------
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
# the old chain rule would have produced P(A=1)=0.8*0.5 style products —
# explicitly assert we did NOT chain-multiply:
assert abs(dist[1] - 0.8 * 0.5) > 1e-9, "chain rule crept back in"

# mean target width = 1 + draft_tokens / verification_steps
m = sm.means_from_delta({"verification_steps": 10, "draft_tokens": 30,
                         "accepted_tokens": 18, "per_pos": {}})
assert abs(m["mean_draft_tokens_per_step"] - 3.0) < 1e-12
assert abs(m["mean_accepted_tokens_per_step"] - 1.8) < 1e-12
assert abs(m["mean_target_width"] - 4.0) < 1e-12
assert sm.means_from_delta({"verification_steps": 0}) is None

# survival_from_delta: unconditional / steps, non-increasing, contiguous
s = sm.survival_from_delta({"verification_steps": 10,
                            "per_pos": {0: 8, 1: 5, 2: 3, 5: 1}})
assert s == [0.8, 0.5, 0.3], s   # pos 5 dropped (gap)
# equal-weight average across requests
avg = sm.average_survival([[1.0, 0.5], [0.5, 0.0]])
assert avg == [0.75, 0.25], avg
print("survival math ok")
PY
then ok "MTP survival-math fixture (no chain rule)"
else bad "MTP survival-math fixture (no chain rule)"; fi

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

# required symbols (same needles as REQUIRED_DEFAULTS) — split over TWO ELFs
needles = [n for _, n in iv.REQUIRED_DEFAULTS]
lds_for = {
    needles[0]: 3072, needles[1]: 3072, needles[2]: 12288,
    needles[3]: 12288, needles[4]: 12288, needles[5]: 9216,
}
# TSV: use the REPO's real TSV (guarantees the loader matches the real format)
tsv = os.path.abspath("build-verification/mmvq-gfx906-kernel-resources.tsv")

def make_elf(path, names, e_flags):
    # minimal ET_REL AMDGPU: [ehdr][.text][.symtab][.strtab][.shstrtab][shdrs]
    text = b""
    syms = [(n, i * 64) for i, n in enumerate(names)]
    for n, off in syms:
        pad = off - len(text)
        if pad > 0: text += b"\0" * pad
        text += struct.pack("<IIII", lds_for[n], 0, 8, 0)   # lds, scratch=0, kernarg
    text += b"\0" * ((64 - len(text) % 64) % 64)
    strtab = b"\0"
    sym_entries = [b"\0" * 24]   # null symbol
    for n, off in syms:
        nm = len(strtab)
        strtab += n.encode() + b"\0"
        # st_name, st_info(STB_GLOBAL|STT_FUNC=0x12), st_other, st_shndx(1), value, size
        sym_entries.append(struct.pack("<IBBHQQ", nm, 0x12, 0, 1, off, 64))
    symtab = b"".join(sym_entries)
    shstr = b"\0.text\0.symtab\0.strtab\0.shstrtab\0"
    # layout
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
        shdr(1, 1, 6, 0, off_text, len(text), 0, 0, 8, 0),            # .text PROGBITS
        shdr(7, 2, 0, 0, off_sym, len(symtab), 3, 1, 8, 24),          # .symtab -> .strtab
        shdr(15, 3, 0, 0, off_str, len(strtab), 0, 0, 1, 0),          # .strtab
        shdr(23, 3, 0, 0, off_shstr, len(shstr), 0, 0, 1, 0),         # .shstrtab
    ])
    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"; ehdr[4] = 2; ehdr[5] = 1; ehdr[6] = 1
    struct.pack_into("<HHIQQQIHHHHHH", ehdr, 16,
                     1, 0xE0, 1, 0, 0, off_shdrs, 0, 64, 0, 0, 64, 5, 4)
    struct.pack_into("<I", ehdr, 48, e_flags)          # e_flags @48 (mach in low 10 bits)
    with open(path, "wb") as f:
        f.write(bytes(ehdr) + text + symtab + strtab + shstr + shdrs)

# ELF A: n1 base+fused (mach ok);  ELF B: n4 + n8 + q5 + q6 (mach ok)
make_elf(os.path.join(elfdir, "a-ok.elf"), needles[0:2], 36)
make_elf(os.path.join(elfdir, "b-ok.elf"), needles[2:6], 36)

# fake objdump: logs argv, prints "<sym>:" lines it was asked for
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

# grouping assertion: one objdump call PER ELF, symbols only from that ELF
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
rc2 = iv.run_verifier(elfdir, os.path.join(d, "out2"), os.path.join(d, "report2.txt"), tsv, fake)
# needles[0] exists in a-ok AND c-bad — verifier still passes via a-ok;
# isolate: remove a-ok so only the bad-flags ELF can satisfy n1-base
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
echo "220"   # default in-range; tests overwrite
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

# ---- 8. contract greps ------------------------------------------------------
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
hdr=$(grep -o 'test,config,workload[^"]*' benchmark-mi50.sh | head -1)
cols=$(printf '%s' "$hdr" | awk -F, '{print NF}')
if [ "$cols" = 29 ]; then ok "requests.csv header has 29 columns"
else bad "requests.csv header has 29 columns (got $cols)"; fi
if grep -nE 'width_estimate\(|surv \*= |\(1 - acc_vec' summarize.py; then
  bad "summarize.py must not chain-rule-multiply acc-per-pos"
else ok "summarize.py has no chain-rule product"; fi
if grep -q 'acc_cfg\[line\[0\]\] = parse_acc_vec' summarize.py; then
  bad "acc_per_pos must not last-wins overwrite"
else ok "acc_per_pos rows are accumulated, not overwritten"; fi
if grep -q 'ALLOW_APPEND=1' benchmark-mi50.sh && grep -q 'ALLOW_POWER_MISMATCH=1' benchmark-mi50.sh; then
  ok "ALLOW_APPEND / ALLOW_POWER_MISMATCH overrides present"
else bad "ALLOW_APPEND / ALLOW_POWER_MISMATCH overrides present"; fi
for t in T0/power T1/power profile/power T9/power_native T10/power_native; do
  if grep -q "mon_start \"\\\$OUT/$t" benchmark-mi50.sh || grep -q "mon_start \"\$OUT/$t" benchmark-mi50.sh; then
    ok "thermal monitor wrap present: $t"
  else
    # fallback: literal match
    if grep -Fq "mon_start \"\$OUT/$t" benchmark-mi50.sh; then
      ok "thermal monitor wrap present: $t"
    else
      bad "thermal monitor wrap present: $t"
    fi
  fi
done
if grep -q '\. "\$HERE/rocm-env.sh"' preflight.sh && \
   grep -q '\. "\$HERE/rocm-env.sh"' build.sh && \
   grep -q '\. "\$HERE/rocm-env.sh"' benchmark-mi50.sh; then
  ok "preflight + build + benchmark all source rocm-env.sh"
else bad "preflight + build + benchmark all source rocm-env.sh"; fi
if grep -q 'rocm-smi missing in ROCm tree' preflight.sh && grep -q 'rocprofv3 missing' preflight.sh; then
  ok "preflight requires rocm-smi + rocprofv3 (READY ⇒ tools in PATH)"
else bad "preflight requires rocm-smi + rocprofv3 (READY ⇒ tools in PATH)"; fi

# ---- summary ----------------------------------------------------------------
echo
echo "static-tests: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
