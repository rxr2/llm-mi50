#!/usr/bin/env bash
# verify-isa.sh — generate gfx906 ISA verification artifacts after a build.
#
#   ./verify-isa.sh <build-dir> [out-dir]        (called automatically by build.sh)
#
# Extracts gfx906 device code objects from the build tree (clang offload
# bundles and embedded AMDGPU ELF), disassembles the required kernels with
# roc-objdump or llvm-objdump, and writes everything to
#   build-verification/disassembly/
#
# Required kernels (at least):
#   q4_0_breit n=1, q4_0_breit n=4, q4_0_breit n=8, q5_K_breit, q6_K_breit
#
# Confirms, without hardware:
#   * the gfx906 ISA for each required kernel was actually generated, and
#   * the default instantiations carry private_segment_fixed_size (scratch) = 0,
#     cross-checked against build-verification/mmvq-gfx906-kernel-resources.tsv.
# NO performance interpretation without the MI50 (that is T0/T1 on hardware).
set -euo pipefail
BUILD="${1:?usage: verify-isa.sh <build-dir> [out-dir]}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${2:-$HERE/build-verification/disassembly}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm-7.1.1}"
TSV="$HERE/build-verification/mmvq-gfx906-kernel-resources.tsv"

die() { echo "ERROR: $*" >&2; exit 1; }
command -v python3 >/dev/null || die "python3 missing"
[ -d "$BUILD" ] || die "build dir $BUILD not found"

# ---------- find an AMDGPU-capable objdump ----------
OBJDUMP=""
for d in "$ROCM_PATH/bin" "$ROCM_PATH/lib/llvm/bin" "$ROCM_PATH/llvm/bin" "$ROCM_PATH/lib/llvm/lib"; do
  [ -x "$d/llvm-objdump" ] && { OBJDUMP="$d/llvm-objdump"; break; }
done
if [ -z "$OBJDUMP" ]; then
  for c in roc-objdump llvm-objdump; do
    command -v "$c" >/dev/null && { OBJDUMP="$(command -v "$c")"; break; }
  done
fi
[ -n "$OBJDUMP" ] || die "roc-objdump/llvm-objdump not found (looked in ROCM_PATH=$ROCM_PATH and PATH). Set ROCM_PATH."

mkdir -p "$OUT"
REPORT="$OUT/isa-verify-report.txt"

# ---------- step 1: extract gfx906 code objects ----------
# Writes ELFs to $OUT/.code-objects/ and the list of extracted files to stdout.
extract_elfs() {
  python3 - "$BUILD" "$OUT/.code-objects" <<'PY'
import os, sys, struct, hashlib, glob

root, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
MAGIC = b"___CLANG_OFFLOAD_BUNDLE___"
seen = set()
n_out = 0

def valid_elf_size(b, off):
    """Return (size, etype, machine_ok) for an ELF64LE at b[off:], or None."""
    if b[off:off+4] != b"\x7fELF" or off + 64 > len(b):
        return None
    if b[off+4] != 2 or b[off+5] != 1:            # ELFCLASS64, ELFDATA2LSB
        return None
    etype = struct.unpack_from("<H", b, off+16)[0]
    machine = struct.unpack_from("<H", b, off+18)[0]
    if machine != 0xE0:                            # EM_AMDGPU
        return None
    e_shoff = struct.unpack_from("<Q", b, off+40)[0]
    e_shentsize = struct.unpack_from("<H", b, off+58)[0]
    e_shnum = struct.unpack_from("<H", b, off+60)[0]
    if e_shoff and e_shentsize and e_shnum:
        size = e_shoff + e_shentsize * e_shnum
    else:
        return None
    if size <= 0 or off + size > len(b):
        return None
    return size, etype

def emit(blob, tag):
    global n_out
    h = hashlib.sha1(blob).hexdigest()[:12]
    if h in seen:
        return
    seen.add(h)
    path = os.path.join(outdir, f"{h}_{tag}.elf")
    with open(path, "wb") as f:
        f.write(blob)
    n_out += 1

# candidate files: mmvq objects first (cheap, most relevant), then the final
# device library, then everything else that may carry device code.
cands = []
for pat in ("*mmvq*", "*mmq*", "libggml-hip*", "*.hsaco", "*.co"):
    cands += glob.glob(os.path.join(root, "**", pat), recursive=True)
cands += glob.glob(os.path.join(root, "**", "*.o"), recursive=True)
cands += glob.glob(os.path.join(root, "**", "*.so"), recursive=True)
cands += glob.glob(os.path.join(root, "**", "*.so.*"), recursive=True)
seen_paths, ordered = set(), []
for p in cands:
    if p in seen_paths or not os.path.isfile(p):
        continue
    seen_paths.add(p)
    ordered.append(p)

for path in ordered[:800]:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        continue
    if len(data) < 64:
        continue
    # 1) clang offload bundles tagged gfx906
    start = 0
    while True:
        i = data.find(MAGIC, start)
        if i < 0:
            break
        try:
            p = i + len(MAGIC)
            num = struct.unpack_from("<Q", data, p)[0]
            p += 8
            if num > 10_000:
                raise ValueError("implausible bundle count")
            for _ in range(num):
                off, size, idlen = struct.unpack_from("<QQQ", data, p)
                p += 24
                idb = data[p:p + idlen]
                p += idlen
                if b"gfx906" not in idb:
                    continue
                blob = data[i + off:i + off + size]
                if blob[:4] != b"\x7fELF":
                    j = blob.find(b"\x7fELF")
                    if j >= 0:
                        blob = blob[j:]
                info = valid_elf_size(blob, 0) if blob[:4] == b"\x7fELF" else None
                if info:
                    emit(blob[:info[0]], "bundle-" + os.path.basename(path))
        except Exception:
            pass
        start = i + 8
    # 2) raw embedded AMDGPU ELF (runtime fatbin inside .so / .o)
    j = 0
    while True:
        j = data.find(b"\x7fELF", j)
        if j < 0:
            break
        info = valid_elf_size(data, j)
        if info:
            emit(data[j:j + info[0]], "embed-" + os.path.basename(path))
            j += info[0]
        else:
            j += 4

print(n_out)
PY
}

echo "== verify-isa: extracting gfx906 code objects from $BUILD" >&2
N_ELF=$(extract_elfs | tail -1)
[ "${N_ELF:-0}" -gt 0 ] || die "no gfx906 code objects found under $BUILD — was the HIP build completed?"
echo "== $N_ELF gfx906 code object(s) extracted" >&2

# ---------- step 2: locate kernels, check scratch, disassemble ----------
RC=0
python3 - "$OUT/.code-objects" "$OUT" "$REPORT" "$TSV" "$OBJDUMP" <<'PY' || RC=$?
import os, sys, struct, subprocess, re

code_dir, out_dir, report_path, tsv_path, objdump = sys.argv[1:6]

# Required kernels (item 13): q4_0_breit n=1/4/8, q5_K_breit, q6_K_breit.
# breit2 kernels are a different symbol (_breit2_) and are intentionally excluded.
REQUIRED = [
    ("q4_0_breit-n1", "mul_mat_vec_q4_0_breit_gfx906ILi1E"),
    ("q4_0_breit-n4", "mul_mat_vec_q4_0_breit_gfx906ILi4E"),
    ("q4_0_breit-n8", "mul_mat_vec_q4_0_breit_gfx906ILi8E"),
    ("q5_K_breit",    "mul_mat_vec_q5_K_breit_gfx906"),
    ("q6_K_breit",    "mul_mat_vec_q6_K_breit_gfx906"),
]

# expected (lds_bytes, scratch_bytes) per symbol from the existing resources TSV
tsv = {}
if os.path.isfile(tsv_path):
    with open(tsv_path) as f:
        hdr = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 5:
                tsv[parts[0]] = (int(parts[3]), int(parts[4]))   # lds, scratch

def tsv_lookup(name):
    """TSV keys are raw Itanium encodings; ELF symbols may carry a _Z prefix."""
    if name in tsv:
        return tsv[name]
    for k, v in tsv.items():
        if k in name:
            return v
    return None

def u16(b, o): return struct.unpack_from("<H", b, o)[0]
def u32(b, o): return struct.unpack_from("<I", b, o)[0]
def u64(b, o): return struct.unpack_from("<Q", b, o)[0]

class Elf:
    def __init__(self, path):
        self.path = path
        self.b = open(path, "rb").read()
        b = self.b
        assert b[:4] == b"\x7fELF" and b[4] == 2 and b[5] == 1
        self.etype = u16(b, 16)
        e_shoff = u64(b, 40)
        self.shentsize = u16(b, 58)
        self.shnum = u16(b, 60)
        self.e_flags = u32(b, 48)
        self.sh = []
        for i in range(self.shnum):
            o = e_shoff + i * self.shentsize
            self.sh.append(dict(
                name_off=u32(b, o), typ=u32(b, o+4), flags=u64(b, o+8),
                addr=u64(b, o+16), off=u64(b, o+24), size=u64(b, o+32),
                link=u32(b, o+40), entsize=u64(b, o+56)))
        # program headers (for vaddr -> file offset on ET_DYN/EXEC)
        e_phoff = u64(b, 32); e_phentsize = u16(b, 54); e_phnum = u16(b, 56)
        self.ph = []
        for i in range(e_phnum):
            o = e_phoff + i * e_phentsize
            self.ph.append(dict(typ=u32(b, o), off=u64(b, o+8),
                                vaddr=u64(b, o+16), filesz=u64(b, o+32), memsz=u64(b, o+40)))

    def symbols(self):
        b = self.b
        out = []
        for s in self.sh:
            if s["typ"] == 2 and s["link"] < len(self.sh):   # SHT_SYMTAB
                strtab = self.sh[s["link"]]
                strb = b[strtab["off"]:strtab["off"]+strtab["size"]]
                entsize = s["entsize"] or 24
                for o in range(s["off"], s["off"] + s["size"], entsize):
                    st_name = u32(b, o)
                    st_value = u64(b, o+8)
                    st_size = u64(b, o+16)
                    st_shndx = u16(b, o+6)
                    end = strb.find(b"\x00", st_name)
                    name = strb[st_name:end].decode("utf-8", "replace")
                    out.append((name, st_value, st_size, st_shndx))
        return out

    def file_offset(self, st_value, st_shndx):
        if self.etype == 1 and 0 < st_shndx < len(self.sh):   # ET_REL: section-relative
            return self.sh[st_shndx]["off"] + st_value
        for p in self.ph:                                     # ET_DYN/EXEC: via PT_LOAD
            if p["typ"] == 1 and p["vaddr"] <= st_value < p["vaddr"] + max(p["memsz"], 1):
                return p["off"] + (st_value - p["vaddr"])
        return None

elfs = []
for fn in sorted(os.listdir(code_dir)):
    if fn.endswith(".elf"):
        try:
            elfs.append(Elf(os.path.join(code_dir, fn)))
        except Exception:
            pass

report = []
def R(line=""):
    report.append(line)
    print(line)

R("ISA VERIFICATION REPORT — gfx906 (generated by verify-isa.sh, no GPU used)")
R(f"code objects scanned: {len(elfs)}")
R(f"required kernels     : {', '.join(p for p, _ in REQUIRED)}")
R()

# find symbols for every required pattern
found = {p: [] for p, _ in REQUIRED}          # pattern -> [(elf, name, value, size, shndx)]
for e in elfs:
    for name, val, size, shndx in e.symbols():
        if "mul_mat_vec_" not in name:
            continue
        if "breit2" in name:
            continue
        for pat, sub in REQUIRED:
            if sub in name:
                found[pat].append((e, name, val, size, shndx))

failures = []
for pat, sub in REQUIRED:
    R(f"--- {pat}  (substring: {sub})")
    syms = found[pat]
    if not syms:
        R("  RESULT: FAIL — no gfx906 symbol generated for this kernel")
        failures.append(f"{pat}: symbol not found in any extracted gfx906 code object")
        R()
        continue
    R(f"  instantiations found: {len(syms)}")
    kdir = os.path.join(out_dir, pat)
    os.makedirs(kdir, exist_ok=True)
    sym_ok = True
    for e, name, val, size, shndx in syms:
        off = e.file_offset(val, shndx)
        # descriptor: group_segment_fixed_size @ +0, private_segment_fixed_size @ +4.
        # Locate it by the known LDS size from the resources TSV (kernel entry
        # symbols have LDS > 0 for every breit kernel); fall back to scanning
        # [-256, +256) around the symbol for a plausible descriptor pair.
        exp_lds, exp_scr = tsv_lookup(name) or (None, None)
        private = None
        where = ""
        if off is not None:
            b = e.b
            cands = []
            if exp_lds:
                cands += [off, off - 64, off - 256]
            cands += [off - 64, off]
            for c in cands:
                if 0 <= c + 12 <= len(b):
                    g, pr = u32(b, c), u32(b, c + 4)
                    if (exp_lds is None or g == exp_lds) and (g % 4 == 0) and (pr % 4 == 0) and pr < 1 << 20:
                        if exp_lds is None or g == exp_lds:
                            private, where = pr, f"descriptor@{c - off:+d}"
                            break
            if private is None and exp_lds:
                for c in range(off - 256, off + 256, 4):
                    if 0 <= c + 8 <= len(b) and u32(b, c) == exp_lds and u32(b, c + 4) % 4 == 0 and u32(b, c + 4) < 1 << 20:
                        private, where = u32(b, c + 4), f"descriptor@{c - off:+d} (scan)"
                        break
        # scratch verdict
        scratch_txt = "UNKNOWN"
        if private is not None:
            scratch_txt = f"{private} B ({where})"
            if exp_scr is not None and private != exp_scr:
                scratch_txt += f" — TSV says {exp_scr} B: MISMATCH"
                failures.append(f"{pat}:{name}: scratch {private} != TSV {exp_scr}")
            elif private != 0:
                failures.append(f"{pat}:{name}: scratch {private} B != 0 for a default instantiation")
        else:
            failures.append(f"{pat}:{name}: could not locate kernel descriptor (scratch unconfirmed)")
        tsv_txt = f"lds={exp_lds} scratch={exp_scr}" if exp_lds is not None else "no TSV row"
        R(f"  * {name}")
        R(f"      source: {os.path.basename(e.path)}  size={size}  tsv: {tsv_txt}")
        R(f"      private_segment (scratch): {scratch_txt}")

    # disassembly
    sym_names = ",".join(name for _, name, _, _, _ in syms)
    dst = os.path.join(kdir, "disassembly.txt")
    ok = False
    try:
        r = subprocess.run([objdump, "-d", f"--disassemble-symbols={sym_names}", e.path],
                           capture_output=True, text=True, timeout=600)
        out = r.stdout
        if any(name in out for _, name, _, _, _ in syms):
            with open(dst, "w") as f:
                f.write(f"# {objdump} -d --disassemble-symbols={sym_names}\n# source: {e.path}\n\n")
                f.write(out)
            ok = True
    except Exception:
        out = ""
    if not ok:
        # fallback: full dump, slice out the symbols by label
        try:
            r = subprocess.run([objdump, "-d", e.path], capture_output=True, text=True, timeout=900)
            full = r.stdout
            with open(dst, "w") as f:
                f.write(f"# {objdump} -d (full dump sliced)  source: {e.path}\n\n")
                for _, name, _, _, _ in syms:
                    m = re.search(rf"^[0-9a-f]+ <{re.escape(name)}>:\n", full, re.M)
                    if not m:
                        continue
                    start = m.start()
                    nxt = re.search(r"^[0-9a-f]+ <", full[m.end():], re.M)
                    end = m.end() + nxt.start() if nxt else len(full)
                    f.write(full[start:end] + "\n")
                    ok = True
        except Exception:
            pass
    if ok:
        R(f"  disassembly: {os.path.relpath(dst, out_dir)}")
        R("  RESULT: OK — gfx906 ISA present")
    else:
        R(f"  RESULT: FAIL — disassembly could not be produced with {objdump}")
        failures.append(f"{pat}: disassembly failed")
    R()

# e_flags sanity (gfx906 machine encoding = 36 = 0x24 in EF_AMDGPU_MACH)
if elfs:
    R(f"code object e_flags: " + ", ".join(f"{os.path.basename(e.path)}=0x{e.e_flags:x}" for e in elfs[:5]))
R()
if failures:
    R("ISA_VERIFY: FAIL")
    for fmsg in failures:
        R(f"  - {fmsg}")
else:
    R("ISA_VERIFY: PASS — required kernels generated for gfx906, scratch = 0 for all defaults.")
    R("NOTE: no performance interpretation without hardware (run T0/T1 on the MI50).")

with open(report_path, "w") as f:
    f.write("\n".join(report) + "\n")
sys.exit(1 if failures else 0)
PY

# keep a pointer next to the artifacts
{
  echo "isa report: $REPORT"
  echo "artifacts : $OUT"
  date -Is
} > "$OUT/INDEX.txt" 2>/dev/null || true

echo "== ISA verification report: $REPORT" >&2
if [ "$RC" != 0 ]; then
  echo "ERROR: ISA verification FAILED — see $REPORT" >&2
  exit 1
fi
echo "== ISA verification PASS (gfx906 ISA present, defaults scratch-free)" >&2
