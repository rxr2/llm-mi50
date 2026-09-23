#!/usr/bin/env bash
# verify-isa.sh — post-build gfx906 ISA verification
# Finds the built llama-server, extracts AMDGPU ELF code objects from it
# (core + embedded bundles), verifies the REQUIRED default breit kernel
# symbols via isa_verify.py (per-ELF grouping, gfx906 e_flags is a PASS
# condition, scratch=0 against the TSV defaults), and stores disassembly
# + a JSON report under build-verification/.
# Standalone/CI: full=false ./verify-isa.sh <path-to-binary>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BINARY="${1:-}"
FULL=false
[ -n "$BINARY" ] && FULL=true

find_binary() {
  if [ -n "${BIN:-}" ]; then
    printf '%s\n' "$BIN"
    return 0
  fi
  local c
  for c in \
    "$HERE/llama.cpp/build/bin/llama-server" \
    "$HERE/llama.cpp/build/bin/llama-mtp-server" \
    "$HERE/llama.cpp/build/bin/llama-cli"; do
    if [ -f "$c" ]; then
      printf '%s\n' "$c"
      return 0
    fi
  done
  return 1
}

if [ "$FULL" = false ]; then
  if ! BINARY="$(find_binary)"; then
    echo "verify-isa.sh: no llama-server binary found — run ./build.sh first" >&2
    exit 1
  fi
fi
if [ ! -f "$BINARY" ]; then
  echo "verify-isa.sh: binary not found: $BINARY" >&2
  exit 1
fi

# --- AMDGPU code objects: extract from the core ELF + every embedded bundle.
# ROCm 7.1 defaults to one bundled object per arch (no -fembed-code-object-split),
# so we must read the embedded objects, not only the core sections.
extract_elfs() {
  python3 - "$1" "$2" <<'PY'
import sys, struct, os
binary, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
# ELF header data
EI_CLASS = 4; EI_DATA = 5
ET_EXEC, ET_DYN, ET_REL = 2, 3, 1
SHT_PROGBITS, SHT_NOBITS = 1, 8
EM_AMDGPU = 0xE0
# ELF symbol table entry (64-bit): name, info, other, shndx, value, size
SYM_FMT = "<IBBHQQ"; SYM_SIZE = struct.calcsize(SYM_FMT)

def u16(b, off, le): return int.from_bytes(b[off:off+2], "little" if le else "big")
def u32(b, off, le): return int.from_bytes(b[off:off+4], "little" if le else "big")
def u64(b, off, le): return int.from_bytes(b[off:off+8], "little" if le else "big")

def parse_elf(data):
    if data[:4] != b"\x7fELF" or len(data) < 64 or data[4] != 2:
        return None
    le = (data[EI_DATA] == 1)
    e_type = u16(data, 16, le)
    if e_type not in (ET_EXEC, ET_DYN, ET_REL):
        return None
    e_machine = u16(data, 18, le)
    e_shoff = u64(data, 40, le)
    e_shentsize = u16(data, 58, le)
    e_shnum = u16(data, 60, le)
    e_shstrndx = u16(data, 62, le)
    if e_shoff == 0 or e_shnum == 0 or e_shentsize < 64:
        return None
    sections = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        if off + 64 > len(data):
            break
        sh_name = u32(data, off, le)
        sh_type = u32(data, off + 4, le)
        sh_offset = u64(data, off + 24, le)
        sh_size = u64(data, off + 32, le)
        sh_link = u32(data, off + 40, le)
        sh_entsize = u64(data, off + 56, le)
        sections.append(dict(name_off=sh_name, type=sh_type,
                             offset=sh_offset, size=sh_size,
                             link=sh_link, entsize=sh_entsize))
    # section name string table
    shstr = b""
    if e_shstrndx < len(sections):
        s = sections[e_shstrndx]
        shstr = data[s["offset"]:s["offset"] + s["size"]]
    def sec_name(s):
        end = shstr.find(b"\0", s["name_off"])
        return shstr[s["name_off"]:end].decode("ascii", "replace") if end >= 0 else ""
    return dict(data=data, le=le, machine=e_machine,
                sections=sections, sec_name=sec_name)

def amd_symbols(elf):
    """Return (global_func_names, workgroup_descriptor_offsets) for AMDGPU."""
    names, descs = [], []
    data = elf["data"]; le = elf["le"]
    for s in elf["sections"]:
        if s["type"] != 2:  # SHT_SYMTAB
            continue
        if s["entsize"] < SYM_SIZE:
            continue
        link = elf["sections"][s["link"]] if s["link"] < len(elf["sections"]) else None
        if link is None:
            continue
        strtab = data[link["offset"]:link["offset"] + link["size"]]
        for off in range(s["offset"], s["offset"] + s["size"], s["entsize"]):
            if off + SYM_SIZE > len(data):
                break
            st_name, st_info, _st_other, st_shndx, st_value, _st_size = \
                struct.unpack_from(SYM_FMT, data, off)
            if st_name == 0:
                continue
            end = strtab.find(b"\0", st_name)
            nm = strtab[st_name:end].decode("ascii", "replace") if end >= 0 else ""
            bind = st_info >> 4; typ = st_info & 0xF
            if nm and bind == 1 and typ == 2:  # STB_GLOBAL + STT_FUNC
                names.append(nm)
            # workgroup descriptor (group, private, kernarg) at symbol value
            if nm and typ == 2 and st_shndx != 0 and st_value >= 0:
                sec = None
                if st_shndx < len(elf["sections"]):
                    sec = elf["sections"][st_shndx]
                if sec and sec["type"] == SHT_PROGBITS:
                    base = sec["offset"] + st_value
                    if base + 16 <= len(data):
                        g = u32(data, base, le)
                        if 0 < g <= 131072 and (g % 64) == 0:
                            descs.append((nm, base))
    return names, descs

elf_count = 0; amd_count = 0; amd_names = 0
seen_ranges = []; written = []
def consider(data, tag):
    global elf_count, amd_count, amd_names
    elf = parse_elf(data)
    if not elf:
        return
    elf_count += 1
    if elf["machine"] != EM_AMDGPU:
        return
    names, descs = amd_symbols(elf)
    if not names and not descs:
        return
    amd_count += 1; amd_names += len(names)
    fn = os.path.join(out_dir, "amd-%s.elf" % tag)
    with open(fn, "wb") as f:
        f.write(data)
    written.append(fn)
    with open(fn + ".syms", "w") as f:
        for n in sorted(set(names)):
            f.write(n + "\n")
    seen_ranges.append((0, len(data)))

with open(binary, "rb") as f:
    blob = f.read()
consider(blob, "core")
# nested AMDGPU bundles in PT_NOTE / SHT_NOTE segments (ROCM note name "GPU")
# and raw .note/ bundle sections; both gcc/clang embed formats.
off = 0
while True:
    i = blob.find(b"GPU", off)
    if i < 0:
        break
    # clang embed bundle: note starts 4 bytes before name ("GPU\0" at +4 in n_namesz)
    # try a few alignments around the marker
    for start in (i - 4, i - 8, i - 12):
        if start < 0 or start + 16 >= len(blob):
            continue
        for hdr in (32, 64):
            end = start + hdr + int.from_bytes(blob[start:start+4], "little") \
                  + int.from_bytes(blob[start+4:start+8], "little")
            if end > len(blob) or end <= start + hdr:
                continue
            cand = blob[start + hdr:end] if start + hdr + 4 <= end else b""
            if cand[:4] == b"\x7fELF":
                consider(cand, "note%d" % len(written))
    off = i + 3
# any other embedded \x7fELF blobs (bundle containers vary by toolchain)
off = 0
while True:
    i = blob.find(b"\x7fELF", off)
    if i < 0:
        break
    if i != 0:
        consider(blob[i:], "embed%d" % len(written))
    off = i + 4
    if len(written) > 64:
        break

print("objects: files=%d amd_gpu=%d symbols=%d extracted=%d"
      % (elf_count, amd_count, amd_names, len(written)))
if amd_count == 0:
    sys.exit(1)
PY
}

# --- objdump: prefer ROCm tools; fall back to PATH / llvm.
find_objdump() {
  local cands=()
  if [ -n "${ROCM_PATH:-}" ]; then
    cands+=("$ROCM_PATH/bin/roc-objdump" "$ROCM_PATH/llvm/bin/llvm-objdump" \
            "$ROCM_PATH/bin/llvm-objdump")
  fi
  cands+=(roc-objdump llvm-objdump)
  local c
  for c in "${cands[@]}"; do
    if command -v "$c" >/dev/null 2>&1; then command -v "$c"; return 0; fi
  done
  return 1
}

OBJDUMP="$(find_objdump)" || { echo "verify-isa.sh: no objdump (roc-objdump/llvm-objdump)" >&2; exit 1; }

BUILD="$HERE/build"
OUT="$HERE/build-verification"
TSV="$HERE/build-verification/mmvq-gfx906-kernel-resources.tsv"
[ -f "$TSV" ] || TSV="$HERE/build/mmvq-gfx906-kernel-resources.tsv"
BIN_DIR="$BUILD/bin"
EC=0

mkdir -p "$OUT/disassembly" "$OUT/.code-objects"

echo "extracting AMDGPU code objects from $BINARY ..."
set +e
EXTRACT_MSG="$(extract_elfs "$BINARY" "$OUT/.code-objects" 2>&1)"
EXTRACT_RC=$?
set -e
echo "$EXTRACT_MSG"
if [ $EXTRACT_RC -ne 0 ]; then
  echo "verify-isa: no AMDGPU code objects found in $(basename "$BINARY")"
  echo "verify-isa: make sure llama.cpp was built with HIP/AMDGPU target (gfx906)"
  EC=1
fi

# Step 2: per-ELF symbol verification (extracted into isa_verify.py so the
# multi-ELF logic, gfx906 e_flags PASS condition and TSV scratch defaults
# are statically testable).
set +e
python3 "$HERE/isa_verify.py" "$OUT/.code-objects" "$OUT" "$OUT/isa-report.txt" \
  "$TSV" "$OBJDUMP"
VERIFY_RC=$?
set -e
if [ $VERIFY_RC -ne 0 ]; then EC=1; fi

if [ $EC -eq 0 ]; then
  # JSON report
  python3 - "$OUT" "$BINARY" "$FULL" > "$OUT/isa-report.json" <<'PY'
import json, sys, glob, os
out, binary, full = sys.argv[1], sys.argv[2], sys.argv[3]
dis = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out, "disassembly", "*")))
kernel_targets = ["q4_0_breit n=1", "q4_0_breit n=4", "q4_0_breit n=8",
                  "q5_K_breit", "q6_K_breit"]
json.dump({
    "kernel_targets": kernel_targets,
    "found": True,
    "gfx906_mach": True,
    "scratch_zero": True,
    "disassembly_files": dis,
    "binary": os.path.basename(binary),
    "full_check": full == "true",
}, sys.stdout, indent=2)
print()
PY
  mkdir -p "$OUT/disassembly"
  cp -f "$OUT/.code-objects"/*.txt "$OUT/disassembly/" 2>/dev/null || true
  if [ "$FULL" = true ]; then
    # Full mode (standalone/CI): keep exact resource tables, drop bulky
    # instruction dumps so the whole tree stays below git/GitHub limits.
    rm -rf "$OUT/.code-objects"
    find "$OUT/disassembly" -type f ! -name 'README.md' -delete 2>/dev/null || true
    echo "verify-isa: PASS (full=true — disassembly purged; tables + isa-report.json kept)"
    echo "verify-isa: all required default symbols present; gfx906=OK; scratch=0 (TSV defaults)"
  else
    echo "verify-isa: PASS — all required default symbols present"
    echo "verify-isa: gfx906=OK e_flags=PASS; scratch=0 vs TSV; disassembly/isa-report.json written"
  fi
else
  echo "verify-isa: FAIL (rc=$EC) — see $OUT/isa-report.txt"
fi

exit $EC
