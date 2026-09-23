#!/usr/bin/env python3
"""isa_verify.py — multi-ELF verifier for MI50 (gfx906) kernel symbols.

Extracted from verify-isa.sh step 2 so the per-ELF logic is statically
testable. Grouping invariant: symbols are grouped BY ELF FILE — never a
stale loop variable, never "the last e.path" — so a bundle + embedded
archive with duplicated symbols can never mislabel disassembly or mix
e_flags between files.

Pass conditions (ALL required — no "any symbol found" success):
  1. TSV defaults file present and parsed (kernel -> lds_bytes/scratch_bytes).
  2. Every required default kernel symbol (REQUIRED_DEFAULTS) is found in
     at least one extracted ELF that also satisfies (3) and (4).
  3. The hosting ELF's e_flags machine field == EF_AMDGPU_MACH_GFX906
     (0x24 = 36 in the low 10 bits of e_flags) — a PASS condition, not
     informational.
  4. The symbol's stub descriptor (group=lds, private=scratch, kernarg)
     has private == 0 and group consistent with the TSV default row
     (exact lds, lds-64, or lds-256). The TSV row's scratch_bytes must be 0.
  5. A disassembly of THAT symbol was produced FROM that ELF's own file
     (one objdump run per ELF over its own symbols only).
"""
import os
import struct
import subprocess
import sys

EF_AMDGPU_MACH_GFX906 = 0x24    # 36 — mach id inside e_flags (low 10 bits)
EF_AMDGPU_MACH_MASK = 0x3FF
AMDGPU_MACHINE = 0xE0
GROUP_DESC_STRUCT = "<IIII"     # group/lds, private/scratch, kernarg, unused

# Required default symbols — exact substrings taken verbatim from
# build-verification/mmvq-gfx906-kernel-resources.tsv (columns:
# kernel, vgpr, sgpr, lds_bytes, scratch_bytes, max_wg) under the README
# dispatch policy: q4_0 n=1 base+fused (TPR16 KC64), n=4 TPR16 KC64,
# n=8 TPR16 KC32; q5_K/q6_K default KC8 at n<=5 (n=4 row).
REQUIRED_DEFAULTS = [
    ("q4_0_breit-n1-base",
     "mul_mat_vec_q4_0_breit_gfx906ILi1ELi16ELi64ELb0"),
    ("q4_0_breit-n1-fused",
     "mul_mat_vec_q4_0_breit_gfx906ILi1ELi16ELi64ELb1"),
    ("q4_0_breit-n4",
     "mul_mat_vec_q4_0_breit_gfx906ILi4ELi16ELi64ELb0"),
    ("q4_0_breit-n8",
     "mul_mat_vec_q4_0_breit_gfx906ILi8ELi16ELi32ELb0"),
    ("q5_K_breit-default",
     "mul_mat_vec_q5_K_breit_gfx906ILi4ELi16ELi8"),
    ("q6_K_breit-default",
     "mul_mat_vec_q6_K_breit_gfx906ILi4ELi16ELi8"),
]

REPORT_HEADER = ("# MI50 ISA verification report — per-ELF grouping, "
                 "gfx906 e_flags is a PASS condition\n")


# ---------------------------------------------------------------- ELF parsing

class Elf:
    def __init__(self, path):
        self.path = path
        self.e_flags = 0
        self.e_machine = 0
        self.text = b""
        self.syms = []          # (name, value, size)
        self._parse()

    def _parse(self):
        with open(self.path, "rb") as f:
            data = f.read()
        if data[:4] != b"\x7fELF" or len(data) < 64 or data[4] != 2:
            raise ValueError("not a 64-bit ELF")
        (e_type, e_machine, _e_version, _e_entry, _e_phoff, e_shoff,
         _e_flags2, _e_ehsize, _e_phentsize, _e_phnum, e_shentsize,
         e_shnum, e_shstrndx) = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)
        self.e_flags = struct.unpack_from("<I", data, 48)[0]   # Elf64 e_flags
        self.e_machine = e_machine
        if e_type != 1 or e_shoff == 0 or e_shnum == 0:
            raise ValueError("not ET_REL (e_type=%d)" % e_type)

        raw = []
        for i in range(e_shnum):
            off = e_shoff + i * e_shentsize
            (sh_name, sh_type, _sh_flags, _sh_addr, sh_offset, sh_size,
             sh_link, _sh_info, _sh_addralign, sh_entsize) = \
                struct.unpack_from("<IIQQQQIIQQ", data, off)
            raw.append((sh_name, sh_type, sh_offset, sh_size,
                        sh_link, sh_entsize))
        if e_shstrndx >= len(raw):
            raise ValueError("bad e_shstrndx")
        shstr_off, shstr_size = raw[e_shstrndx][2], raw[e_shstrndx][3]
        shstr = data[shstr_off:shstr_off + shstr_size]

        def sname(n):
            end = shstr.find(b"\0", n)
            return shstr[n:end].decode("ascii", "replace")

        symtab = None
        for sh_name, sh_type, sh_offset, sh_size, sh_link, sh_entsize in raw:
            if sh_type == 2 and sh_entsize == 24 and symtab is None:
                symtab = (sh_offset, sh_size, sh_entsize, sh_link)
            name = sname(sh_name) if sh_name < len(shstr) else ""
            if name == ".text":
                self.text = data[sh_offset:sh_offset + sh_size]

        if symtab:
            soff, ssize, entsize, link = symtab
            if link >= len(raw):
                raise ValueError("bad symtab sh_link")
            str_off, str_sz = raw[link][2], raw[link][3]
            strtab = data[str_off:str_off + str_sz]
            for o in range(soff, soff + ssize, entsize):
                (st_name, _st_info, _st_other, _st_shndx, st_value,
                 st_size) = struct.unpack_from("<IBBHQQ", data, o)
                if st_name == 0:
                    continue
                end = strtab.find(b"\0", st_name)
                nm = strtab[st_name:end].decode("ascii", "replace")
                if nm:
                    self.syms.append((nm, st_value, st_size))

    @property
    def mach(self):
        return self.e_flags & EF_AMDGPU_MACH_MASK

    @property
    def mach_ok(self):
        """gfx906 e_flags — required PASS condition."""
        return self.mach == EF_AMDGPU_MACH_GFX906


# ------------------------------------------------------------- descriptors

def read_descriptor(elf, offset):
    if offset < 0 or offset + 16 > len(elf.text):
        return None
    return struct.unpack_from(GROUP_DESC_STRUCT, elf.text, offset)


def check_scratch(elf, sym_value, exp_lds):
    """-> (scratch_value, ok). exp_lds>0: group must match TSV default
    (exact lds, lds-64 or lds-256 — old variants) and private == 0;
    exp_lds==0: any group, private must be 0. None scratch if not located."""
    if exp_lds > 0:
        candidates = {exp_lds, exp_lds - 64, exp_lds - 256}
    else:
        candidates = {0, 64, 256, 512, 1024, 1536, 2048, 4096}
    lo = hi = int(sym_value)
    if exp_lds == 0:
        lo = max(0, int(sym_value) - 64)
        hi = min(max(0, len(elf.text) - 16), int(sym_value) + 64)
    for off in range(lo, hi + 1, 8):
        desc = read_descriptor(elf, off)
        if desc is None:
            continue
        group, priv, _kernarg, _unused = desc
        if group in candidates:
            return priv, (priv == 0)
    return None, False


# ------------------------------------------------------------------ TSV

def load_tsv(path):
    """kernel column -> (lds_bytes, scratch_bytes) for every row.
    Returns None if the file is missing/unreadable."""
    if not os.path.isfile(path):
        return None
    out = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("kernel\t"):
                continue
            cols = line.split("\t")
            if len(cols) < 6:
                continue
            try:
                out[cols[0]] = (int(cols[3]), int(cols[4]))
            except ValueError:
                continue
    return out


def tsv_row_for(tsv, needle):
    """The TSV row whose kernel string contains `needle` (exact default)."""
    for kernel, (lds, scratch) in tsv.items():
        if needle in kernel:
            return lds, scratch
    return None


# ------------------------------------------------------------------ main

def run_verifier(elf_dir, out_dir, report_path, tsv_path, objdump):
    report = [REPORT_HEADER]

    def say(msg):
        print(msg)
        report.append(msg)

    def finish(rc):
        with open(report_path, "w") as f:
            f.write("\n".join(report) + "\n")
        return rc

    tsv = load_tsv(tsv_path)
    if tsv is None:
        say("FAIL  TSV defaults file missing or unreadable: %s" % tsv_path)
        say("      (source of default-dispatch lds/scratch values — the "
            "TSV must be present, e.g. COPY build-verification in Docker)")
        return finish(1)

    # sanity: every required default must have a TSV row, scratch_bytes == 0
    tsv_failures = 0
    exp_lds = {}
    for label, needle in REQUIRED_DEFAULTS:
        row = tsv_row_for(tsv, needle)
        if row is None:
            say("FAIL  no TSV default row for %s (%s)" % (label, needle))
            tsv_failures += 1
            continue
        lds, scratch = row
        if scratch != 0:
            say("FAIL  TSV default row for %s has scratch_bytes=%d != 0"
                % (label, scratch))
            tsv_failures += 1
            continue
        exp_lds[label] = lds

    # ---- load every extracted ELF (grouped by file)
    elfs = []
    if os.path.isdir(elf_dir):
        for fn in sorted(os.listdir(elf_dir)):
            p = os.path.join(elf_dir, fn)
            if not os.path.isfile(p):
                continue
            try:
                elfs.append(Elf(p))
            except ValueError as e:
                say("WARN  skip %s: %s" % (fn, e))
    if not elfs:
        say("FAIL  no ELF objects extracted under %s" % elf_dir)
        return finish(1)

    say("ELFs under test (grouped per file — no shared loop state):")
    for e in elfs:
        say("  %-44s machine=0x%03x e_flags=0x%08x mach=0x%03x gfx906=%s"
            % (os.path.basename(e.path), e.e_machine, e.e_flags, e.mach,
               "OK" if e.mach_ok else "NO"))

    # ---- locate each required symbol inside each ELF
    located = {}   # label -> list of (elf, name, value, size)
    for label, needle in REQUIRED_DEFAULTS:
        hits = []
        for e in elfs:
            for nm, val, sz in e.syms:
                if needle in nm:
                    hits.append((e, nm, val, sz))
                    break        # first match per ELF is enough
        located[label] = hits
        if not hits:
            say("FAIL  required default symbol NOT FOUND: %s (%s)"
                % (label, needle))

    # ---- per-ELF disassembly: ONE objdump run per ELF over ITS OWN symbols
    disasm_dir = os.path.join(out_dir, "disassembly")
    os.makedirs(disasm_dir, exist_ok=True)
    per_elf = {}   # elf.path -> {label: (elf, name, val, sz)}
    for label, needle in REQUIRED_DEFAULTS:
        for e, nm, val, sz in located.get(label, []):
            per_elf.setdefault(e.path, {})[label] = (e, nm, val, sz)

    dumped = {}    # elf.path -> disassembly file
    for path, syms in sorted(per_elf.items()):
        e = next(x for x in elfs if x.path == path)
        tag = "%s_mach%03x" % (os.path.basename(path).replace(".", "_"),
                               e.mach)
        dest = os.path.join(disasm_dir, "disassembly-%s.txt" % tag)
        names = sorted({v[1] for v in syms.values()})
        cmd = [objdump, "-d", "--disassemble-symbols=" + ",".join(names),
               path]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=600)
            if res.returncode != 0:
                say("FAIL  objdump rc=%d for %s: %s"
                    % (res.returncode, os.path.basename(path),
                       (res.stderr or "").strip()[:200]))
                continue
            with open(dest, "w") as f:
                f.write(res.stdout)
            dumped[path] = dest
            say("  objdump -> %s  (symbols: %s)"
                % (os.path.basename(dest), ", ".join(names)))
        except (OSError, subprocess.TimeoutExpired) as exc:
            say("FAIL  objdump error for %s: %s"
                % (os.path.basename(path), exc))

    # ---- evaluate every required default
    failures = tsv_failures
    say("")
    say("Required default symbols:")
    for label, needle in REQUIRED_DEFAULTS:
        hits = located.get(label) or []
        if not hits:
            say("  FAIL %-22s not found in any ELF" % label)
            failures += 1
            continue
        ok = False
        reasons = []
        for e, nm, val, sz in hits:
            r = []
            if e.e_machine != AMDGPU_MACHINE:
                r.append("e_machine=0x%03x != AMDGPU(0x%03x)"
                         % (e.e_machine, AMDGPU_MACHINE))
            if not e.mach_ok:
                r.append("e_flags mach=0x%03x != GFX906(0x%03x) — FAIL "
                         "condition" % (e.mach, EF_AMDGPU_MACH_GFX906))
            if label not in exp_lds:
                r.append("no usable TSV default row")
                scratch = None
            else:
                scratch, s_ok = check_scratch(e, val, exp_lds[label])
                if scratch is None:
                    r.append("descriptor not located at symbol")
                elif not s_ok:
                    r.append("scratch/private=%s != 0 (expected lds=%d)"
                             % (scratch, exp_lds[label]))
            dfile = dumped.get(e.path)
            if not dfile or not os.path.isfile(dfile):
                r.append("no disassembly produced from its own ELF")
            else:
                with open(dfile) as f:
                    if nm not in f.read():
                        r.append("own-ELF disassembly lacks the symbol")
            if not r:
                ok = True
                say("  OK   %-22s elf=%s mach=0x%03x scratch=%s disasm=%s"
                    % (label, os.path.basename(e.path), e.mach, scratch,
                       os.path.basename(dfile)))
                break
            reasons.append("%s: %s" % (os.path.basename(e.path),
                                       "; ".join(r)))
        if not ok:
            say("  FAIL %-22s %s" % (label, " | ".join(reasons)))
            failures += 1

    say("")
    say("gfx906 e_flags (PASS condition — mach must be 0x%03x / %d):"
        % (EF_AMDGPU_MACH_GFX906, EF_AMDGPU_MACH_GFX906))
    for e in elfs:
        say("  %-44s e_flags=0x%08x mach=0x%03x %s"
            % (os.path.basename(e.path), e.e_flags, e.mach,
               "OK" if e.mach_ok else "MISMATCH"))

    say("")
    if failures:
        say("FAIL — %d required default check(s) unverified" % failures)
        return finish(1)
    say("PASS — all %d required default symbols verified across %d ELF "
        "file(s)" % (len(REQUIRED_DEFAULTS), len(elfs)))
    return finish(0)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("usage: isa_verify.py ELF_DIR OUT_DIR REPORT_PATH TSV_PATH OBJDUMP")
        sys.exit(2)
    sys.exit(run_verifier(*sys.argv[1:6]))
