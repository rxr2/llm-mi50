#!/usr/bin/env bash
# ROCm userspace for 1x MI50 (gfx906) on Debian 13 / kernel 6.12, INBOX amdgpu driver.
#
#   sudo ./install-rocm.sh stable   # ROCm 7.1.1 (what alex4300 measured on: ROCm 7.1) -> /opt/rocm-7.1.1
#   ./install-rocm.sh modern        # TheRock 10.1.0a20260822, gfx906-only tarball    -> /opt/therock (no root if dir writable)
#   ./install-rocm.sh check         # verify the host without installing anything
#
# NO amdgpu-dkms (caused GPU hangs on this host 2026-09-08). NO HSA_OVERRIDE_GFX_VERSION.
set -euo pipefail
MODE="${1:-check}"

THEROCK_URL=https://rocm.nightlies.amd.com/tarball-multi-arch/therock-dist-linux-gfx906-10.1.0a20260822.tar.gz
THEROCK_SHA256=6ae3c68366ed5ed6130815a20b79fa7e445b6616888cf27689c52435cad1bcd1   # verified 2026-09-23
THEROCK_DIR=/opt/therock
STABLE_VER=7.1.1
STABLE_DEB=https://repo.radeon.com/amdgpu-install/7.1.1/ubuntu/noble/amdgpu-install_7.1.1.70101-1_all.deb

check_host() {
  echo "== kernel: $(uname -r)"
  echo "== amdgpu module:"; modinfo amdgpu 2>/dev/null | grep -E '^(filename|version)' || echo "  not loaded"
  if modinfo amdgpu 2>/dev/null | grep -q updates/dkms; then
    echo "!! amdgpu-dkms module is active. Remove it (apt purge amdgpu-dkms) and reboot into the inbox driver."; fi
  [ -e /dev/kfd ] && echo "== /dev/kfd ok" || echo "!! /dev/kfd missing"
  id -nG | grep -qw render && id -nG | grep -qw video && echo "== user in render+video" || echo "!! add user to groups render,video"
  echo "== PCI:"; lspci -nn | grep -Ei 'vega 20|MI50|66a1' || true
  [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ] && echo "!! HSA_OVERRIDE_GFX_VERSION is set — unset it" || echo "== no HSA override"
  for R in /opt/rocm-$STABLE_VER /opt/rocm $THEROCK_DIR; do
    [ -x "$R/bin/rocminfo" ] || continue
    echo "== $R"
    "$R/bin/rocminfo" 2>/dev/null | grep -E 'Name:\s+gfx906|amdgcn-amd-amdhsa--gfx906' | head -2
    n=$(ls "$R"/lib/rocblas/library 2>/dev/null | grep -c gfx906 || true)
    echo "   rocBLAS gfx906 Tensile files: $n  (0 => F32/F16 GEMM falls over; see fix_rocblas)"
  done
}

fix_rocblas_hint() {
  cat <<'EOF'
rocBLAS in the ROCm 7.x packages can ship WITHOUT gfx906 Tensile kernels. llama.cpp uses MMQ for
Q4_0 prefill, but the F32 tensors (ssm_alpha/ssm_beta 5120x48, norms) and F16 fallbacks go through
hipBLAS/rocBLAS. Fix, in order of preference:
  1) use the Docker image (rocm/dev-ubuntu-24.04:7.1.1-complete) and run the check inside it;
  2) build rocBLAS 7.1 for gfx906 only:
       git clone -b rocm-7.1.1 https://github.com/ROCm/rocBLAS && cd rocBLAS &&
       ./install.sh -a gfx906 --no-tensile-host --cmake_install   (≈1-2 h)
     then copy library/src/build/release/Tensile/library/*gfx906* to /opt/rocm-7.1.1/lib/rocblas/library/;
  3) MODERN variant: TheRock gfx906 tarball ships 210 gfx906 rocBLAS files (verified).
Never mix rocBLAS Tensile files across ROCm major versions.
EOF
}

case "$MODE" in
  check) check_host ;;
  stable)
    [ "$(id -u)" = 0 ] || { echo "stable needs root"; exit 1; }
    # Debian 13 is not an officially supported distro. We install only userspace (no dkms) from the
    # Ubuntu 24.04 (noble) repo, which is what works on trixie; the kernel driver stays inbox.
    tmp=$(mktemp -d); curl -fsSL -o "$tmp/ai.deb" "$STABLE_DEB"; apt-get install -y "$tmp/ai.deb"
    amdgpu-install -y --usecase=rocm,hiplibsdk --no-dkms --rocmrelease="$STABLE_VER"
    # librocprofiler-sdk needs libdw1t64 (missing on Debian 13): fetch it locally, no system change
    mkdir -p /opt/rocm-$STABLE_VER/extra-libs && cd /opt/rocm-$STABLE_VER/extra-libs && \
      (apt-get download libdw1t64 2>/dev/null && dpkg -x libdw1t64_*.deb . || true)
    check_host; fix_rocblas_hint ;;
  modern)
    mkdir -p "$THEROCK_DIR"; cd "$THEROCK_DIR"
    curl -fL -C - -o therock.tgz "$THEROCK_URL"
    echo "$THEROCK_SHA256  therock.tgz" | sha256sum -c -
    tar xzf therock.tgz && rm therock.tgz
    cat > "$THEROCK_DIR/env.sh" <<EOF
export ROCM_PATH=$THEROCK_DIR
export PATH=$THEROCK_DIR/bin:$THEROCK_DIR/lib/llvm/bin:\$PATH
export LD_LIBRARY_PATH=$THEROCK_DIR/lib:$THEROCK_DIR/lib/llvm/lib:\${LD_LIBRARY_PATH:-}
EOF
    echo "source $THEROCK_DIR/env.sh"; check_host ;;
  *) echo "usage: $0 stable|modern|check"; exit 1 ;;
esac
