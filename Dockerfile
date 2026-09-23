# MI50-QWEN38 stack. Two ROCm variants from one file:
#   docker build --build-arg ROCM=stable -t mi50-qwen38:stable .
#   docker build --build-arg ROCM=modern -t mi50-qwen38:modern .
#   docker build --build-arg ROCM=stable --build-arg VARIANT=alex-exact -t mi50-qwen38:alex .
# Run (host keeps the INBOX amdgpu driver; no dkms, no HSA override):
#   docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
#     --security-opt seccomp=unconfined -v /models:/models -p 8080:8080 mi50-qwen38:stable \
#     /opt/mi50/run-golden.sh /models/Qwen3.8-27B-Q4_0.gguf
ARG ROCM=stable

FROM rocm/dev-ubuntu-24.04:7.1.1-complete AS base-stable
ENV ROCM_PATH=/opt/rocm

FROM ubuntu:24.04 AS base-modern
ARG THEROCK_URL=https://rocm.nightlies.amd.com/tarball-multi-arch/therock-dist-linux-gfx906-10.1.0a20260822.tar.gz
ARG THEROCK_SHA256=6ae3c68366ed5ed6130815a20b79fa7e445b6616888cf27689c52435cad1bcd1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates libnuma1 libelf1 libdrm2 libdrm-amdgpu1 \
    && mkdir -p /opt/therock && curl -fL -o /tmp/t.tgz "$THEROCK_URL" \
    && echo "$THEROCK_SHA256  /tmp/t.tgz" | sha256sum -c - \
    && tar xzf /tmp/t.tgz -C /opt/therock && rm /tmp/t.tgz && rm -rf /var/lib/apt/lists/*
ENV ROCM_PATH=/opt/therock \
    PATH=/opt/therock/bin:/opt/therock/lib/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/therock/lib:/opt/therock/lib/llvm/lib

FROM base-${ROCM} AS build
ARG VARIANT=golden
RUN apt-get update && apt-get install -y --no-install-recommends git cmake ninja-build g++ python3 curl ca-certificates libssl-dev jq bc \
    && rm -rf /var/lib/apt/lists/*
COPY patches /opt/mi50/patches
COPY build.sh benchmark-mi50.sh run-golden.sh power.sh preflight.sh summarize.py verify-isa.sh model-provenance.sh /opt/mi50/
COPY prompts /opt/mi50/prompts
RUN chmod +x /opt/mi50/*.sh && SKIP_TESTS=1 SRC=/src/llama.cpp OUT=/opt/mi50/builds ROCM_PATH=$ROCM_PATH \
    /opt/mi50/build.sh "$VARIANT" \
    && ln -s /opt/mi50/builds/$VARIANT/build/bin /opt/mi50/bin
ENV PATH=/opt/mi50/bin:$PATH
# HSA_OVERRIDE_GFX_VERSION intentionally NOT set (gfx906 is native)
WORKDIR /opt/mi50
