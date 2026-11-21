# VeryDiff-Agent exploration sandbox: Julia 1.11 + VeryDiff (same checkout) + Python tools.
# Runs with --network none, non-root, read-only rootfs (see server/verydiff_mcp/sandbox.py).
# Build with ./build.sh (stages VeryDiff's src/Project/Manifest into the build context).
FROM julia:1.11 AS julia

FROM python:3.12-slim
COPY --from=julia /usr/local/julia /usr/local/julia
RUN ln -s /usr/local/julia/bin/julia /usr/local/bin/julia

ENV JULIA_DEPOT_PATH=/opt/julia-depot \
    VERYDIFF_NO_GUROBI=1 \
    JULIA_PKG_PRECOMPILE_AUTO=0 \
    PIP_NO_CACHE_DIR=1

# Build-time only: proxy CA bundle (secret, not stored in the image).
RUN --mount=type=secret,id=ca,target=/run/secrets/ca \
    SSL_CERT_FILE=/run/secrets/ca PIP_CERT=/run/secrets/ca \
    pip install numpy scipy onnx onnxruntime

# The Julia depot (registry, packages, artifacts) is staged from the host by build.sh, so the
# image builds offline and with exactly the package versions the host kernel uses.
COPY depot /opt/julia-depot
COPY verydiff /opt/verydiff
RUN JULIA_PKG_OFFLINE=true julia --project=/opt/verydiff -e \
      'using Pkg; Pkg.instantiate(); Pkg.build("Gurobi"); Pkg.precompile(); using VeryDiff' \
    && chmod -R a+rX /opt/julia-depot /opt/verydiff

# Convenience CLI: verydiff NET1 NET2 SPEC --epsilon E [--timeout T]
RUN printf '#!/bin/sh\nexec julia --project=/opt/verydiff /opt/verydiff/cli.jl "$@"\n' > /usr/local/bin/verydiff \
    && chmod +x /usr/local/bin/verydiff \
    && useradd -m -u 1000 agent
USER agent
WORKDIR /scratch
CMD ["sleep", "infinity"]
