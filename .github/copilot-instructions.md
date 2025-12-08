# Copilot build & test guidance

- Default to the integration docker image in `CIRCT_INTEGRATION_IMAGE` (set by the Copilot setup workflow and the `utils/run-docker.sh` default; currently `ghcr.io/circt/images/circt-integration-test:v19.2`) when compiling or testing.
- Run inside that image via `./utils/run-docker.sh ./utils/run-tests-docker.sh "$CIRCT_INTEGRATION_IMAGE"` or `docker run` with the repo root bind-mounted.
- When cloning or checking out, ensure submodules are present (`git submodule update --init --recursive` if needed).
- Configure builds from the repo root with Ninja, matching the README:
  ```
  cmake -G Ninja llvm/llvm -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=$PWD \
    -DLLVM_ENABLE_LLD=ON
  ```
- Build everything with `ninja -C build check-circt`; use `ninja -C build bin/circt-opt` or `ninja -C build bin/firtool` for tool-only builds.
- Keep Python bindings enabled when needed via `-DMLIR_ENABLE_BINDINGS_PYTHON=ON -DCIRCT_BINDINGS_PYTHON_ENABLED=ON`.
- Prefer the integration image and the setup steps workflow for reliable dependencies; only fall back to host builds when explicitly requested.
