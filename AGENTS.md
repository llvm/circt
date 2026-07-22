# Build & test guidance

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
- For PyCDE `-DCIRCT_ENABLE_FRONTENDS=PyCDE` (keep Python bindings on). Test with `ninja -C build check-pycde` and `ninja -C build check-pycde-integration`. More complex tests of PyCDE are in the ESI runtime's integration tests, so building and testing the ESI runtime is also a good way to test PyCDE.
- For changes to the ESI runtime, read `lib/Directory/ESI/runtime/AGENTS.md` for details on building and testing the runtime and its cosim features. PyCDE is a dependency of the ESI runtime, so the PyCDE build steps are also required for the ESI runtime.
- When running in local agent mode, try to keep all temporary file writes and reads inside the workspace (in a temp folder) to avoid asking the user for permissions to access files outside the repo.
- Always run `clang-format` on C++ code changes and `yapf` on Python code changes.
- Code style: preserve existing style (variable naming, code patterns, etc.) in the existing codebase.
