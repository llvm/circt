{
  description = "CIRCT CI built from reusable Nix LLVM/MLIR derivations";

  inputs = {
    circt-src = {
      url = "github:llvm/circt/27622798bd566646effbc07974500b6a669f4993";
      flake = false;
    };
    llvm-src = {
      url = "github:llvm/llvm-project/040a641988f6ed6f4fab250706ca2b620c1de2d8";
      flake = false;
    };
    circt-nix = {
      url = "github:xinpian-tech/circt-nix";
      inputs.circt-src.follows = "circt-src";
      inputs.llvm-submodule-src.follows = "llvm-src";
    };
    nixpkgs.follows = "circt-nix/nixpkgs";
    cli11-src = {
      url = "github:CLIUtils/CLI11/v2.5.0";
      flake = false;
    };
    fmt-src = {
      url = "github:fmtlib/fmt/11.1.4";
      flake = false;
    };
    googletest-src = {
      url = "github:google/googletest/v1.15.2";
      flake = false;
    };
    ixwebsocket-src = {
      url = "github:machinezone/IXWebSocket/173f442474c4d9db16184c5e15cc96e07605e0e0";
      flake = false;
    };
    nlohmann-json-src = {
      url = "github:nlohmann/json/v3.11.3";
      flake = false;
    };
    slang-src = {
      # Match the revision used by this CIRCT checkout's FetchContent build.
      url = "github:MikePopoloski/slang/44dc55f99b9c64971893013e7931e643fbedcf23";
      flake = false;
    };
    zlib-src = {
      url = "github:madler/zlib/5a82f71ed1dfc0bec044d9702463dbdf84ea3b71";
      flake = false;
    };
  };

  outputs =
    {
      self,
      circt-nix,
      circt-src,
      cli11-src,
      fmt-src,
      googletest-src,
      ixwebsocket-src,
      llvm-src,
      nixpkgs,
      nlohmann-json-src,
      slang-src,
      zlib-src,
    }:
    let
      supportedSystems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      mkPython =
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        pkgs.python312.withPackages (
          ps:
          let
            cocotb = ps.cocotb.overridePythonAttrs (_: {
              version = "1.9.2";
              src = pkgs.fetchFromGitHub {
                owner = "cocotb";
                repo = "cocotb";
                tag = "v1.9.2";
                hash = "sha256-7KCo7g2I1rfm8QDHRm3ZKloHwjDIICnJCF8KhaFdvqY=";
              };
              postPatch = ''
                patchShebangs bin/*.py
              '';
              doCheck = false;
            });
            cocotbTest = ps.buildPythonPackage rec {
              pname = "cocotb_test";
              version = "0.2.6";
              format = "setuptools";
              src = pkgs.fetchPypi {
                inherit pname version;
                hash = "sha256-pGYZSMoUXu5rzK+DIXS177n200IGdxV8BK3nK7IWBpI=";
              };
              propagatedBuildInputs = [ cocotb ];
              doCheck = false;
            };
            nanobind = ps.nanobind.overridePythonAttrs (_: {
              version = "2.9.2";
              src = pkgs.fetchgit {
                url = "https://github.com/wjakob/nanobind";
                rev = "b775c42f2eb3cac13efc5bc266766066306898a6";
                fetchSubmodules = true;
                hash = "sha256-cC+sf2FUm1jdGMRdDoaQK8rjUVkWjn/53c1HQ5gsUWs=";
              };
            });
            pybind11 = ps.buildPythonPackage rec {
              pname = "pybind11";
              version = "2.11.2";
              pyproject = true;
              src = pkgs.fetchFromGitHub {
                owner = "pybind";
                repo = "pybind11";
                tag = "v${version}";
                hash = "sha256-F8+bb6wZ/BygzMGN1q48X9qzYsCUWanuj/MiZ1s8ShM=";
              };
              build-system = [
                ps.cmake
                ps.ninja
                ps.setuptools
              ];
              dontUseCmakeConfigure = true;
              doCheck = false;
              postInstall = ''
                ln -s "$out/${pkgs.python312.sitePackages}/pybind11/include" "$out/include"
                ln -s "$out/${pkgs.python312.sitePackages}/pybind11/share" "$out/share"
              '';
            };
          in
          [
            ps.click
            cocotb
            cocotbTest
            ps.executing
            ps.jinja2
            ps.lit
            nanobind
            ps.numpy
            ps.packaging
            ps.psutil
            pybind11
            ps.pycapnp
            ps.pytest
            ps.pytest-xdist
            ps.pyyaml
            ps.setuptools
            ps.typing-extensions
            ps.wheel
          ]
        );
      mkCore =
        system:
        let
          pkgs = circt-nix.legacyPackages.${system};
          basePkgs = import nixpkgs { inherit system; };
          python = mkPython system;
          upstreamLLVM = pkgs.circtFlakePkgs.llvmPackages_circt;

          # CIRCT tracks LLVM through its gitlink. Keep LLVM and MLIR as
          # independent derivations so all CIRCT configurations share them.
          # Fuzz 3 is needed for nixpkgs' GNU install-dir patch after the
          # per-target runtime-dir block was added upstream.
          llvmPatchFlags = [
            "-p1"
            "-F3"
          ];
          llvmPackages = upstreamLLVM.overrideScope (
            selfLLVM: superLLVM: {
              # llvm-tblgen is bootstrapped in its own derivation and applies
              # the same LLVM patches before the main libllvm build.
              tblgen = superLLVM.tblgen.overrideAttrs (old: {
                patchFlags = llvmPatchFlags;
                postPatch = (old.postPatch or "") + ''
                  # LLVM 23's top-level project discovery expects this sibling
                  # even though the lightweight tblgen build disables libc.
                  chmod u+w ..
                  ln -s ${llvm-src}/libc ../libc
                '';
              });

              libllvm =
                (superLLVM.libllvm.override {
                  buildLlvmPackages = { inherit (selfLLVM) tblgen; };
                  # Python loads several MLIR/CIRCT extension modules into one
                  # process.  A monolithic LLVM dylib keeps those modules from
                  # embedding independent copies of LLVM's global registries.
                  enableSharedLibraries = true;
                }).overrideAttrs
                  (old: {
                    patchFlags = llvmPatchFlags;
                    buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.z3 ];
                    cmakeFlags = (old.cmakeFlags or [ ]) ++ [
                      # circt-nix defaults these to OFF after nixpkgs' own
                      # flags, so repeat them at the end of the final list.
                      "-DLLVM_BUILD_LLVM_DYLIB=ON"
                      "-DLLVM_LINK_LLVM_DYLIB=ON"
                      "-DLLVM_ENABLE_REVERSE_ITERATION=ON"
                      "-DLLVM_ENABLE_Z3_SOLVER=ON"
                    ];
                    passthru = (old.passthru or { }) // {
                      source = llvm-src;
                    };
                  });

              mlir =
                (superLLVM.mlir.override {
                  buildLlvmPackages = { inherit (selfLLVM) tblgen; };
                  inherit (selfLLVM) libllvm;
                }).overrideAttrs
                  (old: {
                    nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ python ];
                    buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.z3 ];
                    cmakeFlags = (old.cmakeFlags or [ ]) ++ [
                      # CIRCT links MLIR component targets directly.  Build
                      # those components as shared libraries, all backed by
                      # the one libLLVM dylib, instead of also creating a
                      # second aggregate libMLIR representation.
                      "-DBUILD_SHARED_LIBS=ON"
                      "-DLLVM_BUILD_LLVM_DYLIB=OFF"
                      "-DLLVM_LINK_LLVM_DYLIB=ON"
                      "-DMLIR_LINK_MLIR_DYLIB=OFF"
                      "-DLLVM_ENABLE_REVERSE_ITERATION=ON"
                      "-DLLVM_ENABLE_Z3_SOLVER=ON"
                      "-DMLIR_ENABLE_BINDINGS_PYTHON=ON"
                      "-DMLIR_INSTALL_AGGREGATE_OBJECTS=ON"
                      "-DPython_EXECUTABLE=${python}/bin/python3"
                      "-DPython3_EXECUTABLE=${python}/bin/python3"
                    ];
                    passthru = (old.passthru or { }) // {
                      inherit (selfLLVM) libllvm;
                      source = llvm-src;
                    };
                  });
            }
          );
          inherit (llvmPackages) libllvm mlir;

          # The CIRCT source under test requires Slang 11, while circt-nix's
          # locked CIRCT release still uses Slang 10. Reuse its packaging and
          # patches, changing only the independently versioned Slang source.
          slang = pkgs.slang.overrideAttrs (old: {
            src = slang-src;
            version = "11.0";
            patches = builtins.filter (
              patch:
              let
                path = toString patch;
              in
              !(pkgs.lib.hasSuffix "slang-don-t-fetch-fmt.patch" path)
              && !(pkgs.lib.hasSuffix "slang-vendored-boost-headers.patch" path)
            ) (old.patches or [ ]);
            buildInputs = builtins.filter (dep: pkgs.lib.getName dep != "catch2") (old.buildInputs or [ ]);
            propagatedBuildInputs = (old.propagatedBuildInputs or [ ]) ++ [
              basePkgs.boost
              basePkgs.fmt
              basePkgs.tomlplusplus
            ];
            cmakeFlags = (old.cmakeFlags or [ ]) ++ [ "-DSLANG_INCLUDE_TESTS=OFF" ];
            postPatch = ''
              substituteInPlace source/util/VersionInfo.cpp.in \
                --subst-var SLANG_VERSION_MAJOR \
                --subst-var SLANG_VERSION_MINOR \
                --subst-var SLANG_VERSION_PATCH \
                --subst-var SLANG_VERSION_HASH
              substituteInPlace CMakeLists.txt \
                --replace-fail 'VERSION ''${SLANG_VERSION_STRING}' \
                               'VERSION "11.0"'
            '';
            SLANG_VERSION_MAJOR = "11";
            SLANG_VERSION_MINOR = "0";
            SLANG_VERSION_PATCH = "0";
            SLANG_VERSION_HASH = slang-src.shortRev or "dirty";
            doCheck = false;
          });

          mkCirct =
            args:
            (pkgs.circt.override (
              {
                inherit libllvm mlir slang;
                python3 = python;
              }
              // args
            )).overrideAttrs
              (old: {
                # circt-nix expects the GitHub source archive's empty llvm
                # gitlink directory. A PR source supplied with git+file omits
                # that directory, so replace the upstream postUnpack with a
                # form that works for both source representations.
                postUnpack = ''
                  if [[ -e "$sourceRoot/llvm" || -L "$sourceRoot/llvm" ]]; then
                    rm -rf -- "$sourceRoot/llvm"
                  fi
                  ln -s ${llvm-src} "$sourceRoot/llvm"
                '';
                postPatch = (old.postPatch or "") + ''
                  substituteInPlace CMakeLists.txt \
                    --replace-fail \
                    '  if (CIRCT_BINDINGS_PYTHON_ENABLED)
                      message(FATAL_ERROR "CIRCT Python bindings require a unified build. \
                                           See docs/PythonBindings.md.")
                    endif()
                  ' \
                    '  # The Nix MLIR package exports its Python source targets.
                    # This lets the standalone CIRCT package reuse installed MLIR.
                  '
                  substituteInPlace CMakeLists.txt \
                    --replace-fail \
                    '  mlir_configure_python_dev_packages()' \
                    '  if(CIRCT_BUILT_STANDALONE)
                      include(MLIRDetectPythonEnv)
                    endif()
                    mlir_configure_python_dev_packages()'
                  # Standalone CIRCT globally disables exceptions and RTTI,
                  # but every ESI runtime configuration requires both. Limit
                  # the override to the runtime directory and its subtargets.
                  substituteInPlace lib/Dialect/ESI/runtime/CMakeLists.txt \
                    --replace-fail \
                    'project(ESIRuntime LANGUAGES CXX)' \
                    'project(ESIRuntime LANGUAGES CXX)

                    if(NOT MSVC)
                      add_compile_options(
                        "$<$<COMPILE_LANGUAGE:CXX>:-fexceptions>"
                        "$<$<COMPILE_LANGUAGE:CXX>:-frtti>"
                      )
                    endif()'

                  # Standalone LLVM exports component targets as static
                  # archives even when LLVM_LINK_LLVM_DYLIB is enabled. CIRCT
                  # must request those components through add_llvm_executable
                  # so they are mapped back to the shared LLVM target; linking
                  # the imported archives directly creates a second copy of
                  # LLVM's process-global registries.
                  substituteInPlace tools/arcilator/CMakeLists.txt \
                    --replace-fail \
                    'set(ARCILATOR_JIT_LLVM_COMPONENTS native)' \
                    'set(ARCILATOR_JIT_LLVM_COMPONENTS native OrcJIT)' \
                    --replace-fail '    LLVMOrcJIT' "" \
                    --replace-fail \
                    'set(LLVM_LINK_COMPONENTS Support ''${ARCILATOR_JIT_LLVM_COMPONENTS})' \
                    'set(LLVM_LINK_COMPONENTS Support TargetParser ''${ARCILATOR_JIT_LLVM_COMPONENTS})' \
                    --replace-fail '  LLVMTargetParser' ""
                  for cmake_file in \
                    tools/circt-bmc/CMakeLists.txt \
                    tools/circt-lec/CMakeLists.txt \
                    unittests/Conversion/ImportVerilog/CMakeLists.txt
                  do
                    substituteInPlace "$cmake_file" \
                      --replace-fail '  LLVMSupport' ""
                  done
                  substituteInPlace tools/circt-synth/CMakeLists.txt \
                    --replace-fail \
                    'add_circt_tool(circt-synth circt-synth.cpp)' \
                    'set(LLVM_LINK_COMPONENTS Support)
                    add_circt_tool(circt-synth circt-synth.cpp)' \
                    --replace-fail '  LLVMSupport' ""
                  substituteInPlace lib/Bindings/Python/CMakeLists.txt \
                    --replace-fail '    LLVMSupport' '    LLVM'

                  # Lit 18 keeps the per-test timeout on LitConfig, while the
                  # source tree's newer lit keeps it on TestingConfig. Support
                  # both APIs in the custom TableGen test format.
                  substituteInPlace \
                    test/Tools/circt-tblgen/self-contained/self_contained_td_format.py \
                    --replace-fail \
                    'timeout = test.config.maxIndividualTestTime or None' \
                    'timeout = getattr(test.config, "maxIndividualTestTime",
                                      getattr(litConfig, "maxIndividualTestTime", 0)) or None'

                  # Let lit preserve the include search path supplied by the
                  # check derivation for clang-tidy's SystemC smoke test.
                  substituteInPlace integration_test/lit.cfg.py \
                    --replace-fail \
                    "['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP']" \
                    "['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH']"
                '';
                preConfigure = (old.preConfigure or "") + ''
                  # Keep the value containing whitespace as one CMake argv.
                  # The generic hook splits string-valued cmakeFlags on
                  # whitespace before appending cmakeFlagsArray.
                  cmakeFlagsArray+=("-DLLVM_LIT_ARGS=-v --show-unsupported")
                '';
                cmakeFlags =
                  builtins.filter
                    (
                      flag:
                      !(pkgs.lib.hasPrefix "-DLLVM_EXTERNAL_LIT=" flag) && !(pkgs.lib.hasPrefix "-DLLVM_LIT_ARGS=" flag)
                    )
                    (
                      map (builtins.replaceStrings
                        [ "-DCIRCT_INSTALL_PACKAGE_DIR==" ]
                        [ "-DCIRCT_INSTALL_PACKAGE_DIR=" ]
                      ) (old.cmakeFlags or [ ])
                    )
                  ++ [
                    "-DMLIR_MAIN_SRC_DIR=${llvm-src}/mlir"
                    "-DMLIR_TOOLS_DIR=${mlir}/bin"
                    # Replace circt-nix's Python 3.13 lit input entirely, not
                    # merely later on the command line, so it is absent from
                    # the derivation closure.
                    "-DLLVM_EXTERNAL_LIT=${python}/bin/.lit-wrapped"
                    # Never copy LLVM/MLIR registries into individual CIRCT
                    # libraries or Python extension modules.
                    "-DLLVM_LINK_LLVM_DYLIB=ON"
                    "-DMLIR_LINK_MLIR_DYLIB=OFF"
                  ];
              });
        in
        {
          inherit
            libllvm
            llvm-src
            mlir
            mkCirct
            pkgs
            slang
            ;
        };
    in
    {
      packages = forAllSystems (
        system:
        let
          core = mkCore system;
          circt = core.mkCirct { };
        in
        {
          # CI overrides circt-src with the pull request checkout and llvm-src
          # with its gitlink revision. The latter changes only when CIRCT
          # intentionally bumps LLVM.
          default = circt;
          inherit circt;
          inherit (core) libllvm mlir;
        }
      );

      checks = forAllSystems (
        system:
        let
          core = mkCore system;
          inherit (core)
            libllvm
            mlir
            mkCirct
            pkgs
            ;
          ciPkgs = import nixpkgs { inherit system; };
          python = mkPython system;

          # Espresso 2.4 carries a pre-ANSI declaration for srandom which is
          # incompatible with current glibc headers. Keep the upstream package
          # and apply the smallest possible compatibility fix locally.
          espresso = pkgs.espresso.overrideAttrs (old: {
            postPatch = (old.postPatch or "") + ''
              substituteInPlace utility/port.h \
                --replace-fail 'extern VOID_HACK srandom();' \
                               'extern void srandom(unsigned int);'
            '';
          });

          ciInputs = [
            ciPkgs.clang-tools
            espresso
            ciPkgs.iverilog
            python
            ciPkgs.sby
            ciPkgs.verilator
            ciPkgs.yosys
            pkgs.z3
          ];

          mkCirctCheck =
            {
              name,
              circt,
              mlir,
              extraCmakeFlags ? [ ],
              extraCheckTargets ? [ ],
              pycde ? false,
            }:
            circt.overrideAttrs (old: {
              pname = "circt-${name}";
              # Simulator/tool discovery happens during CMake configuration,
              # so these must be build inputs rather than check-only inputs.
              nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ ciInputs ++ [ ciPkgs.sccache ];
              nativeCheckInputs = (old.nativeCheckInputs or [ ]) ++ ciInputs;
              buildInputs = (old.buildInputs or [ ]) ++ [
                ciPkgs.systemc
                ciPkgs.zlib.dev
              ];
              cmakeFlags =
                (old.cmakeFlags or [ ])
                ++ extraCmakeFlags
                ++ [
                  "-DPython_EXECUTABLE=${python}/bin/python3"
                  "-DPython3_EXECUTABLE=${python}/bin/python3"
                  "-DCMAKE_C_COMPILER_LAUNCHER=${ciPkgs.sccache}/bin/sccache"
                  "-DCMAKE_CXX_COMPILER_LAUNCHER=${ciPkgs.sccache}/bin/sccache"
                  "-DSYSTEMC_PATH=${ciPkgs.lib.getDev ciPkgs.systemc}/include"
                ];
              preConfigure = (old.preConfigure or "") + ''
                export CIRCT_SOURCE_ROOT="$PWD"
                export PATH="${python}/bin:$PATH"
                # sccache's local LRU updates entry mtimes. Each Nix sandbox
                # uses a different nixbld UID, so import the shared staging
                # cache into a build-owned directory before starting sccache.
                export SCCACHE_DIR="$NIX_BUILD_TOP/.sccache"
                mkdir -p "$SCCACHE_DIR"
                if [[ -d /var/cache/circt-sccache \
                      && -r /var/cache/circt-sccache ]]; then
                  cp -R --no-preserve=ownership,mode,timestamps \
                    /var/cache/circt-sccache/. "$SCCACHE_DIR/"
                fi
                export SCCACHE_BASEDIRS="$NIX_BUILD_TOP"
                export SCCACHE_CACHE_SIZE=1G
                export SCCACHE_ERROR_LOG="''${TMPDIR:-/build}/sccache-errors.log"
                export SCCACHE_IDLE_TIMEOUT=0
                export SCCACHE_IGNORE_SERVER_IO_ERROR=1
                # The variables also cover nested CMake projects in postCheck.
                export CMAKE_C_COMPILER_LAUNCHER="${ciPkgs.sccache}/bin/sccache"
                export CMAKE_CXX_COMPILER_LAUNCHER="${ciPkgs.sccache}/bin/sccache"
              '';
              doCheck = true;
              checkTarget = pkgs.lib.concatStringsSep " " (
                [
                  "check-circt"
                  "check-circt-unit"
                  "check-circt-capi"
                  "check-circt-integration"
                ]
                ++ extraCheckTargets
              );
              preCheck =
                (old.preCheck or "")
                + ''
                  # integration_test adds these source-tree wrappers directly
                  # to PATH. Patch their /usr/bin/env shebangs inside the Nix
                  # sandbox before lit tries to execute them.
                  patchShebangs \
                    "$CIRCT_SOURCE_ROOT/utils/circt-lec.sh" \
                    "$CIRCT_SOURCE_ROOT/utils/equiv-rtl.sh"
                  # Verilator's generated makefiles invoke g++ directly and
                  # its timing support relies on GCC's coroutine flags.  A
                  # direct PATH reference avoids activating a second compiler
                  # setup hook during the Clang configure/build phases.
                  export PATH="${ciPkgs.gcc}/bin:$PATH"
                  export CPLUS_INCLUDE_PATH="${ciPkgs.lib.getDev ciPkgs.systemc}/include:${ciPkgs.zlib.dev}/include:''${CPLUS_INCLUDE_PATH:-}"
                  export LIBRARY_PATH="${ciPkgs.lib.getLib ciPkgs.zlib}/lib:''${LIBRARY_PATH:-}"
                  for tool in clang-tidy iverilog sby verilator yosys \
                    yosys-abc z3
                  do
                    command -v "$tool" >/dev/null
                  done
                  test -r "${ciPkgs.lib.getDev ciPkgs.systemc}/include/systemc"
                ''
                + pkgs.lib.optionalString pycde ''
                  export PYTHONPATH="$PWD/python_packages/pycde:$PWD/lib/Dialect/ESI/runtime/python"
                  runtime_build="$PWD/lib/Dialect/ESI/runtime"
                  export PATH="$PWD/bin:$runtime_build:$PATH"
                  # ESI's C++ integration fixtures run nested CMake projects.
                  # FetchContent uses the immutable CLI11 source directly in
                  # Nix, so expose its include root to those nested searches.
                  export CMAKE_INCLUDE_PATH="${cli11-src}/include:''${CMAKE_INCLUDE_PATH:-}"
                  export LD_LIBRARY_PATH="$PWD/lib:$runtime_build:''${LD_LIBRARY_PATH:-}"
                  export LIBRARY_PATH="$PWD/lib:$runtime_build:''${LIBRARY_PATH:-}"
                  export ESI_RUNTIME_TESTS_BIN="$PWD/lib/Dialect/ESI/runtime/tests/cpp/ESIRuntimeCppTests"
                '';
              postCheck =
                (old.postCheck or "")
                + ''
                  cmake -G Ninja \
                    -S "$CIRCT_SOURCE_ROOT/examples/circt-standalone" \
                    -B "$PWD/circt-standalone" \
                    -DCMAKE_BUILD_TYPE=Release \
                    -DCIRCT_DIR="$PWD/lib/cmake/circt" \
                    -DMLIR_DIR="${pkgs.lib.getDev mlir}/lib/cmake/mlir" \
                    -DLLVM_EXTERNAL_LIT="${python}/bin/.lit-wrapped" \
                    -DPython_EXECUTABLE="${python}/bin/python3" \
                    -DPython3_EXECUTABLE="${python}/bin/python3"
                  cmake --build "$PWD/circt-standalone" \
                    --target check-circt-standalone
                ''
                + pkgs.lib.optionalString pycde ''
                  runtime_prefix="$PWD/lib/Dialect/ESI/runtime/python/esiaccel"
                  cmake --install . \
                    --prefix "$runtime_prefix" \
                    --component ESIRuntime
                  # The Nix CMake hook gives CMAKE_INSTALL_INCLUDEDIR an
                  # absolute $dev path, so --prefix cannot redirect these
                  # runtime headers beside esiaccelConfig.cmake for pytest.
                  mkdir -p "$runtime_prefix/include"
                  cp -r "$CIRCT_SOURCE_ROOT/lib/Dialect/ESI/runtime/cpp/include/esi" \
                    "$runtime_prefix/include/esi"
                  test -r "$runtime_prefix/include/esi/Accelerator.h"
                  test -x "$ESI_RUNTIME_TESTS_BIN"
                  command -v esiquery >/dev/null
                  command -v esitester >/dev/null
                  python3 -c 'import cocotb, cocotb_test, pycde'
                  python3 -m pytest "$CIRCT_SOURCE_ROOT/lib/Dialect/ESI/runtime/tests" \
                    -v --log-cli-level=INFO

                  # Preserve the legacy standalone-runtime packaging coverage
                  # inside the Nix check without rerunning every cosim test.
                  standalone_runtime_build="$PWD/esi-runtime-standalone"
                  standalone_runtime_prefix="$PWD/esi-runtime-install"
                  cmake -G Ninja \
                    -S "$CIRCT_SOURCE_ROOT/lib/Dialect/ESI/runtime" \
                    -B "$standalone_runtime_build" \
                    -DBUILD_TESTING=ON \
                    -DCMAKE_BUILD_TYPE=Release \
                    -DCMAKE_INSTALL_INCLUDEDIR=include \
                    -DCMAKE_INSTALL_LIBDIR=lib \
                    -DESI_COSIM=ON \
                    -DESI_RUNTIME_TRACE=ON \
                    -DFETCHCONTENT_SOURCE_DIR_CLI11_PROJ=${cli11-src} \
                    -DFETCHCONTENT_SOURCE_DIR_FMT=${fmt-src} \
                    -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${googletest-src} \
                    -DFETCHCONTENT_SOURCE_DIR_IXWEBSOCKET=${ixwebsocket-src} \
                    -DFETCHCONTENT_SOURCE_DIR_JSON=${nlohmann-json-src} \
                    -DFETCHCONTENT_SOURCE_DIR_ZLIB=${zlib-src} \
                    -DPython_EXECUTABLE=${python}/bin/python3 \
                    -DPython3_EXECUTABLE=${python}/bin/python3
                  cmake --build "$standalone_runtime_build" \
                    --target ESIRuntime ESIRuntimeCppTests
                  cmake --install "$standalone_runtime_build" \
                    --prefix "$standalone_runtime_prefix" \
                    --component ESIRuntime
                  test -r "$standalone_runtime_prefix/include/esi/Accelerator.h"
                  test -r "$standalone_runtime_prefix/cmake/esiaccelConfig.cmake"
                  test -x "$standalone_runtime_prefix/bin/esiquery"
                  env \
                    ESI_RUNTIME_TESTS_BIN="$standalone_runtime_build/tests/cpp/ESIRuntimeCppTests" \
                    LD_LIBRARY_PATH="$standalone_runtime_prefix/lib:$PWD/lib:''${LD_LIBRARY_PATH:-}" \
                    LIBRARY_PATH="$standalone_runtime_prefix/lib:$PWD/lib:''${LIBRARY_PATH:-}" \
                    PATH="$standalone_runtime_prefix/bin:$PATH" \
                    PYTHONPATH="$standalone_runtime_prefix:$PWD/python_packages/pycde" \
                    python3 -m pytest \
                      "$CIRCT_SOURCE_ROOT/lib/Dialect/ESI/runtime/tests/unit" \
                      -v --log-cli-level=INFO
                ''
                + ''
                  ${ciPkgs.sccache}/bin/sccache --show-stats
                  ${ciPkgs.sccache}/bin/sccache --stop-server >/dev/null
                  if [[ -s "$SCCACHE_ERROR_LOG" ]]; then
                    echo "Last 200 lines from the sccache error log:" >&2
                    tail -n 200 "$SCCACHE_ERROR_LOG" >&2
                  fi
                  # Replace the staging cache with the LRU-bounded local
                  # snapshot. Merging would resurrect entries sccache evicted.
                  if [[ -d /var/cache/circt-sccache \
                        && -w /var/cache/circt-sccache ]]; then
                    find /var/cache/circt-sccache -mindepth 1 -delete
                    cp -R --no-preserve=ownership,mode,timestamps \
                      "$SCCACHE_DIR/." /var/cache/circt-sccache/
                  fi
                '';
              passthru = (old.passthru or { }) // {
                inherit libllvm mlir;
              };
            });

          # Normal CI directly reuses these cached libllvm and mlir derivations.
          fullCirct = mkCirct {
            buildSharedLibs = true;
            enableAssertions = true;
          };

          clangCirct = mkCirct {
            stdenv = ciPkgs.llvmPackages_21.stdenv;
            buildSharedLibs = false;
            # Assertion mode is part of the shared LLVM/MLIR derivation. Keep
            # it identical across variants while retaining the Clang/static
            # CIRCT configuration from the previous matrix.
            enableAssertions = true;
          };
        in
        {
          full = mkCirctCheck {
            name = "full-check";
            circt = fullCirct;
            inherit mlir;
            pycde = true;
            extraCmakeFlags = [
              "-DCIRCT_BINDINGS_PYTHON_ENABLED=ON"
              "-DCIRCT_ENABLE_FRONTENDS=PyCDE"
              "-DESI_COSIM=ON"
              "-DESI_RUNTIME=ON"
              "-DESI_RUNTIME_TRACE=ON"
              "-DFETCHCONTENT_SOURCE_DIR_CLI11_PROJ=${cli11-src}"
              "-DFETCHCONTENT_SOURCE_DIR_FMT=${fmt-src}"
              "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${googletest-src}"
              "-DFETCHCONTENT_SOURCE_DIR_IXWEBSOCKET=${ixwebsocket-src}"
              "-DFETCHCONTENT_SOURCE_DIR_JSON=${nlohmann-json-src}"
              "-DFETCHCONTENT_SOURCE_DIR_ZLIB=${zlib-src}"
              "-DMLIR_ENABLE_BINDINGS_PYTHON=ON"
              # zlib is an ESI build dependency, not part of the CIRCT
              # package.  Its generated pkg-config file combines absolute
              # multi-output paths into an invalid prefix during fixup.
              "-DZLIB_INSTALL=OFF"
            ];
            extraCheckTargets = [
              "check-pycde"
              "check-pycde-integration"
              "ESIRuntime"
              "ESIRuntimeCppTests"
            ];
          };
          clang = mkCirctCheck {
            name = "clang-check";
            circt = clangCirct;
            inherit mlir;
            extraCmakeFlags = [
              # GCC-built shared MLIR and Clang-built CIRCT derive different
              # implicit trait TypeIDs.  The full check covers every Python
              # test with a compiler-consistent stack; keep this variant for
              # Clang/static/reverse-iteration coverage.
              "-DCIRCT_BINDINGS_PYTHON_ENABLED=OFF"
              "-DLLVM_ENABLE_REVERSE_ITERATION=ON"
              "-DMLIR_ENABLE_BINDINGS_PYTHON=OFF"
            ];
          };
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python3.withPackages (ps: [
            ps.yapf
          ]);
        in
        {
          # This shell is only for cheap repository linting. Builds and tests
          # use the derivations above so LLVM is never rebuilt in a dev shell.
          lint = pkgs.mkShell {
            packages = with pkgs; [
              clang-tools
              git
              llvmPackages_21.stdenv.cc.cc.python
              python
            ];
          };
          default = self.devShells.${system}.lint;
        }
      );
    };
}
