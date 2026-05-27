//===- SvDpiRuntime.cpp - Runtime-loaded svdpi entry points ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provide the small svDpi* subset needed by DpiEntryPoints.cpp, but resolve
// the underlying simulator implementation at runtime instead of linking a
// fixed MtiPli import library.
//
//===----------------------------------------------------------------------===//

// On MSVC++ we need svdpi.h to declare exports, not imports.
#define DPI_PROTOTYPES
#undef XXTERN
#define XXTERN DPI_EXTERN DPI_DLLESPEC
#undef EETERN
#define EETERN DPI_EXTERN DPI_DLLESPEC

#include "svdpi.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#endif

namespace {

template <typename FnTy>
FnTy resolveSvDpi(const char *name) {
#ifdef _WIN32
  // Resolve from the host simulator executable. This supports the Verilator
  // flow where svDpi* symbols are exported by the generated simulator binary.
  auto *host = GetModuleHandleA(nullptr);
  void *sym =
      host ? reinterpret_cast<void *>(GetProcAddress(host, name)) : nullptr;
#else
  // Look up in the process-global namespace provided by the host simulator.
  // Use RTLD_DEFAULT first (simulator-exported/global symbols), then fall
  // back to the main-program handle for environments with stricter policies.
  void *sym = dlsym(RTLD_DEFAULT, name);
  if (!sym) {
    void *mainProg = dlopen(nullptr, RTLD_NOW);
    if (mainProg)
      sym = dlsym(mainProg, name);
  }
#endif
  if (!sym) {
#ifdef _WIN32
    OutputDebugStringA(
        "error: EsiCosimDpiServer failed to resolve required svdpi symbol '");
    OutputDebugStringA(name);
    OutputDebugStringA("'\n");
    TerminateProcess(GetCurrentProcess(), 1);
#else
    fprintf(stderr,
            "error: EsiCosimDpiServer failed to resolve required "
            "svdpi symbol '%s'\n",
            name);
    fflush(stderr);
    abort();
#endif
  }
  return reinterpret_cast<FnTy>(sym);
}

} // namespace

extern "C" int svDimensions(const svOpenArrayHandle h) {
  using FnTy = int (*)(const svOpenArrayHandle);
  static FnTy fn = resolveSvDpi<FnTy>("svDimensions");
  return fn(h);
}

extern "C" void *svGetArrayPtr(const svOpenArrayHandle h) {
  using FnTy = void *(*)(const svOpenArrayHandle);
  static FnTy fn = resolveSvDpi<FnTy>("svGetArrayPtr");
  return fn(h);
}

extern "C" int svSizeOfArray(const svOpenArrayHandle h) {
  using FnTy = int (*)(const svOpenArrayHandle);
  static FnTy fn = resolveSvDpi<FnTy>("svSizeOfArray");
  return fn(h);
}

extern "C" int svSize(const svOpenArrayHandle h, int d) {
  using FnTy = int (*)(const svOpenArrayHandle, int);
  static FnTy fn = resolveSvDpi<FnTy>("svSize");
  return fn(h, d);
}

extern "C" void *svGetArrElemPtr1(const svOpenArrayHandle h, int idx) {
  using FnTy = void *(*)(const svOpenArrayHandle, int);
  static FnTy fn = resolveSvDpi<FnTy>("svGetArrElemPtr1");
  return fn(h, idx);
}
