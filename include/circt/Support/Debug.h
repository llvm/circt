//===- Debug.h - Debug Utilities --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities related to generating run-time debug information.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_DEBUG_H
#define CIRCT_SUPPORT_DEBUG_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Twine.h"

namespace circt {

/// Write a "header"-like string to the debug stream with a certain width.  This
/// produces output like the following:
///
///     ===- Hello World --------------------===
///
/// This is commonly used for generating a header in debug information.  The
/// format is modeled after LLVM/MLIR/CIRCT source file headers.
llvm::raw_ostream &debugHeader(const llvm::Twine &str, unsigned width = 80);

/// Write a boilerplate header for a pass to the debug stream.  This generates
/// output like the following if the pass's name is "FooPass":
///
///    ===- Running FooPass -----------------===
///
/// This is commonly used to generate a header in debug when a pass starts
/// running.
llvm::raw_ostream &debugPassHeader(const mlir::Pass *pass, unsigned width = 80);

/// Write a boilerplate footer to the debug stream to indicate that a pass has
/// ended.  This produces text like the following:
///
///    ===-----------------------------------===
llvm::raw_ostream &debugFooter(unsigned width = 80);

/// RAII helper for creating a pass header and footer automatically.  The heaer
/// is printed on construction and the footer on destruction.
class ScopedDebugPassLogger {

public:
  ScopedDebugPassLogger(const mlir::Pass *pass, unsigned width = 80)
      : width(width) {
    debugPassHeader(pass, width) << "\n";
  }

  ~ScopedDebugPassLogger() {
    debugFooter(width) << "\n";
  }

private:
  unsigned width;
};

} // namespace circt

#endif // CIRCT_SUPPORT_DEBUG_H
