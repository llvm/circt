//===- DebugInfoUtils.h - Helpers for debug info emission -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_TARGET_DEBUGINFO_DEBUGINFOUTILS_H
#define CIRCT_TARGET_DEBUGINFO_DEBUGINFOUTILS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Location.h"

namespace circt {

struct DIInstance;
struct DIModule;

/// Find the best location to report as source location ("HGL", emitted = false)
/// or as emitted location ("HDL", emitted = true). Returns any non-FIR file it
/// finds, and only falls back to FIR files if nothing else is found.
FileLineColLoc findBestLocation(Location loc, bool emitted, bool fileMustExist);

/// Return the name of the Verilog module emitted for `module`.
StringAttr getVerilogModuleName(DIModule &module);

/// Return the name of the Verilog instance emitted for `inst`.
StringAttr getVerilogInstanceName(DIInstance &inst);

} // namespace circt

#endif // CIRCT_TARGET_DEBUGINFO_DEBUGINFOUTILS_H
