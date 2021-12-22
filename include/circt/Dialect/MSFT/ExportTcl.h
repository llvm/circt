//===- ExportTcl.h - MSFT Tcl Exporters -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Expose the Tcl exporters.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_EXPORTTCL_H
#define CIRCT_DIALECT_MSFT_EXPORTTCL_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
struct LogicalResult;
class ModuleOp;
} // namespace mlir

namespace circt {
namespace hw {
class SymbolCache;
} // namespace hw

namespace msft {
class MSFTModuleOp;

/// Export TCL for a specific hw module.
mlir::LogicalResult exportQuartusTcl(MSFTModuleOp module,
                                     llvm::StringRef outputFile = "");

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_EXPORTTCL_H
