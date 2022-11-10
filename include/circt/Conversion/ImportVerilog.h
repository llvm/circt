//===- ImportVerilog.h - Slang Verilog frontend integration ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the Verilog frontend.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_IMPORTVERILOG_H
#define CIRCT_CONVERSION_IMPORTVERILOG_H

#include "circt/Support/LLVM.h"

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
class LocationAttr;
class TimingScope;
} // namespace mlir

namespace circt {

/// Parse files in a source manager as Verilog source code.
mlir::OwningOpRef<mlir::ModuleOp> importVerilog(llvm::SourceMgr &sourceMgr,
                                                mlir::MLIRContext *context,
                                                mlir::TimingScope &ts);

/// Register the `import-verilog` MLIR translation.
void registerFromVerilogTranslation();

} // namespace circt

#endif // CIRCT_CONVERSION_IMPORTVERILOG_H
