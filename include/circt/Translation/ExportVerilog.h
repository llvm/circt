//===- ExportVerilog.h - Verilog Exporter -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the Verilog emitter.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TRANSLATION_EXPORTVERILOG_H
#define CIRCT_TRANSLATION_EXPORTVERILOG_H

#include <functional>

namespace llvm {
class raw_ostream;
class StringRef;
} // namespace llvm

namespace mlir {
struct LogicalResult;
class ModuleOp;
} // namespace mlir

namespace circt {

/// Export a module containing HW, and SV dialect code. Requires that the SV
/// dialect is loaded in to the context.
mlir::LogicalResult exportVerilog(mlir::ModuleOp module, llvm::raw_ostream &os);

/// Export a module containing HW, and SV dialect code, as one file per SV
/// module. Requires that the SV dialect is loaded in to the context.
///
/// Files are created in the directory indicated by \p dirname.
mlir::LogicalResult exportSplitVerilog(mlir::ModuleOp module,
                                       llvm::StringRef dirname);

/// Register a translation for exporting HW, Comb and SV to SystemVerilog.
void registerToVerilogTranslation();

} // namespace circt

#endif // CIRCT_TRANSLATION_EXPORTVERILOG_H
