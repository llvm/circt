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

#include "mlir/Pass/Pass.h"

namespace circt {

std::unique_ptr<mlir::Pass> createExportVerilogPass(llvm::raw_ostream &os);
std::unique_ptr<mlir::Pass> createExportVerilogPass();

std::unique_ptr<mlir::Pass>
createExportSplitVerilogPass(llvm::StringRef directory = "./");

/// Export a module containing HW, and SV dialect code. Requires that the SV
/// dialect is loaded in to the context.
/// If `separateModules` is set to `true`, this function will output the split
/// file header with every file (matching the semantics of
/// `exportSplitVerilog`). Otherwise, only files explicitly specified using the
/// `hw.output_file` attribute will emit a file header.
mlir::LogicalResult exportVerilog(mlir::ModuleOp module, bool separateModules,
                                  llvm::raw_ostream &os);

/// Export a module containing HW, and SV dialect code, as one file per SV
/// module. Requires that the SV dialect is loaded in to the context.
///
/// Files are created in the directory indicated by \p dirname.
mlir::LogicalResult exportSplitVerilog(mlir::ModuleOp module,
                                       llvm::StringRef dirname);

} // namespace circt

#endif // CIRCT_TRANSLATION_EXPORTVERILOG_H
