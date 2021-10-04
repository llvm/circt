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
namespace translations {

std::unique_ptr<mlir::Pass> createExportVerilogFilePass(llvm::raw_ostream &os);
std::unique_ptr<mlir::Pass> createExportVerilogFilePass();

std::unique_ptr<mlir::Pass>
createExportSplitVerilogPass(llvm::StringRef directory = "./");

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Translation/TranslationPasses.h.inc"

} // namespace translations

/// Export a module containing HW, and SV dialect code. Requires that the SV
/// dialect is loaded in to the context.
mlir::LogicalResult exportVerilog(mlir::ModuleOp module, llvm::raw_ostream &os);

/// Export a module containing HW, and SV dialect code, as one file per SV
/// module. Requires that the SV dialect is loaded in to the context.
///
/// Files are created in the directory indicated by \p dirname.
mlir::LogicalResult exportSplitVerilog(mlir::ModuleOp module,
                                       llvm::StringRef dirname);

} // namespace circt

#endif // CIRCT_TRANSLATION_EXPORTVERILOG_H
