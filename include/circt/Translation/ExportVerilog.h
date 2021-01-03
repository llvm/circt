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

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
struct LogicalResult;
class ModuleOp;
} // namespace mlir

namespace circt {

/// Export a module containing RTL, and SV dialect code.
mlir::LogicalResult exportVerilog(mlir::ModuleOp module, llvm::raw_ostream &os);

/// Register a translation for exporting FIRRTL, RTL, and SV
void registerToVerilogTranslation();

/// Export a module containing RTL, and SV dialect code.
mlir::LogicalResult exportFIRRTLToVerilog(mlir::ModuleOp module,
                                          llvm::raw_ostream &os);
/// Register a translation for exporting FIRRTL, RTL, and SV
void registerFIRRTLToVerilogTranslation();

} // namespace circt

#endif // CIRCT_TRANSLATION_EXPORTVERILOG_H
