//===- TranslateToVerilog.h - Verilog Printer -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the LLHD to Verilog Printer.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_TRANSLATION_TRANSLATETOVERILOG_H
#define CIRCT_DIALECT_LLHD_TRANSLATION_TRANSLATETOVERILOG_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
struct LogicalResult;
class ModuleOp;
} // namespace mlir

namespace circt {
namespace llhd {

mlir::LogicalResult exportVerilog(mlir::ModuleOp module, llvm::raw_ostream &os);

void registerToVerilogTranslation();

} // namespace llhd
} // namespace circt

#endif // CIRCT_DIALECT_LLHD_TRANSLATION_TRANSLATETOVERILOG_H
