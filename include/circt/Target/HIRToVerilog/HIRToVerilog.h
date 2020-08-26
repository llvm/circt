//=====- HIRToVerilog.h - Verilog Printer -----------------------*- C++ -*-===//
//
// Defines the interface to the HIR to Verilog Printer.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TARGET_VERILOG_HIRTOVERILOG_H
#define CIRCT_TARGET_VERILOG_HIRTOVERILOG_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {

struct LogicalResult;
class ModuleOp;

namespace hir {

mlir::LogicalResult printVerilog(mlir::ModuleOp module, llvm::raw_ostream &os);

void registerHIRToVerilogTranslation();

} // namespace hir
} // namespace mlir

#endif // CIRCT_TARGET_VERILOG_TRANSLATETOVERILOG_H
