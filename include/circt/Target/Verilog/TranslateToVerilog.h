//===- TranslateToVerilog.h - Verilog Printer -------------------*- C++ -*-===//
//
// Defines the interface to the LLHD to Verilog Printer.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TARGET_VERILOG_TRANSLATETOVERILOG_H
#define CIRCT_TARGET_VERILOG_TRANSLATETOVERILOG_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
struct LogicalResult;
class ModuleOp;
} // namespace mlir

namespace circt {
namespace llhd {

mlir::LogicalResult printVerilog(mlir::ModuleOp module, llvm::raw_ostream &os);

void registerToVerilogTranslation();

} // namespace llhd
} // namespace circt

#endif // CIRCT_TARGET_VERILOG_TRANSLATETOVERILOG_H
