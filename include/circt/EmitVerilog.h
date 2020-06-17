//===- EmitVerilog.h - Verilog Emitter --------------------------*- C++ -*-===//
//
// Defines the interface to the Verilog emitter.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_EMIT_VERILOG_H
#define CIRCT_EMIT_VERILOG_H

#include "circt/Support/LLVM.h"

namespace mlir {
struct LogicalResult;
class ModuleOp;
} // namespace mlir

namespace circt {

mlir::LogicalResult emitVerilog(mlir::ModuleOp module, llvm::raw_ostream &os);

void registerVerilogEmitterTranslation();

} // namespace circt

#endif // CIRCT_EMIT_VERILOG_H
