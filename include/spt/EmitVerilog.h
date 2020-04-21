//===- EmitVerilog.h - Verilog Emitter --------------------------*- C++ -*-===//
//
// Defines the interface to the Verilog emitter.
//
//===----------------------------------------------------------------------===//

#ifndef SPT_EMIT_VERILOG_H
#define SPT_EMIT_VERILOG_H

#include "spt/Support/LLVM.h"

namespace mlir {
struct LogicalResult;
class ModuleOp;
} // namespace mlir

namespace spt {

mlir::LogicalResult emitVerilog(mlir::ModuleOp module, llvm::raw_ostream &os);

void registerVerilogEmitterTranslation();

} // namespace spt

#endif // SPT_EMIT_VERILOG_H
