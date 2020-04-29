//===- EmitVerilog.h - Verilog Emitter --------------------------*- C++ -*-===//
//
// Defines the interface to the Verilog emitter.
//
//===----------------------------------------------------------------------===//

#ifndef SPT_EMIT_VERILOG_H
#define SPT_EMIT_VERILOG_H

#include "cirt/Support/LLVM.h"

namespace mlir {
struct LogicalResult;
class ModuleOp;
} // namespace mlir

namespace cirt {

mlir::LogicalResult emitVerilog(mlir::ModuleOp module, llvm::raw_ostream &os);

void registerVerilogEmitterTranslation();

} // namespace cirt

#endif // SPT_EMIT_VERILOG_H
