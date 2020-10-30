//=====- SheduleVerifier.h - HIR schedule verifier ---------------*- C++-*-===//
//
// Verify that the time schedule in the ir is valid.
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

/// Register the schedule verifier for HIR dialect.
void registerHIRScheduleVerifier();

} // namespace hir.
} // namespace mlir.

#endif // CIRCT_TARGET_VERILOG_HIRTOVERILOG_H.
