//=====- BitwidthReductionPass.h - HIR biwidth reduction pass -----*-C++-*-===//
//
// Reduce bitwidth of variables when it does not affect program behaviour.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TARGET_VERILOG_HIRBITWIDTHREDUCTION_H
#define CIRCT_TARGET_VERILOG_HIRBITWIDTHREDUCTION_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {

struct LogicalResult;
class ModuleOp;

namespace hir {

/// Register the bitwidth reduction pass for HIR dialect.
void registerBitwidthReductionPass();

} // namespace hir.
} // namespace mlir.

#endif // CIRCT_TARGET_VERILOG_HIRBITWIDTHREDUCTION_H.
