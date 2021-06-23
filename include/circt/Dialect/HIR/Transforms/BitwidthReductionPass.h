//=====- BitwidthReductionPass.h - HIR biwidth reduction pass -----*-C++-*-===//
//
// Reduce bitwidth of variables when it does not affect program behaviour.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TARGET_VERILOG_HIRBITWIDTHREDUCTION_H
#define CIRCT_TARGET_VERILOG_HIRBITWIDTHREDUCTION_H

namespace circt {
class raw_ostream;
} // namespace circt

namespace circt {

struct LogicalResult;
class ModuleOp;

namespace hir {

/// Register the bitwidth reduction pass for HIR dialect.
void registerBitwidthReductionPass();

} // namespace hir.
} // namespace circt

#endif // CIRCT_TARGET_VERILOG_HIRBITWIDTHREDUCTION_H.
