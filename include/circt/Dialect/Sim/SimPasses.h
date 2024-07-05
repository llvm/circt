#ifndef CIRCT_DIALECT_SIM_SIMPASSES_H
#define CIRCT_DIALECT_SIM_SIMPASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace sim {

#define GEN_PASS_DECL
#include "circt/Dialect/Sim/SimPasses.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Sim/SimPasses.h.inc"
} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_SIMPASSES_H
