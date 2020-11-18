//===- ESIToRTL.cpp - ESI to RTL/SV conversion passes -----------*- C++ -*-===//
//
// Lower ESI to RTL and SV.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ESIToRTL/ESIToRTL.h"
#include "circt/Dialect/RTL/Dialect.h"
#include "circt/Dialect/SV/Dialect.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <memory>

#define DEBUG_TYPE "esi-to-rtl"

namespace circt {
namespace esi {
#define GEN_PASS_CLASSES
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace esi
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::esi;

namespace {
struct ESIToRTLLoweringPass : public LowerESIToRTLBase<ESIToRTLLoweringPass> {

  void runOnOperation() override;
};
} // end anonymous namespace.

void ESIToRTLLoweringPass::runOnOperation() { llvm::outs() << "test!\n"; }

namespace circt {
namespace esi {
/// Create a FIRRTL to LLHD conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createESILoweringPass() {
  return std::make_unique<ESIToRTLLoweringPass>();
}
} // namespace esi
} // namespace circt

/// Register the FIRRTL to LLHD convesion pass.
namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace

void circt::esi::registerESIToRTLPasses() { registerPasses(); }
