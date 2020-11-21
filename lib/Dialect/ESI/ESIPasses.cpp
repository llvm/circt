//===- ESIToRTL.cpp - ESI to RTL/SV conversion passes -----------*- C++ -*-===//
//
// Lower ESI to RTL and SV.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
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
struct ESIToRTLPass : public LowerESIToRTLBase<ESIToRTLPass> {

  void runOnOperation() override;
};
} // end anonymous namespace.

void ESIToRTLPass::runOnOperation() { llvm::outs() << "test!\n"; }

namespace circt {
namespace esi {
std::unique_ptr<OperationPass<ModuleOp>> createESILoweringPass() {
  return std::make_unique<ESIToRTLPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createESIToRTLPass() {
  return std::make_unique<ESIToRTLPass>();
}
} // namespace esi
} // namespace circt

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/ESI/ESIPasses.h.inc"
} // namespace

void circt::esi::registerESIPasses() { registerPasses(); }
