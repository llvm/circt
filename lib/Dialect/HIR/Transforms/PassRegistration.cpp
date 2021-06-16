#include "circt/Dialect/HIR/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/HIR/Transforms/Passes.h.inc"
} // namespace

void mlir::hir::initHIRTransformationPasses() {
  registerPasses();
  registerHIRLoweringPassPipeline();
}
