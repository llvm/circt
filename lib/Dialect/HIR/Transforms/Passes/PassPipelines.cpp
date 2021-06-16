#include "circt/Dialect/HIR/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

void mlir::hir::registerHIRLoweringPassPipeline() {
  // Register an inline pipeline builder.
  PassPipelineRegistration<>(
      "hir-simplify",
      "Simplify HIR dialect to a bare minimum for lowering to verilog.",
      [](OpPassManager &pm) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(createMemrefLoweringPass());
        pm.addPass(mlir::createCSEPass());
      });
}
