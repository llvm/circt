//===- DialectRegistration.cpp - Register FIRRTL dialect ------------------===//
//
//===----------------------------------------------------------------------===//

#include "cirt/Dialect/FIRRTL/Dialect.h"
#include "cirt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
using namespace cirt;
using namespace firrtl;

// Static initialization for FIRRTL dialect registration.
static mlir::DialectRegistration<FIRRTLDialect> FIRRTLOps;

/// Register all of the FIRRTL transformation passes with the PassManager.
void cirt::firrtl::registerFIRRTLPasses() {
  using namespace mlir;
  registerPass(
      "lower-firrt-to-rtl", "Lower FIRRTL to RTL",
      []() -> std::unique_ptr<Pass> { return createLowerFIRRTLToRTLPass(); });
}
