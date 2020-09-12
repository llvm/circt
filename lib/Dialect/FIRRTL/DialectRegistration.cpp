//===- DialectRegistration.cpp - Register FIRRTL dialect ------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/Dialect.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
using namespace circt;
using namespace firrtl;

// Static initialization for FIRRTL dialect registration.
static mlir::DialectRegistration<FIRRTLDialect> FIRRTLOps;

/// Register all of the FIRRTL transformation passes with the PassManager.
namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/FIRRTL/FIRRTLPasses.h.inc"
} // namespace
void circt::firrtl::registerFIRRTLPasses() { registerPasses(); }
