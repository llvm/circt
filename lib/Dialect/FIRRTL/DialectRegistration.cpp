//===- DialectRegistration.cpp - Register FIRRTL dialect ------------------===//
//
//===----------------------------------------------------------------------===//

#include "spt/Dialect/FIRRTL/IR/Ops.h"
using namespace spt;
using namespace firrtl;

// Static initialization for FIRRTL dialect registration.
static mlir::DialectRegistration<FIRRTLDialect> FIRRTLOps;
