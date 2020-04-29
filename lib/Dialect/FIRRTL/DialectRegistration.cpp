//===- DialectRegistration.cpp - Register FIRRTL dialect ------------------===//
//
//===----------------------------------------------------------------------===//

#include "cirt/Dialect/FIRRTL/Dialect.h"
using namespace cirt;
using namespace firrtl;

// Static initialization for FIRRTL dialect registration.
static mlir::DialectRegistration<FIRRTLDialect> FIRRTLOps;
