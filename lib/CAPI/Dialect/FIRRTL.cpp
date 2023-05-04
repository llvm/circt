//===- FIRRTL.cpp - C Interface for the FIRRTL Dialect --------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/FIRRTL.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FIRRTL, firrtl,
                                      circt::firrtl::FIRRTLDialect)
