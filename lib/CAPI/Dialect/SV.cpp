//===- SVDialect.cpp - C Interface for the SV Dialect -------------------===//
//
//  Implements a C Interface for the SV Dialect
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/SV.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Transforms/Passes.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SystemVerilog, sv, circt::sv::SVDialect)

void registerSVPasses() {
  mlir::registerCanonicalizerPass();
  circt::sv::registerPasses();
}
