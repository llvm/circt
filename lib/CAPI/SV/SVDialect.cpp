//===- SVDialect.cpp - C Interface for the SV Dialect -------------------===//
//
//  Implements a C Interface for the SV Dialect
//
//===----------------------------------------------------------------------===//

#include "circt-c/SVDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

void mlirContextRegisterSVDialect(MlirContext context) {
  unwrap(context)->getDialectRegistry().insert<circt::sv::SVDialect>();
}

MlirDialect mlirContextLoadSVDialect(MlirContext context) {
  return wrap(unwrap(context)->getOrLoadDialect<circt::sv::SVDialect>());
}

MlirStringRef mlirSVDialectGetNamespace() {
  return wrap(circt::sv::SVDialect::getDialectNamespace());
}
