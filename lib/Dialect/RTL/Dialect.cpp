//===- Dialect.cpp - Implement the RTL dialect ----------------------------===//
//
//===----------------------------------------------------------------------===//

#include "cirt/Dialect/RTL/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

using namespace cirt;
using namespace rtl;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

RTLDialect::RTLDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {}

RTLDialect::~RTLDialect() {}
