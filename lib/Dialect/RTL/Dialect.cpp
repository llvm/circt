//===- Dialect.cpp - Implement the RTL dialect ----------------------------===//
//
//===----------------------------------------------------------------------===//

#include "cirt/Dialect/RTL/Dialect.h"
#include "cirt/Dialect/RTL/Ops.h"
#include "mlir/IR/DialectImplementation.h"

using namespace cirt;
using namespace rtl;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

RTLDialect::RTLDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "cirt/Dialect/RTL/RTL.cpp.inc"
      >();
}

RTLDialect::~RTLDialect() {}
