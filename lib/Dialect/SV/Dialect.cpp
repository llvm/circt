//===- Dialect.cpp - Implement the SV dialect -----------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/Dialect.h"
#include "circt/Dialect/SV/Ops.h"

using namespace circt;
using namespace sv;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

SVDialect::SVDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, getTypeID()) {

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/SV/SV.cpp.inc"
      >();
}

SVDialect::~SVDialect() {}
