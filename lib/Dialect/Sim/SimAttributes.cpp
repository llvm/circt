//===- SimAttributes.cpp - Implement Sim attributes -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimAttributes.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace sim;

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Sim/SimAttributes.cpp.inc"

namespace circt {
namespace sim {
LogicalResult NeverTriggerAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::Type type) {
  if (!isTriigerType(type)) {
    emitError() << "type of never attribute must be a trigger type.";
    return failure();
  }
  return success();
}
} // namespace sim
} // namespace circt

void SimDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/Sim/SimAttributes.cpp.inc"
      >();
}
