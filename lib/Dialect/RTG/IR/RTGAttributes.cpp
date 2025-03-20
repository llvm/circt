//===- RTGAttributes.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace rtg;

//===----------------------------------------------------------------------===//
// ImmediateAttr
//===----------------------------------------------------------------------===//

Type ImmediateAttr::getType() const {
  return ImmediateType::get(getContext(), getWidth());
}

LogicalResult
ImmediateAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      uint32_t width, uint64_t value) {

  if (static_cast<uint32_t>(llvm::bit_width(value)) > width)
    return emitError() << "cannot represent " << value << " with " << width
                       << " bits";

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

void RTGDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/RTG/IR/RTGAttributes.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/RTG/IR/RTGAttributes.cpp.inc"
