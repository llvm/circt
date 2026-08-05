//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Attributes.h"
#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace circt;
using namespace axi4;
using namespace mlir;

#include "circt/Dialect/AXI4/AXI4Enums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/AXI4/AXI4Attributes.cpp.inc"

LogicalResult
BurstSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      BurstKind kind, uint32_t len) {
  // AXI4 permits 1-16 beats for 'fixed', 1-256 for 'incr', and only 2, 4, 8,
  // or 16 for 'wrap'.
  switch (kind) {
  case BurstKind::Fixed:
    if (len == 0 || len > 16)
      return emitError() << "'fixed' burst 'len' must be between 1 and 16, got "
                         << len;
    break;
  case BurstKind::Incr:
    if (len == 0 || len > 256)
      return emitError() << "'incr' burst 'len' must be between 1 and 256, got "
                         << len;
    break;
  case BurstKind::Wrap:
    if (len < 2 || len > 16 || !llvm::isPowerOf2_32(len))
      return emitError() << "'wrap' burst 'len' must be 2, 4, 8, or 16, got "
                         << len;
    break;
  }
  return success();
}

void AXI4Dialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/AXI4/AXI4Attributes.cpp.inc"
      >();
}
