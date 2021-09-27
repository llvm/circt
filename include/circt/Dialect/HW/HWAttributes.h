//===- HWAttributes.h - Declare HW dialect attributes ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_ATTRIBUTES_H
#define CIRCT_DIALECT_HW_ATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace circt {
namespace hw {
class PEOAttr;
enum class PEO : uint32_t;
} // namespace hw
} // namespace circt

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/HW/HWAttributes.h.inc"

#endif // CIRCT_DIALECT_HW_ATTRIBUTES_H
