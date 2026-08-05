//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4TYPES_H
#define CIRCT_DIALECT_AXI4_AXI4TYPES_H

#include "circt/Dialect/AXI4/AXI4Attributes.h"
#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/AXI4/AXI4Types.h.inc"

#endif // CIRCT_DIALECT_AXI4_AXI4TYPES_H
