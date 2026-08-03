//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4OPS_H
#define CIRCT_DIALECT_AXI4_AXI4OPS_H

#include "mlir/IR/OpImplementation.h"

#include "circt/Dialect/AXI4/AXI4Dialect.h"

#define GET_OP_CLASSES
#include "circt/Dialect/AXI4/AXI4.h.inc"

#endif // CIRCT_DIALECT_AXI4_AXI4OPS_H
