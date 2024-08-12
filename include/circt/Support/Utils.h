//===- Utils.h - Miscellaneous utilities ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Miscellaneous utilities for CIRCT that do not fit in with other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_UTILS_H
#define CIRCT_SUPPORT_UTILS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace circt {
/// Return true if a Value is created "underneath" an operation.  This is
/// frequently useful when negated as that indicates that a Value is defined
/// "outside" the region of an Operation and that the Operation is thereby
/// "capturing" the value.
inline bool isAncestorOfValueOwner(Operation *op, Value value) {
  return op->isAncestor(value.getParentBlock()->getParentOp());
}
} // namespace circt

#endif // CIRCT_SUPPORT_UTILS_H
