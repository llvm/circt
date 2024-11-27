//===- RTGTypes.h - RTG dialect types ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTGTYPES_H
#define CIRCT_DIALECT_RTG_IR_RTGTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/RTG/IR/RTGTypes.h.inc"

#endif // CIRCT_DIALECT_RTG_IR_RTGTYPES_H
