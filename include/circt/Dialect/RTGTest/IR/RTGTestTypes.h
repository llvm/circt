//===- RTGTestTypes.h - RTG Test dialect types ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTGTEST_IR_RTGTESTTYPES_H
#define CIRCT_DIALECT_RTGTEST_IR_RTGTESTTYPES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "circt/Dialect/RTG/IR/RTGISAAssemblyTypeInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGTypeInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/RTGTest/IR/RTGTestTypes.h.inc"

#endif // CIRCT_DIALECT_RTGTEST_IR_RTGTESTTYPES_H
