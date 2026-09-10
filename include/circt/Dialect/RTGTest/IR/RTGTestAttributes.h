//===- RTGTestAttributes.h - RTG Test dialect attributes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTGTEST_IR_RTGTESTATTRIBUTES_H
#define CIRCT_DIALECT_RTGTEST_IR_RTGTESTATTRIBUTES_H

#include "circt/Dialect/RTGTest/IR/RTGTestTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "circt/Dialect/RTG/IR/RTGAttrInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGISAAssemblyAttrInterfaces.h"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/RTGTest/IR/RTGTestAttributes.h.inc"

#endif // CIRCT_DIALECT_RTGTEST_IR_RTGTESTATTRIBUTES_H
