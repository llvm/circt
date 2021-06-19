//===- CalyxAttributes.h - Calyx dialect attributes -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Calyx dialect custom attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYX_CALYXATTRIBUTES_H
#define CIRCT_DIALECT_CALYX_CALYXATTRIBUTES_H

#include "circt/Dialect/Calyx/CalyxDialect.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinAttributes.h"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Calyx/CalyxAttributes.h.inc"

#endif // CIRCT_DIALECT_CALYX_CALYXATTRIBUTES_H
