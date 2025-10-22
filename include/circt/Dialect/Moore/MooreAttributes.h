//===- MooreAttributes.h - Declare Moore dialect attributes ------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the attributes for the Moore dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREATTRIBUTES_H
#define CIRCT_DIALECT_MOORE_MOOREATTRIBUTES_H

#include "circt/Support/FVInt.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringExtras.h"

#include "circt/Dialect/Moore/MooreEnums.h.inc"
// Include generated attributes.
#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Moore/MooreAttributes.h.inc"

#endif // CIRCT_DIALECT_MOORE_MOOREATTRIBUTES_H
