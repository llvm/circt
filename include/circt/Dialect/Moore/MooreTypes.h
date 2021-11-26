//===- MooreTypes.h - Declare Moore dialect types ----------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types for the Moore dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOORETYPES_H
#define CIRCT_DIALECT_MOORE_MOORETYPES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Moore/MooreTypes.h.inc"

#endif // CIRCT_DIALECT_MOORE_MOORETYPES_H
