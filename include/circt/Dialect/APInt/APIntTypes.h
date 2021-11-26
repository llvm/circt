//===- APIntTypes.h - types for the APInt dialect ---------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Types for APInt are mostly in tablegen. This file should contain C++ types
// used in MLIR type parameters and other supporting declarations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_APINT_APINTTYPES_H
#define CIRCT_DIALECT_APINT_APINTTYPES_H

#include "circt/Support/LLVM.h"

#include "APIntDialect.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/APInt/APIntTypes.h.inc"

#endif
