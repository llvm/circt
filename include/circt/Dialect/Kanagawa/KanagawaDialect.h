//===- KanagawaDialect.h - Definition of Kanagawa dialect -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_KANAGAWA_KANAGAWADIALECT_H
#define CIRCT_DIALECT_KANAGAWA_KANAGAWADIALECT_H

#include "mlir/IR/Dialect.h"

// Pull in the dialect definition.
#include "circt/Dialect/Kanagawa/KanagawaDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
#include "circt/Dialect/Kanagawa/KanagawaEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Kanagawa/KanagawaAttributes.h.inc"

#endif // CIRCT_DIALECT_KANAGAWA_KANAGAWADIALECT_H
