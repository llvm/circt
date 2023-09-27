//===- ESIAttributes.h - attributes for the ESI dialect ---------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESIATTRIBUTES_H
#define CIRCT_DIALECT_ESI_ESIATTRIBUTES_H

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "ESIDialect.h"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/ESI/ESIAttributes.h.inc"

#endif
