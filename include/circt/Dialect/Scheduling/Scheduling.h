//===- Scheduling.h - Scheduling dialect definition -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Scheduling dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SCHEDULING_H
#define CIRCT_DIALECT_SCHEDULING_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"

#include "circt/Dialect/Scheduling/SchedulingDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Scheduling/SchedulingAttributes.h.inc"

#endif // CIRCT_DIALECT_SCHEDULING_H
