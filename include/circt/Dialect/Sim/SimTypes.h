//===- SimTypes.h - Sim dialect types ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMTYPES_H
#define CIRCT_DIALECT_SIM_SIMTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "circt/Dialect/HW/HWEnums.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Sim/SimTypes.h.inc"

#endif // CIRCT_DIALECT_SIM_SIMTYPES_H
