//===- FSMTypes.h - FSM dialect types definition file ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FSM_FSMTYPES_H
#define CIRCT_DIALECT_FSM_FSMTYPES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/FSM/FSMTypes.h.inc"

#endif // CIRCT_DIALECT_FSM_FSMTYPES_H
