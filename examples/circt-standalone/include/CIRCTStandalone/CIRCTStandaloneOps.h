//===- CIRCTStandaloneOps.h - CIRCT standalone ops -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STANDALONE_CIRCTSTANDALONEOPS_H
#define CIRCT_STANDALONE_CIRCTSTANDALONEOPS_H

#include "CIRCTStandalone/CIRCTStandaloneDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "CIRCTStandalone/CIRCTStandaloneOps.h.inc"

#endif // CIRCT_STANDALONE_CIRCTSTANDALONEOPS_H
