//===- ProbeOps.h - Probe dialect operations --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PROBE_PROBEOPS_H
#define CIRCT_DIALECT_PROBE_PROBEOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/Probe/ProbeDialect.h"
#include "circt/Dialect/Probe/ProbeTypes.h"

// Operation definitions generated from `Probe.td`
#define GET_OP_CLASSES
#include "circt/Dialect/Probe/Probe.h.inc"

#endif // CIRCT_DIALECT_PROBE_PROBEOPS_H
