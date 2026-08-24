//===- PIROps.h - PIR dialect operations ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PIR_PIROPS_H
#define CIRCT_DIALECT_PIR_PIROPS_H

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/PIR/PIRDialect.h"
#include "circt/Dialect/PIR/PIRTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_OP_CLASSES
#include "circt/Dialect/PIR/PIR.h.inc"

#endif // CIRCT_DIALECT_PIR_PIROPS_H
