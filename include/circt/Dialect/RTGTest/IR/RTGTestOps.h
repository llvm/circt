//===- RTGTestOps.h - Declare RTGTest dialect operations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the RTGTest dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTGTEST_IR_RTGTESTOPS_H
#define CIRCT_DIALECT_RTGTEST_IR_RTGTESTOPS_H

#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGISAAssemblyOpInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGOpInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTGTest/IR/RTGTestAttributes.h"
#include "circt/Dialect/RTGTest/IR/RTGTestDialect.h"
#include "circt/Dialect/RTGTest/IR/RTGTestTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/RTGTest/IR/RTGTest.h.inc"

#endif // CIRCT_DIALECT_RTGTEST_IR_RTGTESTOPS_H
