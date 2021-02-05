//===- RTLDialect.cpp - C Interface for the RTL Dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Implements a C Interface for the RTL Dialect
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/RTL.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(RTL, rtl, circt::rtl::RTLDialect)
