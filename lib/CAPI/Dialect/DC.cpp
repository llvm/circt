//===- DC.cpp - C interface for the DC dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/DC.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/DC/DCDialect.h"
#include "circt/Dialect/DC/DCPasses.h"
#include "circt/Transforms/Passes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

void registerDCPasses() {
  circt::registerMapArithToCombPass();
  circt::dc::registerPasses();
}
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(DC, dc, circt::dc::DCDialect)
