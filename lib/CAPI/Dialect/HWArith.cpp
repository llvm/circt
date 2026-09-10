//===- HWArith.cpp - C interface for the HWArith dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/HWArith.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

void registerHWArithPasses() { circt::registerHWArithToHWPass(); }
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HWArith, hwarith,
                                      circt::hwarith::HWArithDialect)
