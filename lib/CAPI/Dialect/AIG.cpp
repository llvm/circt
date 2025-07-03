//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/AIG.h"
#include "circt/Dialect/AIG/AIGDialect.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(AIG, aig, circt::aig::AIGDialect)

void registerAIGPasses() { circt::aig::registerPasses(); }
