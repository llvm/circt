//===- Ibis.cpp - C interface for the Ibis dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Ibis.h"
#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

void registerIbisPasses() { circt::ibis::registerPasses(); }
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Ibis, ibis, circt::ibis::IbisDialect)
