//===- Sim.cpp - C interface for the Sim dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Sim.h"
#include "circt/Dialect/Sim/SimDialect.h"

#include "mlir/CAPI/Registration.h"

using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Sim, sim, SimDialect)
