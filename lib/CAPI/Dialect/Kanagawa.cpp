//===- Kanagawa.cpp - C interface for the Kanagawa dialect ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Kanagawa.h"
#include "circt/Dialect/Kanagawa/KanagawaDialect.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"

#include "mlir/CAPI/Registration.h"

using namespace circt::kanagawa;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Kanagawa, kanagawa,
                                      circt::kanagawa::KanagawaDialect)

void registerKanagawaPasses() { circt::kanagawa::registerPasses(); }
