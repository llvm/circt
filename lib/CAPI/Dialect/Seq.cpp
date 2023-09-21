//===- Seq.cpp - C Interface for the Seq Dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Seq.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Seq/SeqTypes.h"

#include "mlir/CAPI/Registration.h"

using namespace circt::seq;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Sequential, seq, circt::seq::SeqDialect)

void registerSeqPasses() { circt::seq::registerPasses(); }

bool seqTypeIsAClock(MlirType type) { return unwrap(type).isa<ClockType>(); }

MlirType seqClockTypeGet(MlirContext ctx) {
  return wrap(ClockType::get(unwrap(ctx)));
}