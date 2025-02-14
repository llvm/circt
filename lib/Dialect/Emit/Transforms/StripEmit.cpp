//===- StripEmit.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/Emit/EmitPasses.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

namespace circt {
namespace emit {
#define GEN_PASS_DEF_STRIPEMITPASS
#include "circt/Dialect/Emit/EmitPasses.h.inc"
} // namespace emit
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {
struct StripEmitPass : public emit::impl::StripEmitPassBase<StripEmitPass> {
  void runOnOperation() override {
    for (auto &op : llvm::make_early_inc_range(getOperation().getOps())) {
      if (isa_and_nonnull<emit::EmitDialect>(op.getDialect())) {
        op.erase();
        continue;
      }
      op.removeAttr("emit.fragments");
    }
  }
};
} // namespace
