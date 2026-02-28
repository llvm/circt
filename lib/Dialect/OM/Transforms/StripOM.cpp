//===- StripOM.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

namespace circt {
namespace om {
#define GEN_PASS_DEF_STRIPOMPASS
#include "circt/Dialect/OM/OMPasses.h.inc"
} // namespace om
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {
struct StripOMPass : public om::impl::StripOMPassBase<StripOMPass> {
  void runOnOperation() override {
    for (auto &op : llvm::make_early_inc_range(getOperation().getOps()))
      if (isa_and_nonnull<om::OMDialect>(op.getDialect()))
        op.erase();
  }
};
} // namespace
