//===- Dedup.cpp - Moore module deduping --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file  implements moore module deduplication.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_DEDUP
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;

//===----------------------------------------------------------------------===//
// DedupPass
//===----------------------------------------------------------------------===//
namespace {
struct DedupPass : public circt::moore::impl::DedupBase<DedupPass> {
  void runOnOperation() override;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createDedupPass() {
  return std::make_unique<DedupPass>();
}

void DedupPass::runOnOperation() {}
