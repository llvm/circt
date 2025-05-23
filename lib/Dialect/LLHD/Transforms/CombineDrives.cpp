//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-combine-drives"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_COMBINEDRIVESPASS
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;
using hw::HWModuleOp;

namespace {
struct SignalContext {};
} // namespace

namespace {
struct ModuleContext {
  ModuleContext(HWModuleOp moduleOp) : moduleOp(moduleOp) {}

  void registerDrive(DrvOp op);
  void traceProjection(Value signal);

  /// The module within which we are combining drives.
  HWModuleOp moduleOp;
};
} // namespace

void ModuleContext::registerDrive(DrvOp op) {
  LLVM_DEBUG(llvm::dbgs() << "- Tracing " << op << "\n");
}

void ModuleContext::traceProjection(Value signal) {}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct CombineDrivesPass
    : public llhd::impl::CombineDrivesPassBase<CombineDrivesPass> {
  void runOnOperation() override;
};
} // namespace

void CombineDrivesPass::runOnOperation() {
  ModuleContext context(getOperation());
  for (auto op : context.moduleOp.getOps<DrvOp>())
    context.registerDrive(op);
}
