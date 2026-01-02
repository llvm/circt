//===- InferContracts.cpp - Formal Preparations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Infers contracts from existing assertions and assumptions.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"

using namespace circt;

namespace circt {
namespace verif {
#define GEN_PASS_DEF_INFERCONTRACTSPASS
#include "circt/Dialect/Verif/Passes.h.inc"
} // namespace verif
} // namespace circt

using namespace mlir;

namespace {
struct InferContractsPass
    : verif::impl::InferContractsPassBase<InferContractsPass> {

  /// For all modules (hw::HWModuleOp and verif::FormalOp)
  /// that don't already define a contract:
  /// 1. Accumulate all of the conditions from the assertions found in the
  /// region into a single conjuction, this will become the postcondition of the
  /// contract-
  /// 2. Accumulate all of the conditions from the assumptions found in the
  /// region that condition over the inputs of the module (block arguments for
  /// hw::HWModuleOp, verif.symbolic_value for the verif::FromalOp), this
  /// conjunction will become the preconditions of the contract.
  void runOnOperation() override;
};

/// Given an assert or an assume operation, convert them into a contract
static Operation *convertAssertLike(OpBuilder &builder, RequireLike op,
                                    IRMapping &mapping) {
  StringAttr labelAttr = op.getLabelAttr();

  Value enableValue;
  if (auto enable = op.getEnable())
    enable = mapping.lookup(enable);

  auto loc = op.getLoc();
  auto property = mapping.lookup(op.getProperty());

  if ((isa<EnsureOp>(op) && !assumeContract) ||
      (isa<RequireOp>(op) && assumeContract))
    return AssertOp::create(builder, loc, property, enableValue, labelAttr);
  if ((isa<EnsureOp>(op) && assumeContract) ||
      (isa<RequireOp>(op) && !assumeContract))
    return AssumeOp::create(builder, loc, property, enableValue, labelAttr);
  return nullptr;
}

template <typename T>
/// Generic runner, only supports `hw::HWModuleOp` and `verif::FormalOp`.
LogicalResult run(T op, ModuleOp mlirModuleOp) {
  // Sanity check: Only support HwModules and Formal Ops
  if (!(isa<hw::HWModuleOp>(op) || isa<verif::FormalOp>(op)))
    return failure();

  return success();
}

} // namespace

void InferContractsPass::runOnOperation() {
  auto mlirModule = getOperation();

  // Run the pass on all `hw.module` and `verif.formal` ops
  for (auto module : mlirModule.getOps<hw::HWModuleOp>())
    if (failed(run<hw::HWModuleOp>(module, mlirModule)))
      signalPassFailure();

  for (auto formal : mlirModule.getOps<verif::FormalOp>())
    if (failed(run<verif::FormalOp>(formal, mlirModule)))
      signalPassFailure();
}
