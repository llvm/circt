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

#include "circt/Dialect/HW/HWOps.h"
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

  /// It is assumed that `combine-assert-like` was run before this pass.
  /// For all modules (hw::HWModuleOp and verif::FormalOp)
  /// that don't already define a contract:
  /// 1. Find the accumulated assertion for that block and replace it
  /// 2. Find the accumulated assumptions for this region that condition over
  /// the inputs of the module (block arguments for hw::HWModuleOp,
  /// verif.symbolic_value for the verif::FromalOp), this conjunction will
  /// become the preconditions of the contract.
  void runOnOperation() override;
};

/// Given an assert or an assume operation, convert them into a contract
template <typename T>
static Operation *convertAssertLike(OpBuilder &builder, T op,
                                    IRMapping &mapping) {
  StringAttr labelAttr = op.getLabelAttr();

  Value enableValue;
  if (auto enable = op.getEnable())
    enable = mapping.lookup(enable);

  auto loc = op.getLoc();
  auto property = mapping.lookup(op.getProperty());

  return nullptr;
}

template <typename T>
/// Generic runner, only supports `hw::HWModuleOp` and `verif::FormalOp`.
LogicalResult run(T modop, ModuleOp mlirModuleOp) {
  // Sanity check: Only support HwModules and Formal Ops
  if (!(isa<hw::HWModuleOp>(modop) || isa<verif::FormalOp>(modop)))
    return failure();

  // Check if a contract is already defined; if so fail
  modop.walk([&](verif::ContractOp cnt) {
    return failure("Inference is only supported when no contract is present.");
  });

  // Create mapping and builder
  IRMapping mapping;
  OpBuilder builder(modop);

  // Convert all assertlike ops into contracts
  modop.walk([&](Operation *op) {
    TypeSwitch<Operations *, void>(op)
        .Case<verif::AssertOp, verif::AssumeOp>(
            [&](auto assop) { convertAssertLike(builder, assop, mapping); })
        .Default([&](auto) {});
  });

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
