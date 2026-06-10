//===- SVMaskNonSynthesizable.cpp - Mask non-synthesizable SV ops ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass hides a fixed set of `sv` operations that are not generally
// synthesizable (concurrent and property assertions) from the SystemVerilog
// output. It supports two masking modes:
//
//   * `delete`: erase the matched ops.
//   * `ifdef` : wrap each matched op individually in an
//               `sv.ifdef`/`sv.ifdef.procedural` whose else region holds the
//               moved op, plus a single `sv.macro.decl @SYNTHESIS` at the top
//               of the module if absent.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/ProceduralRegionTrait.h"
#include "circt/Support/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include <tuple>

namespace circt {
namespace sv {
#define GEN_PASS_DEF_SVMASKNONSYNTHESIZABLE
#include "circt/Dialect/SV/SVPasses.h.inc"
} // namespace sv
} // namespace circt

using namespace circt;
using namespace circt::sv;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns true if `op` is one of the SV ops we mask.
static bool isMaskedOp(Operation *op) {
  return isa<sv::AssertConcurrentOp, sv::AssumeConcurrentOp,
             sv::CoverConcurrentOp, sv::AssertPropertyOp, sv::AssumePropertyOp,
             sv::CoverPropertyOp>(op);
}

/// If `op` is an `sv.ifdef`/`sv.ifdef.procedural` whose macro matches
/// `expectedMacro`, returns its else region; otherwise returns nullptr. Used
/// to identify the region that need not be re-wrapped in `ifdef` mode.
static Region *getMatchingIfDefElseRegion(Operation *op,
                                          StringRef expectedMacro) {
  return llvm::TypeSwitch<Operation *, Region *>(op)
      .Case<sv::IfDefOp, sv::IfDefProceduralOp>([&](auto ifdef) -> Region * {
        if (ifdef.getCond().getName() != expectedMacro)
          return nullptr;
        return &ifdef.getElseRegion();
      })
      .Default(static_cast<Region *>(nullptr));
}

/// `delete` mode: walk the block and erase every masked op.
static void processBlockDelete(Block &block) {
  block.walk([&](Operation *op) {
    if (isMaskedOp(op))
      op->erase();
  });
}

/// Wrap a single masked op in its own `sv.ifdef`/`sv.ifdef.procedural`. Picks
/// the procedural variant based on the enclosing region.
static void maskOpByIfdef(Operation *op, StringRef macro) {
  OpBuilder builder(op);
  Location loc = op->getLoc();

  // Passing a non-null `elseCtor` is what triggers the builder to create an
  // else block.
  Block *elseBlock;
  if (isInProceduralRegion(op))
    elseBlock =
        sv::IfDefProceduralOp::create(builder, loc, macro,
                                      /*thenCtor=*/{}, /*elseCtor=*/[]() {})
            .getElseBlock();
  else
    elseBlock = sv::IfDefOp::create(builder, loc, macro, /*thenCtor=*/{},
                                    /*elseCtor=*/[]() {})
                    .getElseBlock();

  // Move the op into the else block.
  op->moveBefore(elseBlock, elseBlock->end());
}

/// `ifdef` mode: wrap each masked op in its own
/// `sv.ifdef`/`sv.ifdef.procedural`. Recurse into nested regions of non-masked
/// ops, but skip recursion into the else region of a matching
/// `sv.ifdef @<macro>` -- those ops are already guarded.
static bool processBlockIfdef(Block &block, StringRef macro) {
  bool changed = false;
  // `make_early_inc_range` lets us move the current op out of `block` inside
  // the loop body without invalidating the iterator.
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (isMaskedOp(&op)) {
      maskOpByIfdef(&op, macro);
      changed = true;
      continue;
    }
    Region *guardedElseRegion = getMatchingIfDefElseRegion(&op, macro);
    for (Region &region : op.getRegions()) {
      if (&region == guardedElseRegion)
        continue;
      for (Block &nested : region)
        changed |= processBlockIfdef(nested, macro);
    }
  }
  return changed;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct SVMaskNonSynthesizablePass
    : public circt::sv::impl::SVMaskNonSynthesizableBase<
          SVMaskNonSynthesizablePass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void SVMaskNonSynthesizablePass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();

  // For `ifdef` mode, resolve the symbol name to use for the macro decl
  // up front.
  StringAttr macroSymName;
  bool macroDeclNeedsCreation = false;
  if (mode == MaskNonSynthesizableMode::Ifdef)
    std::tie(macroSymName, macroDeclNeedsCreation) =
        lookupOrGenerateMacroSymName(moduleOp.getBody(), macro);

  StringRef passDownMacro =
      macroSymName ? macroSymName.getValue() : StringRef();
  // Each `hw.module`'s body is independent: `processBlock*` only mutates ops
  // inside the module it's called on.
  SmallVector<hw::HWModuleOp> hwModules(moduleOp.getOps<hw::HWModuleOp>());
  unsigned numChanged =
      transformReduce(&getContext(), hwModules, /*init=*/0u, std::plus<>(),
                      [&](hw::HWModuleOp hwModule) -> unsigned {
                        Block &body = *hwModule.getBodyBlock();
                        switch (mode) {
                        case MaskNonSynthesizableMode::Delete:
                          processBlockDelete(body);
                          return 0;
                        case MaskNonSynthesizableMode::Ifdef:
                          return processBlockIfdef(body, passDownMacro) ? 1 : 0;
                        }
                        llvm_unreachable("all modes handled");
                      });

  if (mode == MaskNonSynthesizableMode::Ifdef && numChanged > 0 &&
      macroDeclNeedsCreation) {
    auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());
    sv::MacroDeclOp::create(builder, moduleOp.getLoc(), macroSymName,
                            /*args=*/ArrayAttr{}, builder.getStringAttr(macro));
  }
}
