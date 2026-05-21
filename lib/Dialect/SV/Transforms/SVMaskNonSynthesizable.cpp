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
#include "circt/Support/Namespace.h"
#include "circt/Support/ProceduralRegionTrait.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include <atomic>

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
  StringRef macro;
  Region *elseRegion = nullptr;
  if (auto ifdef = dyn_cast<sv::IfDefOp>(op)) {
    macro = ifdef.getCond().getName();
    elseRegion = &ifdef.getElseRegion();
  } else if (auto ifdef = dyn_cast<sv::IfDefProceduralOp>(op)) {
    macro = ifdef.getCond().getName();
    elseRegion = &ifdef.getElseRegion();
  } else {
    return nullptr;
  }
  return macro == expectedMacro ? elseRegion : nullptr;
}

/// Resolve the symbol name to use for the `sv.macro.decl` referenced by
/// `ifdef` mode's `sv.ifdef` ops. Returns the existing decl's sym_name if a
/// `sv.macro.decl` whose Verilog identifier matches `verilogName` already
/// exists at top level; otherwise returns a fresh sym_name that does not
/// collide with any existing top-level symbol (which may differ from
/// `verilogName` if a non-`sv.macro.decl` symbol of that name is present).
/// `created` is set to true iff the returned name belongs to a decl that
/// the caller still needs to materialize.
static StringAttr resolveMacroSymName(mlir::ModuleOp moduleOp,
                                      StringRef verilogName, bool &created) {
  for (auto decl : moduleOp.getOps<sv::MacroDeclOp>()) {
    if (decl.getMacroIdentifier() == verilogName) {
      created = false;
      return decl.getSymNameAttr();
    }
  }
  Namespace ns;
  ns.add(moduleOp);
  StringRef name = ns.newName(verilogName);
  created = true;
  return StringAttr::get(moduleOp.getContext(), name);
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
  bool procedural = isInProceduralRegion(op);
  OpBuilder builder(op);
  Location loc = op->getLoc();

  // Passing a non-null `elseCtor` is what triggers the builder to create an
  // else block.
  Block *elseBlock;
  if (procedural)
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
    macroSymName = resolveMacroSymName(moduleOp, macro, macroDeclNeedsCreation);

  StringRef passDownMacro =
      macroSymName ? macroSymName.getValue() : StringRef();
  // Each `hw.module`'s body is independent: `processBlock*` only mutates ops
  // inside the module it's called on.
  SmallVector<hw::HWModuleOp> hwModules(moduleOp.getOps<hw::HWModuleOp>());
  std::atomic<bool> anyChanged{false};
  mlir::parallelForEach(&getContext(), hwModules, [&](hw::HWModuleOp hwModule) {
    Block &body = *hwModule.getBodyBlock();
    switch (mode) {
    case MaskNonSynthesizableMode::Delete:
      processBlockDelete(body);
      break;
    case MaskNonSynthesizableMode::Ifdef:
      if (processBlockIfdef(body, passDownMacro))
        anyChanged.store(true, std::memory_order_relaxed);
      break;
    }
  });

  if (mode == MaskNonSynthesizableMode::Ifdef &&
      anyChanged.load(std::memory_order_relaxed) && macroDeclNeedsCreation) {
    auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());
    sv::MacroDeclOp::create(builder, moduleOp.getLoc(), macroSymName,
                            /*args=*/ArrayAttr{}, builder.getStringAttr(macro));
  }
}
