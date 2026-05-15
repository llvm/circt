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
// output. It supports three masking modes:
//
//   * `delete`: erase the matched ops.
//   * `ifdef` : wrap ops in a single `sv.ifdef`/`sv.ifdef.procedural` whose
//               else region holds the moved ops, plus a single `sv.macro.decl
//               @SYNTHESIS` at the top of the module if absent.
//   * `pragma`: bracket ops with `// synthesis translate_off` /
//               `// synthesis translate_on` `sv.verbatim` ops.
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
// Constants
//===----------------------------------------------------------------------===//

/// Comment text emitted by the `pragma` masking mode.
static constexpr llvm::StringLiteral kPragmaTranslateOff =
    "// synthesis translate_off";
static constexpr llvm::StringLiteral kPragmaTranslateOn =
    "// synthesis translate_on";

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

/// A maximal sequence of consecutive sibling ops within a single block that are
/// all in the masked set.
struct Run {
  Operation *first = nullptr;
  Operation *last = nullptr; // Inclusive.

  /// If this run holds any ops, append a copy onto `runs` and clear ourselves
  /// so we can start accumulating the next one.
  void endRun(SmallVectorImpl<Run> &runs) {
    if (!first)
      return;
    runs.push_back(*this);
    first = nullptr;
    last = nullptr;
  }
};

} // namespace

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

/// Returns true if `op` is an `sv.verbatim` whose format string equals
/// `expected`.
static bool isPragmaVerbatim(Operation *op, StringRef expected) {
  auto verbatim = dyn_cast<sv::VerbatimOp>(op);
  if (!verbatim)
    return false;
  return verbatim.getFormatString() == expected;
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
static bool processBlockDelete(Block &block) {
  SmallVector<Operation *> toErase;
  block.walk([&](Operation *op) {
    if (isMaskedOp(op))
      toErase.push_back(op);
  });
  for (Operation *op : toErase)
    op->erase();
  return !toErase.empty();
}

/// Apply the `ifdef` mode to a run. Picks `sv.ifdef.procedural` vs `sv.ifdef`
/// based on the enclosing region.
static void maskRunByIfdef(Run run, StringRef macro) {
  bool procedural = isInProceduralRegion(run.first);
  OpBuilder builder(run.first);
  Block *parentBlock = run.first->getBlock();
  Location loc = run.first->getLoc();

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

  // Move the run's ops into the else block, preserving order.
  auto firstIt = run.first->getIterator();
  auto endIt = std::next(run.last->getIterator());
  elseBlock->getOperations().splice(
      elseBlock->end(), parentBlock->getOperations(), firstIt, endIt);
}

/// `ifdef` mode: collect runs of consecutive masked sibling ops and wrap
/// each in `sv.ifdef`/`sv.ifdef.procedural`. Recurse into nested regions of
/// non-masked ops, but skip recursion into the else region of a matching
/// `sv.ifdef @<macro>` -- those ops are already guarded.
static bool processBlockIfdef(Block &block, StringRef macro) {
  bool changed = false;
  Run currentRun;
  SmallVector<Run> runs;

  for (Operation &op : block) {
    if (isMaskedOp(&op)) {
      if (currentRun.first)
        currentRun.last = &op;
      else
        currentRun = Run{&op, &op};
      continue;
    }

    currentRun.endRun(runs);
    Region *guardedElseRegion = getMatchingIfDefElseRegion(&op, macro);
    for (Region &region : op.getRegions()) {
      if (&region == guardedElseRegion)
        continue;
      for (Block &nested : region)
        changed |= processBlockIfdef(nested, macro);
    }
  }
  currentRun.endRun(runs);

  for (const Run &run : runs)
    maskRunByIfdef(run, macro);
  return changed || !runs.empty();
}

/// Apply the `pragma` mode to a run.
static void maskRunByPragma(Run run) {
  OpBuilder builder(run.first);
  sv::VerbatimOp::create(builder, run.first->getLoc(),
                         builder.getStringAttr(kPragmaTranslateOff));
  builder.setInsertionPointAfter(run.last);
  sv::VerbatimOp::create(builder, run.last->getLoc(),
                         builder.getStringAttr(kPragmaTranslateOn));
}

/// `pragma` mode: collect runs of consecutive masked sibling ops and bracket
/// each with `// synthesis translate_off` / `_on` `sv.verbatim` ops. Track
/// the bracket state across siblings so ops already sandwiched between
/// matching brackets are skipped. `inheritedGuard` carries the bracket state
/// from the enclosing block so nested blocks wholly contained in an outer
/// pragma bracket are also skipped.
static bool processBlockPragma(Block &block, bool inheritedGuard) {
  bool changed = false;
  bool inGuard = inheritedGuard;
  Run currentRun;
  SmallVector<Run> runs;

  for (Operation &op : block) {
    if (isPragmaVerbatim(&op, kPragmaTranslateOff))
      inGuard = true;
    else if (isPragmaVerbatim(&op, kPragmaTranslateOn))
      inGuard = false;

    if (isMaskedOp(&op)) {
      if (inGuard)
        currentRun.endRun(runs);
      else if (currentRun.first)
        currentRun.last = &op;
      else
        currentRun = Run{&op, &op};
      continue;
    }

    currentRun.endRun(runs);
    for (Region &region : op.getRegions())
      for (Block &nested : region)
        changed |= processBlockPragma(nested, inGuard);
  }
  currentRun.endRun(runs);

  if (runs.empty())
    return changed;
  for (const Run &run : runs)
    maskRunByPragma(run);
  return true;
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

  bool changed = false;
  StringRef passDownMacro =
      macroSymName ? macroSymName.getValue() : StringRef();
  // Each `hw.module`'s body is independent: `processBlock` only mutates ops
  // inside the module it's called on.
  SmallVector<hw::HWModuleOp> hwModules(moduleOp.getOps<hw::HWModuleOp>());
  std::atomic<bool> anyChanged{false};
  mlir::parallelForEach(&getContext(), hwModules, [&](hw::HWModuleOp hwModule) {
    Block &body = *hwModule.getBodyBlock();
    bool localChanged = false;
    switch (mode) {
    case MaskNonSynthesizableMode::Delete:
      localChanged = processBlockDelete(body);
      break;
    case MaskNonSynthesizableMode::Ifdef:
      localChanged = processBlockIfdef(body, passDownMacro);
      break;
    case MaskNonSynthesizableMode::Pragma:
      localChanged = processBlockPragma(body, /*inheritedGuard=*/false);
      break;
    }
    if (localChanged)
      anyChanged.store(true, std::memory_order_relaxed);
  });
  changed = anyChanged.load(std::memory_order_relaxed);

  if (mode == MaskNonSynthesizableMode::Ifdef && changed &&
      macroDeclNeedsCreation) {
    auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());
    sv::MacroDeclOp::create(builder, moduleOp.getLoc(), macroSymName,
                            /*args=*/ArrayAttr{}, builder.getStringAttr(macro));
  }
}
