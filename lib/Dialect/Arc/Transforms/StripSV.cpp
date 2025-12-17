//===- StripSV.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <variant>

#define DEBUG_TYPE "arc-strip-sv"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_STRIPSV
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;

namespace {
struct StripSVPass : public arc::impl::StripSVBase<StripSVPass> {
  void runOnOperation() override;
  SmallVector<Operation *> opsToDelete;
  SmallPtrSet<StringAttr, 4> clockGateModuleNames;
};
} // namespace

void StripSVPass::runOnOperation() {
  auto mlirModule = getOperation();
  opsToDelete.clear();
  clockGateModuleNames.clear();

  auto expectedClockGateInputs =
      ArrayAttr::get(&getContext(), {StringAttr::get(&getContext(), "in"),
                                     StringAttr::get(&getContext(), "test_en"),
                                     StringAttr::get(&getContext(), "en")});
  auto expectedClockGateOutputs =
      ArrayAttr::get(&getContext(), {StringAttr::get(&getContext(), "out")});
  auto i1Type = IntegerType::get(&getContext(), 1);

  for (auto extModOp : mlirModule.getOps<hw::HWModuleExternOp>()) {
    if (extModOp.getVerilogModuleName() == "EICG_wrapper") {
      if (!llvm::equal(extModOp.getInputNames(), expectedClockGateInputs) ||
          !llvm::equal(extModOp.getOutputNames(), expectedClockGateOutputs)) {
        extModOp.emitError("clock gate module `")
            << extModOp.getModuleName() << "` has incompatible port names "
            << extModOp.getInputNames() << " -> " << extModOp.getOutputNames();
        return signalPassFailure();
      }
      if (!llvm::equal(extModOp.getInputTypes(),
                       ArrayRef<Type>{i1Type, i1Type, i1Type}) ||
          !llvm::equal(extModOp.getOutputTypes(), ArrayRef<Type>{i1Type})) {
        extModOp.emitError("clock gate module `")
            << extModOp.getModuleName() << "` has incompatible port types "
            << extModOp.getInputTypes() << " -> " << extModOp.getOutputTypes();
        return signalPassFailure();
      }
      clockGateModuleNames.insert(extModOp.getModuleNameAttr());
      opsToDelete.push_back(extModOp);
      continue;
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "Found " << clockGateModuleNames.size()
                          << " clock gates\n");

  // Remove `sv.*` operation attributes.
  mlirModule.walk([](Operation *op) {
    auto isSVAttr = [](NamedAttribute attr) {
      return attr.getName().getValue().starts_with("sv.");
    };
    if (llvm::any_of(op->getAttrs(), isSVAttr)) {
      SmallVector<NamedAttribute> newAttrs;
      newAttrs.reserve(op->getAttrs().size());
      for (auto attr : op->getAttrs())
        if (!isSVAttr(attr))
          newAttrs.push_back(attr);
      op->setAttrs(newAttrs);
    }
  });

  // Remove ifdefs and verbatim.
  for (auto verb : mlirModule.getOps<sv::VerbatimOp>())
    opsToDelete.push_back(verb);
  for (auto verb : mlirModule.getOps<sv::IfDefOp>())
    opsToDelete.push_back(verb);
  for (auto verb : mlirModule.getOps<sv::MacroDeclOp>())
    opsToDelete.push_back(verb);

  for (auto module : mlirModule.getOps<hw::HWModuleOp>()) {
    for (Operation &op : *module.getBodyBlock()) {
      // Remove ifdefs and verbatim.
      if (isa<sv::IfDefOp, sv::CoverOp, sv::CoverConcurrentOp>(&op)) {
        opsToDelete.push_back(&op);
        continue;
      }
      if (isa<sv::VerbatimOp, sv::AlwaysOp>(&op)) {
        opsToDelete.push_back(&op);
        continue;
      }

      // Remove wires.
      if (auto assign = dyn_cast<sv::AssignOp>(&op)) {
        auto wire = assign.getDest().getDefiningOp<sv::WireOp>();
        if (!wire) {
          assign.emitOpError("expected wire lhs");
          return signalPassFailure();
        }
        for (Operation *user : wire->getUsers()) {
          if (user == assign)
            continue;
          auto readInout = dyn_cast<sv::ReadInOutOp>(user);
          if (!readInout) {
            user->emitOpError("has user that is not `sv.read_inout`");
            return signalPassFailure();
          }
          readInout.replaceAllUsesWith(assign.getSrc());
          opsToDelete.push_back(readInout);
        }
        opsToDelete.push_back(assign);
        opsToDelete.push_back(wire);
        continue;
      }

      // Canonicalize registers.
      if (auto reg = dyn_cast<seq::FirRegOp>(&op)) {
        OpBuilder builder(reg);
        Value next = reg.getNext();

        if (reg.getIsAsync()) {
          reg.emitOpError("only synchronous resets are currently supported");
          return signalPassFailure();
        }

        Value reset = reg.hasReset() ? reg.getReset() : Value{};
        Value resetValue = reg.hasReset() ? reg.getResetValue() : Value{};

        Value presetValue;
        // Materialize initial value, assume zero initialization as default.
        if (reg.getPreset() && !reg.getPreset()->isZero()) {
          assert(hw::type_isa<IntegerType>(reg.getType()) &&
                 "cannot lower non integer preset");
          presetValue = circt::seq::createConstantInitialValue(
              builder, reg.getLoc(),
              IntegerAttr::get(reg.getType(), *reg.getPreset()));
        }

        Value compReg = seq::CompRegOp::create(
            builder, reg.getLoc(), next.getType(), next, reg.getClk(),
            reg.getNameAttr(), reset, resetValue, /*initialValue*/ presetValue,
            reg.getInnerSymAttr());
        reg.replaceAllUsesWith(compReg);
        opsToDelete.push_back(reg);
        continue;
      }

      // Replace clock gate instances with the dedicated `seq.clock_gate` op and
      // stub out other external modules.
      if (auto instOp = dyn_cast<hw::InstanceOp>(&op)) {
        auto modName = instOp.getModuleNameAttr().getAttr();
        ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
        if (clockGateModuleNames.contains(modName)) {
          auto gated = seq::ClockGateOp::create(
              builder, instOp.getOperand(0), instOp.getOperand(1),
              instOp.getOperand(2), hw::InnerSymAttr{});
          instOp.replaceAllUsesWith(gated);
          opsToDelete.push_back(instOp);
        }
        continue;
      }
    }
  }
  for (auto *op : opsToDelete)
    op->erase();
}

std::unique_ptr<Pass> arc::createStripSVPass() {
  return std::make_unique<StripSVPass>();
}
