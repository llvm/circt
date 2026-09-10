//===- SFCCompat.cpp - SFC Compatible Pass ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass makes a number of updates to the circuit that are required to match
// the behavior of the Scala FIRRTL Compiler (SFC).  This pass removes invalid
// values from the circuit.  This is a combination of the Scala FIRRTL
// Compiler's RemoveRests pass and RemoveValidIf.  This is done to remove two
// "interpretations" of invalid.  Namely: (1) registers that are initialized to
// an invalid value (module scoped and looking through wires and connects only)
// are converted to an unitialized register and (2) invalid values are converted
// to zero (after rule 1 is applied).  Additionally, this pass checks and
// disallows async reset registers that are not driven with a constant when
// looking through wires, connects, and nodes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-remove-resets"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_SFCCOMPAT
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

struct SFCCompatPass
    : public circt::firrtl::impl::SFCCompatBase<SFCCompatPass> {
  void runOnOperation() override;
};

void SFCCompatPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "==----- Running SFCCompat "
                      "---------------------------------------------------===\n"
                   << "Module: '" << getOperation().getName() << "'\n";);

  bool madeModifications = false;
  SmallVector<InvalidValueOp> invalidOps;

  auto fullResetAttr = StringAttr::get(&getContext(), fullResetAnnoClass);
  auto isFullResetAnno = [fullResetAttr](Annotation anno) {
    auto annoClassAttr = anno.getClassAttr();
    return annoClassAttr == fullResetAttr;
  };
  bool fullResetExists = AnnotationSet::removePortAnnotations(
      getOperation(),
      [&](unsigned argNum, Annotation anno) { return isFullResetAnno(anno); });
  getOperation()->walk([isFullResetAnno, &fullResetExists](Operation *op) {
    fullResetExists |= AnnotationSet::removeAnnotations(op, isFullResetAnno);
  });
  madeModifications |= fullResetExists;

  auto result = getOperation()->walk([&](Operation *op) {
    // Populate invalidOps for later handling.
    if (auto inv = dyn_cast<InvalidValueOp>(op)) {
      invalidOps.push_back(inv);
      return WalkResult::advance();
    }
    auto reg = dyn_cast<RegResetOp>(op);
    if (!reg)
      return WalkResult::advance();

    // If the `RegResetOp` has an invalidated initialization and we
    // are not running FART, then replace it with a `RegOp`.
    if (!fullResetExists && walkDrivers(reg.getResetValue(), true, true, false,
                                        [](FieldRef dst, FieldRef src) {
                                          return src.isa<InvalidValueOp>();
                                        })) {
      ImplicitLocOpBuilder builder(reg.getLoc(), reg);
      RegOp newReg = RegOp::create(
          builder, reg.getResult().getType(), reg.getClockVal(),
          reg.getNameAttr(), reg.getNameKindAttr(), reg.getAnnotationsAttr(),
          reg.getInnerSymAttr(), reg.getForceableAttr());
      reg.replaceAllUsesWith(newReg);
      reg.erase();
      madeModifications = true;
      return WalkResult::advance();
    }

    // If the `RegResetOp` has an asynchronous reset and the reset value is not
    // a module-scoped constant when looking through wires and nodes, then
    // generate an error.  This implements the SFC's CheckResets pass.
    if (!isa<AsyncResetType>(reg.getResetSignal().getType()))
      return WalkResult::advance();
    if (walkDrivers(
            reg.getResetValue(), true, true, true,
            [&](FieldRef dst, FieldRef src) {
              if (src.isa<ConstantOp, InvalidValueOp, SpecialConstantOp,
                          AggregateConstantOp>())
                return true;
              auto diag = emitError(reg.getLoc());
              auto [fieldName, rootKnown] = getFieldName(dst);
              diag << "register " << reg.getNameAttr()
                   << " has an async reset, but its reset value";
              if (rootKnown)
                diag << " \"" << fieldName << "\"";
              diag << " is not driven with a constant value through wires, "
                      "nodes, or connects";
              std::tie(fieldName, rootKnown) = getFieldName(src);
              diag.attachNote(src.getLoc())
                  << "reset driver is "
                  << (rootKnown ? ("\"" + fieldName + "\"") : "here");
              return false;
            }))
      return WalkResult::advance();
    return WalkResult::interrupt();
  });

  if (result.wasInterrupted())
    return signalPassFailure();

  // Convert all invalid values to zero.
  for (auto inv : invalidOps) {
    // Delete invalids which have no uses.
    if (inv->getUses().empty()) {
      inv->erase();
      madeModifications = true;
      continue;
    }
    ImplicitLocOpBuilder builder(inv.getLoc(), inv);
    Value replacement =
        FIRRTLTypeSwitch<FIRRTLType, Value>(inv.getType())
            .Case<ClockType, AsyncResetType, ResetType>(
                [&](auto type) -> Value {
                  return SpecialConstantOp::create(builder, type,
                                                   builder.getBoolAttr(false));
                })
            .Case<IntType>([&](IntType type) -> Value {
              return ConstantOp::create(builder, type, getIntZerosAttr(type));
            })
            .Case<FEnumType, BundleType, FVectorType>([&](auto type) -> Value {
              auto width = circt::firrtl::getBitWidth(type);
              assert(width && "width must be inferred");
              auto zero = ConstantOp::create(builder, APSInt(*width));
              return BitCastOp::create(builder, type, zero);
            })
            .Default([&](auto) {
              llvm_unreachable("all types are supported");
              return Value();
            });
    inv.replaceAllUsesWith(replacement);
    inv.erase();
    madeModifications = true;
  }

  if (!madeModifications)
    return markAllAnalysesPreserved();
}
