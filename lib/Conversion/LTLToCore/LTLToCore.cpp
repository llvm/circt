//===- LTLToCore.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts LTL and Verif operations to Core operations
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LTLToCore.h"
#include "../PassDetail.h"
#include "circt/Conversion/HWToSV.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace circt;
using namespace hw;

static sv::EventControl ltlToSVEventControl(ltl::ClockEdge ce) {
  switch (ce) {
  case ltl::ClockEdge::Pos:
    return sv::EventControl::AtPosEdge;
  case ltl::ClockEdge::Neg:
    return sv::EventControl::AtNegEdge;
  case ltl::ClockEdge::Both:
    return sv::EventControl::AtEdge;
  }
  llvm_unreachable("Unknown event control kind");
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

struct HasBeenResetOpConversion : OpConversionPattern<verif::HasBeenResetOp> {
  using OpConversionPattern<verif::HasBeenResetOp>::OpConversionPattern;

  // HasBeenReset generates a 1 bit register that is set to one once the reset
  // has been raised and lowered at at least once.
  LogicalResult
  matchAndRewrite(verif::HasBeenResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i1 = rewriter.getI1Type();
    // Generate the constant used to set the register value
    Value constZero = rewriter.create<hw::ConstantOp>(op.getLoc(), i1, 0);

    // Generate the constant used to enegate the
    Value constOne = rewriter.create<hw::ConstantOp>(op.getLoc(), i1, 1);

    // Create a backedge for the register to be used in the OrOp
    circt::BackedgeBuilder bb(rewriter, op.getLoc());
    circt::Backedge reg = bb.get(rewriter.getI1Type());

    // Generate an or between the reset and the register's value to store
    // whether or not the reset has been active at least once
    Value orReset =
        rewriter.create<comb::OrOp>(op.getLoc(), adaptor.getReset(), reg);

    // This register should not be reset, so we give it dummy reset and resetval
    // operands to fit the build signature
    Value reset, resetval;

    // Finally generate the register to set the backedge
    reg.setValue(rewriter.create<seq::CompRegOp>(
        op.getLoc(), orReset,
        rewriter.createOrFold<seq::ToClockOp>(op.getLoc(), adaptor.getClock()),
        reset, resetval, llvm::StringRef("hbr"), constZero));

    // We also need to consider the case where we are currently in a reset cycle
    // in which case our hbr register should be down-
    // Practically this means converting it to (and hbr (not reset))
    Value notReset =
        rewriter.create<comb::XorOp>(op.getLoc(), adaptor.getReset(), constOne);
    rewriter.replaceOpWithNewOp<comb::AndOp>(op, reg, notReset);

    return success();
  }
};

struct AssertOpConversionPattern : OpConversionPattern<verif::AssertOp> {
  using OpConversionPattern<verif::AssertOp>::OpConversionPattern;

  Value visit(ltl::DisableOp op, ConversionPatternRewriter &rewriter,
              Value operand = nullptr) const {
    // Replace the ltl::DisableOp with an OR op as it represents a disabling
    // implication: (implies (not condition) input) is equivalent to
    // (or (not (not condition)) input) which becomes (or condition input)
    return rewriter.replaceOpWithNewOp<comb::OrOp>(
        op, op.getCondition(), operand ? operand : op.getInput());
  }

  // Creates and returns a logical implication:
  // a -> b which is encoded as !a || b
  // The Value for the OrOp will be returned
  Value makeImplication(Location loc, Value antecedent, Value consequent,
                        ConversionPatternRewriter &rewriter) const {
    Value constOne =
        rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(), 1);
    Value nA = rewriter.create<comb::XorOp>(loc, antecedent, constOne);
    return rewriter.create<comb::OrOp>(loc, nA, consequent);
  }

  // NOI case: Generate a register delaying our antecedent
  // for each cycle in delayN, as well as a register to count the
  // delay, e.g. a ##n true |-> b would have the following assertion:
  // assert(delay < n || (!a_n || b) || reset)

  bool makeNonOverlappingImplication(ltl::ImplicationOp implop,
                                     ltl::DelayOp delayN, ltl::ClockOp ltlclock,
                                     ltl::ConcatOp concat,
                                     ConversionPatternRewriter &rewriter,
                                     Value &res) const {

    // Start by recovering the number of registers we need to generate
    uint64_t delayCycles = delayN.getDelay();

    // The width of our delay register can simply be log2(delayCycles) +
    // 1 as we can saturate it once it's reached delayCycles
    uint64_t delayRegW = llvm::Log2_64(delayCycles) + 1;
    auto idrw = IntegerType::get(getContext(), delayRegW);

    // Build out the delay register: delay' = delay + 1; reset(delay, 0)
    // Generate the constant used to enegate the
    Value constZero = rewriter.create<hw::ConstantOp>(concat.getLoc(), idrw, 0);

    // Create a constant to be used in the delay next statement
    Value constOneW = rewriter.create<hw::ConstantOp>(delayN.getLoc(), idrw, 1);

    // Create a backedge for the delay register
    circt::BackedgeBuilder bb(rewriter, delayN.getLoc());
    circt::Backedge delayReg = bb.get(idrw);

    // Increment the delay register by 1
    Value delayInc =
        rewriter.create<comb::AddOp>(delayN.getLoc(), delayReg, constOneW);

    // Saturate the register if it reaches the delay, i.e. delay' = (delay
    // == delayCycles) ? delayCycles : delay + 1
    Value delayMax =
        rewriter.create<hw::ConstantOp>(delayN.getLoc(), idrw, delayCycles);
    Value delayEqMax = rewriter.create<comb::ICmpOp>(
        delayN.getLoc(),
        comb::ICmpPredicateAttr::get(getContext(), comb::ICmpPredicate::eq),
        delayReg, delayMax, mlir::UnitAttr::get(getContext()));
    Value delayMux = rewriter.create<comb::MuxOp>(delayN.getLoc(), delayEqMax,
                                                  delayMax, delayInc);

    // Retrieve Parent module and look for a reset
    hw::HWModuleOp parent = dyn_cast<hw::HWModuleOp>(ltlclock->getParentOp());
    Value reset;

    // Find reset
    if (parent)
      for (PortInfo &pi : parent.getPortList())
        if (pi.getName() == "reset")
          reset = parent.getArgumentForInput(pi.argNum);

    // Sanity check: Enforce the existence of a reset
    if (!reset) {
      delayN->emitError("Parent Module must have a reset argument!");
      return false;
    }

    // Extract the actual clock
    auto clock = rewriter.createOrFold<seq::ToClockOp>(delayN.getLoc(),
                                                       ltlclock.getClock());

    // Create the actual register
    delayReg.setValue(rewriter.create<seq::CompRegOp>(
        delayN.getLoc(), delayMux, clock, reset, constZero,
        llvm::StringRef("delay_"), constZero));

    // Previous register in the pipeline
    Value aI;
    Value a = concat.getInputs().front();

    // Generate reset values
    auto aType = a.getType();
    auto itype = isa<IntegerType>(aType)
                     ? aType
                     : IntegerType::get(getContext(), hw::getBitWidth(aType));
    Value resetVal = rewriter.create<hw::ConstantOp>(delayN.getLoc(), itype, 0);

    aI = rewriter.create<seq::CompRegOp>(
        delayN.getLoc(), a, clock, reset, resetVal,
        llvm::StringRef("antecedent_0"), resetVal);

    // Create a pipeline of delay registers
    for (size_t i = 1; i < delayCycles; ++i)
      aI = rewriter.create<seq::CompRegOp>(
          delayN.getLoc(), aI, clock, reset, resetVal,
          llvm::StringRef("antecedent_" + std::to_string(i)), resetVal);

    // Generate the final assertion: assert(delayReg < delayMax ||
    // (aI -> consequent) || reset)
    Value condMin = rewriter.create<comb::ICmpOp>(
        delayN.getLoc(),
        comb::ICmpPredicateAttr::get(getContext(), comb::ICmpPredicate::ult),
        delayReg, delayMax, mlir::UnitAttr::get(getContext()));
    Value constOneAi =
        rewriter.create<hw::ConstantOp>(delayN.getLoc(), aI.getType(), 1);
    Value notAi = rewriter.create<comb::XorOp>(delayN.getLoc(), aI, constOneAi);
    Value implAiConsequent = rewriter.create<comb::OrOp>(
        delayN.getLoc(), notAi, implop.getConsequent());
    Value condLhs =
        rewriter.create<comb::OrOp>(delayN.getLoc(), condMin, implAiConsequent);

    // Finally create the final assertion condition
    res = rewriter.create<comb::OrOp>(delayN.getLoc(), condLhs, reset);
    return true;
  }

  // Special case : we want to detect the Non-overlapping implication
  // pattern and reject everything else for now:
  // antecedent : ltl::concatOp || immediate predicate
  // consequent : any other non-sequence op
  // We want to support a ##n true |-> b and a |-> b
  bool visit(ltl::ImplicationOp implop, ConversionPatternRewriter &rewriter,
             ltl::ClockOp ltlclock, Value &res) const {
    // Sanity check: Make sure that a clock was found for the assertion that
    // uses this implication
    if (!ltlclock) {
      implop->emitError("No clock was found associated to this sequence!");
      return false;
    }

    // Figure out what pattern we are in
    // a: Non-Overlapping Implication (NOI) or b: Overlapping Implication (OI)
    Operation *antecedent = implop.getAntecedent().getDefiningOp();

    // Check that the consequent is legal (non-property type)
    // Conseuqent may also potentially be an input but not the antcedent
    bool isConsequentLegal = true;
    if (!isa<BlockArgument>(implop.getConsequent()))
      isConsequentLegal =
          llvm::TypeSwitch<Operation *, bool>(
              implop.getConsequent().getDefiningOp())
              .Case<ltl::AndOp, ltl::ClockOp, ltl::ConcatOp, ltl::DelayOp,
                    ltl::DisableOp, ltl::EventuallyOp, ltl::ImplicationOp,
                    ltl::NotOp, ltl::OrOp>([&](auto op) {
                op->emitError(
                    "Invalid consequent type, must be an immediate predicate!");
                return false;
              })
              .Default([&](auto) { return true; });

    bool isAntecedentLegal = true;
    if (!isa<BlockArgument>(implop.getAntecedent()))
      isAntecedentLegal =
          llvm::TypeSwitch<Operation *, bool>(antecedent)
              .Case<ltl::AndOp, ltl::ClockOp, ltl::DelayOp, ltl::DisableOp,
                    ltl::EventuallyOp, ltl::ImplicationOp, ltl::NotOp,
                    ltl::OrOp>([&](auto op) {
                op->emitError("Invalid antecedent type, must be an immediate "
                              "predicate or a concat op!");
                return false;
              })
              .Default([&](auto) { return true; });

    if (bool isLegal = !(isConsequentLegal && isAntecedentLegal))
      return isLegal;

    // Check for NOI case: If antecedent is an input, then we know that no delay
    // is associated to the implication and thus it's an overlapping implication
    if (!isa<BlockArgument>(implop.getAntecedent())) {
      auto concat = dyn_cast<ltl::ConcatOp>(antecedent);
      if (concat) {
        // We are only supporting sequences of the type a ##n true
        auto inputs = concat.getInputs();
        auto nInputs = inputs.size();
        if (nInputs > 2) {
          concat->emitError("Antecedent must be of the form a ##n true");
          return false;
        }

        // Figure out if we are in the NOI case of "a ##n true"
        if (nInputs == 2) {
          if (!isa<BlockArgument>(inputs.front())) {
            if (dyn_cast<ltl::DelayOp>(inputs.front().getDefiningOp())) {
              concat->emitError("Antecedent must be of the form a ##n true");
              return false;
            }
          }
          // Figure out what our NOI delay is
          // Make sure that we aren't trying to case a block argument
          ltl::DelayOp delayN;
          if (!isa<BlockArgument>(inputs.back())) {
            if ((delayN =
                     dyn_cast<ltl::DelayOp>(inputs.back().getDefiningOp()))) {

              // Make sure that you only allow for a ##n true |-> b
              hw::ConstantOp hwconst;
              if (!(hwconst = dyn_cast<hw::ConstantOp>(
                        delayN.getInput().getDefiningOp()))) {
                delayN->emitError("Only a ##n true |-> b is supported. RHS of "
                                  "the concatenation must be true");
                return false;
              } else {
                if (hwconst.getValue().isZero()) {
                  delayN->emitError(
                      "Only a ##n true |-> b is supported. RHS of "
                      "the concatenation must be true");
                  return false;
                }

                // NOI case: generate the hardware needed to encode the
                // non-overlapping implication
                if (!makeNonOverlappingImplication(implop, delayN, ltlclock,
                                                   concat, rewriter, res))
                  return false;

                rewriter.eraseOp(delayN);
              }
            }
          } else {
            concat->emitError("Antecedent must be of the form a ##n true");
            return false;
          }
        } else {
          // OI case: simply generate an implication so
          // (or (not antecedant) consequent)
          res = makeImplication(implop.getLoc(), inputs.front(),
                                implop.getConsequent(), rewriter);
        }
        rewriter.eraseOp(concat);
      }
    } else {
      // OI case: simply generate an implication so
      // (or (not antecedant) consequent)
      res = makeImplication(implop.getLoc(), implop.getAntecedent(),
                            implop.getConsequent(), rewriter);
    }
    rewriter.eraseOp(implop);
    return true;
  }

  LogicalResult
  matchAndRewrite(verif::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Operation *curOp, *nextOp;
    ltl::ClockOp ltlclock;
    Value disableOp, convimplop;
    llvm::SmallVector<Operation *> defferlist;

    // Start with the assertion's operand
    curOp = adaptor.getProperty().getDefiningOp();

    // Walk backwards from the assertion and generate all of the operands first
    while (curOp != nullptr) {
      // TODO make this into a DFS aproach with a worklist (again)
      bool isLegal =
          llvm::TypeSwitch<Operation *, bool>(curOp)
              .Case<ltl::ImplicationOp>([&](auto implop) {
                if (!visit(implop, rewriter, ltlclock, convimplop)) {
                  implop->emitError(
                      "Invalid implication format, only a ##n true |-> b or  a "
                      "|-> b are supported!");
                  return false;
                }
                nextOp = nullptr;
                return true;
              })
              .Case<ltl::DisableOp>([&](auto disable) {
                defferlist.push_back(disable);
                nextOp = disable.getInput().getDefiningOp();
                return true;
              })
              .Case<ltl::ClockOp>([&](auto clockop) {
                // Simply register the clock that we just found and move onto
                // the clocked operation
                ltlclock = clockop;
                nextOp = clockop.getInput().getDefiningOp();
                return true;
              })
              .Case<mlir::UnrealizedConversionCastOp>([&](auto cast) {
                // Simply forward the operand to be converted
                nextOp = cast->getOperand(0).getDefiningOp();
                return true;
              })
              .Default([&](auto e) {
                nextOp = nullptr;
                return true;
              });
      if (!isLegal)
        return rewriter.notifyMatchFailure(op,
                                           " Current operation is invalid!");

      // Go to the next operation
      curOp = nextOp;
    }

    // Convert the operations that have been deferred (only disables for now)
    while (!defferlist.empty()) {
      Operation *defferop = defferlist[defferlist.size() - 1];
      defferlist.pop_back();
      auto disable = dyn_cast<ltl::DisableOp>(defferop);
      if (disable) {
        // Check if the operand needed conversion (mostly for ltl::implications)
        if (dyn_cast<ltl::ImplicationOp>(disable.getInput().getDefiningOp()))
          disableOp = visit(disable, rewriter, convimplop);
        else
          disableOp = visit(disable, rewriter);
      }
    }

    // Sanity check, we should have found a clock
    if (!ltlclock)
      return rewriter.notifyMatchFailure(
          op, "verif.assert property is not associated to a clock! ");

    if (!disableOp)
      return rewriter.notifyMatchFailure(
          op, "verif.assert property is not properly disabled! ");

    // Generate the parenting sv.always posedge clock from the ltl
    // clock, containing the generated sv.assert
    rewriter.create<sv::AlwaysOp>(
        ltlclock.getLoc(), ltlToSVEventControl(ltlclock.getEdge()),
        ltlclock.getClock(), [&] {
          // Generate the sv assertion using the input to the
          // parenting clock
          rewriter.replaceOpWithNewOp<sv::AssertOp>(
              op, disableOp,
              sv::DeferAssertAttr::get(getContext(),
                                       sv::DeferAssert::Immediate),
              op.getLabelAttr());
        });

    rewriter.eraseOp(ltlclock);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Lower LTL To Core pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerLTLToCorePass : public LowerLTLToCoreBase<LowerLTLToCorePass> {
  LowerLTLToCorePass() = default;
  void runOnOperation() override;
};
} // namespace

// Simply applies the conversion patterns defined above
void LowerLTLToCorePass::runOnOperation() {

  // Set target dialects: We don't want to see any ltl or verif that might come
  // from an AssertProperty left in the result
  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<sv::SVDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  target.addLegalDialect<ltl::LTLDialect>();
  target.addIllegalDialect<verif::VerifDialect>();

  // Create type converters, mostly just to convert an ltl property to a bool
  mlir::TypeConverter converter;

  // Convert the ltl property type to a built-in type
  converter.addConversion([](IntegerType type) { return type; });
  converter.addConversion([](ltl::PropertyType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  converter.addConversion([](ltl::SequenceType type) {
    return IntegerType::get(type.getContext(), 1);
  });

  // Basic materializations
  converter.addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });

  converter.addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });

  // Create the operation rewrite patters
  RewritePatternSet patterns(&getContext());
  patterns.add<AssertOpConversionPattern, HasBeenResetOpConversion>(
      converter, patterns.getContext());

  // Apply the conversions
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createLowerLTLToCorePass() {
  return std::make_unique<LowerLTLToCorePass>();
}
