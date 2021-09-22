//===- SCFToCalyx.cpp - SCF to Calyx pass entry point -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main SCF to Calyx conversion pass implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SCFToCalyx/SCFToCalyx.h"
#include "../PassDetail.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include <variant>

using namespace llvm;
using namespace mlir;

namespace circt {

//===----------------------------------------------------------------------===//
// Utility types
//===----------------------------------------------------------------------===//

/// A mapping is maintained between a function operation and its corresponding
/// Calyx component.
using FuncMapping = DenseMap<FuncOp, calyx::ComponentOp>;

using Scheduleable = std::variant<calyx::GroupOp>;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Tries to match a constant value defined by op. If the match was
/// successful, returns true and binds the constant to 'value'. If unsuccessful,
/// the value is unmodified.
static bool matchConstantOp(Operation *op, APInt &value) {
  return mlir::detail::constant_int_op_binder(&value).match(op);
}

/// Creates a DictionaryAttr containing a unit attribute 'name'. Used for
/// defining mandatory port attributes for calyx::ComponentOp's.
static DictionaryAttr getMandatoryPortAttr(MLIRContext *ctx, StringRef name) {
  return DictionaryAttr::get(
      ctx, {NamedAttribute(Identifier::get(name, ctx), UnitAttr::get(ctx))});
}

/// Adds the mandatory Calyx component I/O ports (->[clk, reset, go], [done]->)
/// to ports.
static void
addMandatoryComponentPorts(PatternRewriter &rewriter,
                           SmallVectorImpl<calyx::PortInfo> &ports) {
  MLIRContext *ctx = rewriter.getContext();
  ports.push_back({rewriter.getStringAttr("clk"), rewriter.getI1Type(),
                   calyx::Direction::Input, getMandatoryPortAttr(ctx, "clk")});
  ports.push_back({rewriter.getStringAttr("reset"), rewriter.getI1Type(),
                   calyx::Direction::Input,
                   getMandatoryPortAttr(ctx, "reset")});
  ports.push_back({rewriter.getStringAttr("go"), rewriter.getI1Type(),
                   calyx::Direction::Input, getMandatoryPortAttr(ctx, "go")});
  ports.push_back({rewriter.getStringAttr("done"), rewriter.getI1Type(),
                   calyx::Direction::Output,
                   getMandatoryPortAttr(ctx, "done")});
}

/// Creates a new calyx::CombGroupOp or calyx::GroupOp group within compOp.
template <typename TGroup>
static TGroup createGroup(PatternRewriter &rewriter, calyx::ComponentOp compOp,
                          Location loc, Twine uniqueName) {

  IRRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(compOp.getWiresOp().getBody());
  auto groupOp = rewriter.create<TGroup>(loc, uniqueName.str());
  rewriter.createBlock(&groupOp.getBodyRegion());
  return groupOp;
}

/// Get the index'th output of compOp, which is associated with
/// funcOp.
/// This assumes the invariant that both the ComponentOp and FuncOp have the
/// same port ordering.
static Value getComponentOutput(mlir::FuncOp funcOp, calyx::ComponentOp compOp,
                                unsigned index) {
  size_t resIdx = funcOp.getNumArguments() + 3 /*go, reset, clk*/ + index;
  assert(compOp.getNumArguments() > resIdx &&
         "Exceeded number of arguments in the Component");
  return compOp.getArgument(resIdx);
}

/// Creates a SeqOp containing an inner body block.
static calyx::SeqOp createSeqOp(PatternRewriter &rewriter, Location loc) {
  auto seqOp = rewriter.create<calyx::SeqOp>(loc);
  rewriter.createBlock(&seqOp.getRegion());
  return seqOp;
}

//===----------------------------------------------------------------------===//
// Lowering state classes
//===----------------------------------------------------------------------===//

/// ComponentLoweringState handles the current state of lowering of a Calyx
/// component. It is mainly used as a key/value store for recording information
/// during partial lowering, which is required at later lowering passes.
class ProgramLoweringState;
class ComponentLoweringState {
public:
  ComponentLoweringState(ProgramLoweringState &pls, calyx::ComponentOp compOp)
      : programLoweringState(pls), compOp(compOp) {}

  ProgramLoweringState &getProgramState() { return programLoweringState; }

  /// Returns the calyx::ComponentOp associated with this lowering state.
  calyx::ComponentOp getComponentOp() { return compOp; }

  /// Returns a unique name within compOp with the provided prefix.
  std::string getUniqueName(StringRef prefix) {
    std::string prefixStr = prefix.str();
    unsigned idx = prefixIdMap[prefixStr];
    ++prefixIdMap[prefixStr];
    return (prefix + "_" + std::to_string(idx)).str();
  }

  /// Returns a unique name associated with a specific operation.
  StringRef getUniqueName(Operation *op) {
    auto it = opNames.find(op);
    assert(it != opNames.end() && "A unique name should have been set for op");
    return it->second;
  }

  /// Registers a unique name for a given operation using a provided prefix.
  void setUniqueName(Operation *op, StringRef prefix) {
    auto it = opNames.find(op);
    assert(it == opNames.end() && "A unique name was already set for op");
    opNames[op] = getUniqueName(prefix);
  }

  template <typename TLibraryOp>
  TLibraryOp getNewLibraryOpInstance(PatternRewriter &rewriter, Location loc,
                                     TypeRange resTypes) {
    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(compOp.getBody(), compOp.getBody()->begin());
    auto name = TLibraryOp::getOperationName().split(".").second;
    auto uniqueName = getUniqueName(name);
    return rewriter.create<TLibraryOp>(loc, rewriter.getStringAttr(uniqueName),
                                       resTypes);
  }

  /// Register value v as being evaluated when scheduling group.
  void registerEvaluatingGroup(Value v, calyx::GroupInterface group) {
    valueGroupAssigns[v] = group;
  }

  /// Return the group which evaluates the value v. Optionally, caller may
  /// specify the expected type of the group.
  template <typename TGroupOp = calyx::GroupInterface>
  TGroupOp getEvaluatingGroup(Value v) {
    auto it = valueGroupAssigns.find(v);
    assert(it != valueGroupAssigns.end() && "No group evaluating value!");
    if constexpr (std::is_same<TGroupOp, calyx::GroupInterface>::value)
      return it->second;
    else {
      auto group = dyn_cast<TGroupOp>(it->second.getOperation());
      assert(group && "Actual group type differed from expected group type");
      return group;
    }
  }

  /// Register reg as being the idx'th return value register.
  void addReturnReg(calyx::RegisterOp reg, unsigned idx) {
    assert(returnRegs.count(idx) == 0 &&
           "A register was already registered for this index");
    returnRegs[idx] = reg;
  }

  /// Returns the idx'th return value register.
  calyx::RegisterOp getReturnReg(unsigned idx) {
    assert(returnRegs.count(idx) && "No register registered for index!");
    return returnRegs[idx];
  }

  /// Returns an SSA value for an arbitrary precision constant defined within
  /// compOp. A new constant is created if no constant is found.
  Value getConstant(PatternRewriter &rewriter, Location loc, int64_t value,
                    unsigned width) {
    IRRewriter::InsertionGuard guard(rewriter);
    Value v;
    auto it = constants.find(APInt(value, width));
    if (it == constants.end()) {
      rewriter.setInsertionPointToStart(compOp.getBody());
      return v = rewriter.create<hw::ConstantOp>(loc, APInt(value, width));
    }
    return it->second;
  }

  /// Register 'scheduleable' as being generated through lowering 'block'.
  ///
  /// TODO(mortbopet): Add a post-insertion check to ensure that the use-def
  /// ordering invariant holds for the groups. When the control schedule is
  /// generated, scheduleables within a block are emitted sequentially based on
  /// the order that this function was called during conversion.
  ///
  /// Currently, we assume this to always be true. Walking the FuncOp IR implies
  /// sequential iteration over operations within basic blocks.
  void addBlockScheduleable(mlir::Block *block,
                            const Scheduleable &scheduleable) {
    blockScheduleables[block].push_back(scheduleable);
  }

  /// Returns an ordered list of schedulables which registered themselves to be
  /// a result of lowering the block in the source program. The list order
  /// follows def-use chains between the scheduleables in the block.
  SmallVector<Scheduleable> getBlockScheduleables(mlir::Block *block) {
    auto it = blockScheduleables.find(block);
    if (it != blockScheduleables.end())
      return it->second;
    /// In cases of a block resulting in purely combinational logic, no
    /// scheduleables registered themselves with the block.
    return {};
  }

private:
  /// A reference to the parent program lowering state.
  ProgramLoweringState &programLoweringState;

  /// The component which this lowering state is associated to.
  calyx::ComponentOp compOp;

  /// A mapping of string prefixes and the current uniqueness counter for that
  /// prefix. Used to generate unique names.
  std::map<std::string, unsigned> prefixIdMap;

  /// A mapping from Operations and previously assigned unique name of the op.
  std::map<Operation *, std::string> opNames;

  /// A mapping between SSA values and the groups which assign them.
  DenseMap<Value, calyx::GroupInterface> valueGroupAssigns;

  /// A mapping from return value indexes to return value registers.
  DenseMap<unsigned, calyx::RegisterOp> returnRegs;

  /// A mapping of currently available constants in this component.
  DenseMap<APInt, Value> constants;

  /// BlockScheduleables is a list of scheduleables that should be
  /// sequentially executed when executing the associated basic block.
  DenseMap<mlir::Block *, SmallVector<Scheduleable>> blockScheduleables;
};

/// ProgramLoweringState handles the current state of lowering of a Calyx
/// program. It is mainly used as a key/value store for recording information
/// during partial lowering, which is required at later lowering passes.
class ProgramLoweringState {
public:
  explicit ProgramLoweringState(calyx::ProgramOp program,
                                StringRef topLevelFunction)
      : topLevelFunction(topLevelFunction), program(program) {
    getProgram();
  }

  /// Returns a meaningful name for a value within the program scope.
  template <typename ValueOrBlock>
  std::string irName(ValueOrBlock &v) {
    std::string s;
    llvm::raw_string_ostream os(s);
    AsmState asmState(program);
    v.printAsOperand(os, asmState);
    return s;
  }

  /// Returns a meaningful name for a block within the program scope (removes
  /// the ^ prefix from block names).
  std::string blockName(Block *b) {
    auto blockName = irName(*b);
    blockName.erase(std::remove(blockName.begin(), blockName.end(), '^'),
                    blockName.end());
    return blockName;
  }

  /// Returns the component lowering state associated with compOp.
  ComponentLoweringState &compLoweringState(calyx::ComponentOp compOp) {
    auto it = compStates.find(compOp);
    if (it != compStates.end())
      return it->second;

    /// Create a new ComponentLoweringState for the compOp.
    auto newCompStateIt = compStates.try_emplace(compOp, *this, compOp);
    return newCompStateIt.first->second;
  }

  /// Returns the current program.
  calyx::ProgramOp getProgram() {
    assert(program.getOperation() != nullptr);
    return program;
  }

  /// Returns the name of the top-level function in the source program.
  StringRef getTopLevelFunction() const { return topLevelFunction; }

private:
  StringRef topLevelFunction;
  calyx::ProgramOp program;
  DenseMap<Operation *, ComponentLoweringState> compStates;
};

/// Creates register assignment operations within the provided groupOp.
static void buildAssignmentsForRegisterWrite(ComponentLoweringState &state,
                                             PatternRewriter &rewriter,
                                             calyx::GroupOp groupOp,
                                             calyx::RegisterOp &reg,
                                             Value inputValue) {
  IRRewriter::InsertionGuard guard(rewriter);
  auto loc = inputValue.getLoc();
  rewriter.setInsertionPointToEnd(groupOp.getBody());
  rewriter.create<calyx::AssignOp>(loc, reg.in(), inputValue, Value());
  rewriter.create<calyx::AssignOp>(
      loc, reg.write_en(), state.getConstant(rewriter, loc, 1, 1), Value());
  rewriter.create<calyx::GroupDoneOp>(loc, reg.donePort(), Value());
}

//===----------------------------------------------------------------------===//
// Partial lowering infrastructure
//===----------------------------------------------------------------------===//

/// Base class for partial lowering passes. A partial lowering pass
/// modifies the root operation in place, but does not replace the root
/// operation.
/// The RewritePatternType template parameter allows for using both
/// OpRewritePattern (default) or OpInterfaceRewritePattern.
template <class OpType,
          template <class> class RewritePatternType = OpRewritePattern>
class PartialLoweringPattern : public RewritePatternType<OpType> {
public:
  using RewritePatternType<OpType>::RewritePatternType;
  PartialLoweringPattern(MLIRContext *ctx, LogicalResult &resRef)
      : RewritePatternType<OpType>(ctx), partialPatternRes(resRef) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(
        op, [&] { partialPatternRes = partiallyLower(op, rewriter); });
    return partialPatternRes;
  }

  virtual LogicalResult partiallyLower(OpType op,
                                       PatternRewriter &rewriter) const = 0;

private:
  LogicalResult &partialPatternRes;
};

/// Creates a new register within the component associated to 'compState'.
static calyx::RegisterOp createReg(ComponentLoweringState &compState,
                                   PatternRewriter &rewriter, Location loc,
                                   Twine prefix, size_t width) {
  IRRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(compState.getComponentOp().getBody());
  return rewriter.create<calyx::RegisterOp>(
      loc, rewriter.getStringAttr(prefix + "_reg"), width);
}

//===----------------------------------------------------------------------===//
// Partial lowering patterns
//===----------------------------------------------------------------------===//

/// FuncOpPartialLoweringPatterns are patterns which intend to match on FuncOps
/// and then perform their own walking of the IR. FuncOpPartialLoweringPatterns
/// have direct access to the ComponentLoweringState for the corresponding
/// component of the matched FuncOp.
class FuncOpPartialLoweringPattern
    : public PartialLoweringPattern<mlir::FuncOp> {
public:
  FuncOpPartialLoweringPattern(MLIRContext *context, LogicalResult &resRef,
                               FuncMapping &_funcMap, ProgramLoweringState &pls)
      : PartialLoweringPattern(context, resRef), funcMap(_funcMap), pls(pls) {}

  LogicalResult partiallyLower(mlir::FuncOp funcOp,
                               PatternRewriter &rewriter) const override final {
    // Initialize the component op references if a calyx::ComponentOp has been
    // created for the matched funcOp.
    auto it = funcMap.find(funcOp);
    if (it != funcMap.end()) {
      compOp = &it->second;
      compLoweringState = &pls.compLoweringState(*getComponent());
    }

    return PartiallyLowerFuncToComp(funcOp, rewriter);
  }

  // Returns the component operation associated with the currently executing
  // partial lowering.
  calyx::ComponentOp *getComponent() const {
    assert(
        compOp != nullptr &&
        "Expected component op to have been set during pattern construction");
    return compOp;
  }

  // Returns the component state associated with the currently executing
  // partial lowering.
  ComponentLoweringState &getComponentState() const {
    assert(compLoweringState != nullptr &&
           "Expected component lowering state to have been set during pattern "
           "construction");
    return *compLoweringState;
  }

  ProgramLoweringState &progState() const { return pls; }

  /// Partial lowering implementation.
  virtual LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const = 0;

protected:
  FuncMapping &funcMap;

private:
  mutable calyx::ComponentOp *compOp = nullptr;
  mutable ComponentLoweringState *compLoweringState = nullptr;
  ProgramLoweringState &pls;
};

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Iterate through the operations of a source function and instantiate
/// components or primitives based on the type of the operations.
class BuildOpGroups : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    /// We walk the operations of the funcOp to ensure that all def's have
    /// been visited before their uses.
    bool opBuiltSuccessfully = true;
    funcOp.walk([&](Operation *_op) {
      opBuiltSuccessfully &=
          TypeSwitch<mlir::Operation *, bool>(_op)
              .template Case<ConstantOp, ReturnOp,
                             /// standard arithmetic
                             AddIOp, SubIOp, CmpIOp, ShiftLeftOp,
                             UnsignedShiftRightOp, SignedShiftRightOp, AndOp,
                             XOrOp, OrOp, ZeroExtendIOp, TruncateIOp>(
                  [&](auto op) { return buildOp(rewriter, op).succeeded(); })
              .template Case<scf::WhileOp, mlir::FuncOp, scf::ConditionOp>(
                  [&](auto) {
                    /// Skip: these special cases will be handled separately.
                    return true;
                  })
              .Default([&](auto) {
                assert(false && "Unhandled operation during BuildOpGroups()");
                return false;
              });

      return opBuiltSuccessfully ? WalkResult::advance()
                                 : WalkResult::interrupt();
    });

    return success(opBuiltSuccessfully);
  }

private:
  /// Op builder specializations.
  LogicalResult buildOp(PatternRewriter &rewriter, ConstantOp constOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AddIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, SubIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        UnsignedShiftRightOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, SignedShiftRightOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShiftLeftOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AndOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, OrOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, XOrOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, CmpIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, TruncateIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ZeroExtendIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ReturnOp op) const;

  /// buildLibraryOp will build a TCalyxLibOp inside a TGroupOp based on the
  /// source operation TSrcOp.
  template <typename TGroupOp, typename TCalyxLibOp, typename TSrcOp>
  LogicalResult buildLibraryOp(PatternRewriter &rewriter, TSrcOp op,
                               TypeRange srcTypes, TypeRange dstTypes) const {
    SmallVector<Type> types;
    llvm::append_range(types, srcTypes);
    llvm::append_range(types, dstTypes);

    auto calyxOp = getComponentState().getNewLibraryOpInstance<TCalyxLibOp>(
        rewriter, op.getLoc(), types);

    auto directions = calyxOp.portDirections();
    SmallVector<Value, 4> opInputPorts;
    SmallVector<Value, 4> opOutputPorts;
    for (auto dir : enumerate(directions)) {
      if (dir.value() == calyx::Direction::Input)
        opInputPorts.push_back(calyxOp.getResult(dir.index()));
      else
        opOutputPorts.push_back(calyxOp.getResult(dir.index()));
    }
    assert(
        opInputPorts.size() == op->getNumOperands() &&
        opOutputPorts.size() == op->getNumResults() &&
        "Expected an equal number of in/out ports in the Calyx library op with "
        "respect to the number of operands/results of the source operation.");

    /// Create assignments to the inputs of the library op.
    auto group = createGroupForOp<TGroupOp>(rewriter, op);
    rewriter.setInsertionPointToEnd(group.getBody());
    for (auto dstOp : enumerate(opInputPorts))
      rewriter.create<calyx::AssignOp>(op.getLoc(), dstOp.value(),
                                       op->getOperand(dstOp.index()), Value());

    /// Replace the result values of the source operator with the new operator.
    for (auto res : enumerate(opOutputPorts)) {
      getComponentState().registerEvaluatingGroup(res.value(), group);
      op->getResult(res.index()).replaceAllUsesWith(res.value());
    }
    return success();
  }

  /// buildLibraryOp which provides in- and output types based on the operands
  /// and results of the op argument.
  template <typename TGroupOp, typename TCalyxLibOp, typename TSrcOp>
  LogicalResult buildLibraryOp(PatternRewriter &rewriter, TSrcOp op) const {
    return buildLibraryOp<TGroupOp, TCalyxLibOp, TSrcOp>(
        rewriter, op, op.getOperandTypes(), op->getResultTypes());
  }

  /// Creates a group named by the basic block which the input op resides in.
  template <typename TGroupOp>
  TGroupOp createGroupForOp(PatternRewriter &rewriter, Operation *op) const {
    Block *block = op->getBlock();
    auto groupName = getComponentState().getUniqueName(
        getComponentState().getProgramState().blockName(block));
    return createGroup<TGroupOp>(rewriter, getComponentState().getComponentOp(),
                                 block->front().getLoc(), groupName);
  }
};

/// For each return statement, we create a new group for assigning to the
/// previously created return value registers.
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ReturnOp retOp) const {
  if (retOp.getNumOperands() == 0)
    return success();

  std::string groupName = getComponentState().getUniqueName("ret_assign");
  Value anyRegDone;
  auto groupOp = createGroup<calyx::GroupOp>(rewriter, *getComponent(),
                                             retOp.getLoc(), groupName);
  for (auto op : enumerate(retOp.getOperands())) {
    auto reg = getComponentState().getReturnReg(op.index());
    buildAssignmentsForRegisterWrite(getComponentState(), rewriter, groupOp,
                                     reg, op.value());
  }
  /// Schedule group for execution for when executing the return op block.
  getComponentState().addBlockScheduleable(retOp->getBlock(), groupOp);
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ConstantOp constOp) const {
  /// Move constant operations to the compOp body as hw::ConstantOp's.
  APInt value;
  matchConstantOp(constOp, value);
  auto hwConstOp = rewriter.replaceOpWithNewOp<hw::ConstantOp>(constOp, value);
  hwConstOp->moveAfter(getComponent()->getBody(),
                       getComponent()->getBody()->begin());
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AddIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::AddLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     SubIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SubLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     UnsignedShiftRightOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::RshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     SignedShiftRightOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SrshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShiftLeftOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::LshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AndOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::AndLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter, OrOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::OrLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     XOrOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::XorLibOp>(rewriter, op);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     CmpIOp op) const {
  switch (op.predicate()) {
  case CmpIPredicate::eq:
    return buildLibraryOp<calyx::CombGroupOp, calyx::EqLibOp>(rewriter, op);
  case CmpIPredicate::ne:
    return buildLibraryOp<calyx::CombGroupOp, calyx::NeqLibOp>(rewriter, op);
  case CmpIPredicate::uge:
    return buildLibraryOp<calyx::CombGroupOp, calyx::GeLibOp>(rewriter, op);
  case CmpIPredicate::ult:
    return buildLibraryOp<calyx::CombGroupOp, calyx::LtLibOp>(rewriter, op);
  case CmpIPredicate::ugt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::GtLibOp>(rewriter, op);
  case CmpIPredicate::ule:
    return buildLibraryOp<calyx::CombGroupOp, calyx::LeLibOp>(rewriter, op);
  case CmpIPredicate::sge:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SgeLibOp>(rewriter, op);
  case CmpIPredicate::slt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SltLibOp>(rewriter, op);
  case CmpIPredicate::sgt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SgtLibOp>(rewriter, op);
  case CmpIPredicate::sle:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SleLibOp>(rewriter, op);
  default:
    llvm_unreachable("unsupported comparison predicate");
  }
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     TruncateIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SliceLibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ZeroExtendIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::PadLibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}

/// Creates a new Calyx component for each FuncOp in the program.
struct FuncOpConversion : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    /// Create I/O ports.
    SmallVector<calyx::PortInfo> ports;
    FunctionType funcType = funcOp.getType();
    for (auto &arg : enumerate(funcOp.getArguments()))
      ports.push_back(calyx::PortInfo{
          rewriter.getStringAttr("in" + std::to_string(arg.index())),
          arg.value().getType(), calyx::Direction::Input,
          DictionaryAttr::get(rewriter.getContext(), {})});

    for (auto &res : enumerate(funcType.getResults()))
      ports.push_back(calyx::PortInfo{
          rewriter.getStringAttr("out" + std::to_string(res.index())),
          res.value(), calyx::Direction::Output,
          DictionaryAttr::get(rewriter.getContext(), {})});

    addMandatoryComponentPorts(rewriter, ports);

    /// Create a calyx::ComponentOp corresponding to the to-be-lowered function.
    auto compOp = rewriter.create<calyx::ComponentOp>(
        funcOp.getLoc(), rewriter.getStringAttr(funcOp.sym_name()), ports);
    rewriter.createBlock(&compOp.getWiresOp().getBodyRegion());
    rewriter.createBlock(&compOp.getControlOp().getBodyRegion());

    /// Rewrite the funcOp SSA argument values to the CompOp arguments.
    for (auto &arg : enumerate(funcOp.getArguments())) {
      arg.value().replaceAllUsesWith(compOp.getArgument(arg.index()));
    }

    /// Store function to component mapping for future reference.
    funcMap[funcOp] = compOp;
    return success();
  }
};

/// Builds registers for the return statement of the program and constant
/// assignments to the component return value.
class BuildReturnRegs : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {

    for (auto argType : enumerate(funcOp.getType().getResults())) {
      assert(argType.value().isa<IntegerType>() && "unsupported return type");
      unsigned width = argType.value().getIntOrFloatBitWidth();
      std::string name = "ret_arg" + std::to_string(argType.index());
      auto reg = createReg(getComponentState(), rewriter, funcOp.getLoc(), name,
                           width);
      getComponentState().addReturnReg(reg, argType.index());

      rewriter.setInsertionPointToStart(getComponent()->getWiresOp().getBody());
      rewriter.create<calyx::AssignOp>(
          funcOp->getLoc(),
          getComponentOutput(funcOp, *getComponent(), argType.index()),
          reg.out(), Value());
    }
    return success();
  }
};

struct ModuleOpConversion : public OpRewritePattern<mlir::ModuleOp> {
  ModuleOpConversion(MLIRContext *context, StringRef topLevelFunction,
                     calyx::ProgramOp *programOpOutput)
      : OpRewritePattern<mlir::ModuleOp>(context),
        programOpOutput(programOpOutput), topLevelFunction(topLevelFunction) {
    assert(programOpOutput->getOperation() == nullptr &&
           "this function will set programOpOutput post module conversion");
  }

  LogicalResult matchAndRewrite(mlir::ModuleOp moduleOp,
                                PatternRewriter &rewriter) const override {
    if (!moduleOp.getOps<calyx::ProgramOp>().empty())
      return failure();

    rewriter.updateRootInPlace(moduleOp, [&] {
      // Create ProgramOp
      rewriter.setInsertionPointAfter(moduleOp);
      auto programOp = rewriter.create<calyx::ProgramOp>(
          moduleOp.getLoc(), StringAttr::get(getContext(), topLevelFunction));

      // Inline the module body region
      rewriter.inlineRegionBefore(moduleOp.getBodyRegion(),
                                  programOp.getBodyRegion(),
                                  programOp.getBodyRegion().end());

      // Inlining the body region also removes ^bb0 from the module body
      // region, so recreate that, before finally inserting the programOp
      auto moduleBlock = rewriter.createBlock(&moduleOp.getBodyRegion());
      rewriter.setInsertionPointToStart(moduleBlock);
      rewriter.insert(programOp);
      *programOpOutput = programOp;
    });
    return success();
  }

private:
  calyx::ProgramOp *programOpOutput = nullptr;
  StringRef topLevelFunction;
};

/// Builds a control schedule by traversing the CFG of the function and
/// associating this with the previously created groups.
/// For simplicity, the generated control flow is expanded for all possible
/// paths in the input DAG. This elaborated control flow is later reduced in
/// the runControlFlowSimplification passes.
class BuildControl : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    auto *entryBlock = &funcOp.getBlocks().front();
    rewriter.setInsertionPointToStart(getComponent()->getControlOp().getBody());
    auto topLevelSeqOp = createSeqOp(rewriter, funcOp.getLoc());
    DenseSet<Block *> path;
    return buildCFGControl(path, rewriter, topLevelSeqOp.getBody(), nullptr,
                           entryBlock);
  }

private:
  /// Sequentially schedules the groups that registered themselves with
  /// 'block'.
  LogicalResult scheduleBasicBlock(PatternRewriter &rewriter,
                                   const DenseSet<Block *> &path,
                                   mlir::Block *parentCtrlBlock,
                                   mlir::Block *block) const {
    auto compBlockScheduleables =
        getComponentState().getBlockScheduleables(block);
    auto loc = block->front().getLoc();

    if (compBlockScheduleables.size() > 1) {
      auto seqOp = createSeqOp(rewriter, loc);
      parentCtrlBlock = seqOp.getBody();
    }

    rewriter.setInsertionPointToEnd(parentCtrlBlock);

    for (auto &group : compBlockScheduleables) {
      if (auto groupPtr = std::get_if<calyx::GroupOp>(&group); groupPtr) {
        rewriter.create<calyx::EnableOp>(loc, groupPtr->sym_name(),
                                         rewriter.getArrayAttr({}));
      } else
        llvm_unreachable("Unknown scheduleable");
    }
    return success();
  }

  /// Schedules a block by inserting a branch argument assignment block (if any)
  /// before recursing into the scheduling of the block innards.
  /// Blocks 'from' and 'to' refer to blocks in the source program.
  /// parentCtrlBlock refers to the control block wherein control operations are
  /// to be inserted.
  LogicalResult schedulePath(PatternRewriter &rewriter,
                             const DenseSet<Block *> &path, Location loc,
                             Block *from, Block *to,
                             Block *parentCtrlBlock) const {
    /// In the following commits, basic block argument passing will be performed
    /// here...
    return buildCFGControl(path, rewriter, parentCtrlBlock, from, to);
  }

  LogicalResult buildCFGControl(DenseSet<Block *> path,
                                PatternRewriter &rewriter,
                                mlir::Block *parentCtrlBlock,
                                mlir::Block *preBlock,
                                mlir::Block *block) const {
    if (path.count(block) != 0)
      return preBlock->getTerminator()->emitError()
             << "CFG backedge detected. Loops must be raised to 'scf.while' or "
                "'scf.for' operations.";

    rewriter.setInsertionPointToEnd(parentCtrlBlock);
    LogicalResult bbSchedResult =
        scheduleBasicBlock(rewriter, path, parentCtrlBlock, block);
    if (bbSchedResult.failed())
      return bbSchedResult;

    path.insert(block);
    auto successors = block->getSuccessors();
    auto nSuccessors = successors.size();
    if (nSuccessors > 0) {
      auto brOp = dyn_cast<BranchOpInterface>(block->getTerminator());
      assert(brOp);
      if (nSuccessors > 1) {
        /// @todo: we could choose to support ie. std.switch, but it would
        /// probably be easier to just require it to be lowered beforehand.
        assert(nSuccessors == 2 &&
               "only conditional branches supported for now...");
        /// Wrap each branch inside an if/else.
        auto cond = brOp->getOperand(0);
        auto condGroup =
            getComponentState().getEvaluatingGroup<calyx::CombGroupOp>(cond);
        auto symbolAttr = FlatSymbolRefAttr::get(
            StringAttr::get(getContext(), condGroup.sym_name()));
        auto ifOp =
            rewriter.create<calyx::IfOp>(brOp->getLoc(), cond, symbolAttr);
        auto *thenCtrlBlock =
            rewriter.createBlock(&ifOp.thenRegion(), ifOp.thenRegion().end());
        auto *elseCtrlBlock =
            rewriter.createBlock(&ifOp.elseRegion(), ifOp.elseRegion().end());
        rewriter.setInsertionPointToEnd(thenCtrlBlock);
        auto thenSeqOp = createSeqOp(rewriter, brOp.getLoc());
        rewriter.setInsertionPointToEnd(elseCtrlBlock);
        auto elseSeqOp = createSeqOp(rewriter, brOp.getLoc());
        bool trueBrSchedSuccess =
            schedulePath(rewriter, path, brOp.getLoc(), block, successors[0],
                         thenSeqOp.getBody())
                .succeeded();
        bool falseBrSchedSuccess = true;
        if (trueBrSchedSuccess) {
          falseBrSchedSuccess =
              schedulePath(rewriter, path, brOp.getLoc(), block, successors[1],
                           elseSeqOp.getBody())
                  .succeeded();
        }

        return success(trueBrSchedSuccess && falseBrSchedSuccess);
      } else {
        /// Schedule sequentially within the current parent control block.
        return schedulePath(rewriter, path, brOp.getLoc(), block,
                            successors.front(), parentCtrlBlock);
      }
    }
    return success();
  }
};

/// Erases FuncOp operations.
class CleanupFuncOps : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    rewriter.eraseOp(funcOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//
class SCFToCalyxPass : public SCFToCalyxBase<SCFToCalyxPass> {
public:
  SCFToCalyxPass()
      : SCFToCalyxBase<SCFToCalyxPass>(), partialPatternRes(success()) {}
  void runOnOperation() override;

  LogicalResult setTopLevelFunction(mlir::ModuleOp moduleOp,
                                    std::string &topLevelFunction) {
    if (!topLevelFunctionOpt.empty()) {
      if (SymbolTable::lookupSymbolIn(moduleOp, topLevelFunctionOpt) ==
          nullptr) {
        moduleOp.emitError() << "Top level function '" << topLevelFunctionOpt
                             << "' not found in module.";
        return failure();
      }
      topLevelFunction = topLevelFunctionOpt;
    } else {
      /// No top level function set; infer top level if the module only contains
      /// a single function, else, throw error.
      auto funcOps = moduleOp.getOps<mlir::FuncOp>();
      if (std::distance(funcOps.begin(), funcOps.end()) == 1)
        topLevelFunction = (*funcOps.begin()).sym_name().str();
      else {
        moduleOp.emitError()
            << "Module contains multiple functions, but no top level "
               "function was set. Please see --top-level-function";
        return failure();
      }
    }
    return success();
  }

  struct LoweringPattern {
    enum class Strategy { Once, Greedy };
    RewritePatternSet pattern;
    Strategy strategy;
  };

  //// Creates a new Calyx program with the contents of the source module
  /// inlined within.
  /// Furthermore, this function performs validation on the input function,
  /// to ensure that we've implemented the capabilities necessary to convert
  /// it.
  LogicalResult createProgram(StringRef topLevelFunction,
                              calyx::ProgramOp *programOpOut) {
    // Program legalization - the partial conversion driver will not run
    // unless some pattern is provided - provide a dummy pattern.
    struct DummyPattern : public OpRewritePattern<mlir::ModuleOp> {
      using OpRewritePattern::OpRewritePattern;
      LogicalResult matchAndRewrite(mlir::ModuleOp,
                                    PatternRewriter &) const override {
        return failure();
      }
    };

    ConversionTarget target(getContext());
    target.addLegalDialect<calyx::CalyxDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addIllegalDialect<hw::HWDialect>();
    target.addIllegalDialect<comb::CombDialect>();

    // For loops should have been lowered to while loops
    target.addIllegalOp<scf::ForOp>();

    // Only accept std operations which we've added lowerings for
    target.addIllegalDialect<StandardOpsDialect>();
    target.addLegalOp<AddIOp, SubIOp, CmpIOp, ShiftLeftOp, UnsignedShiftRightOp,
                      SignedShiftRightOp, AndOp, XOrOp, OrOp, ZeroExtendIOp,
                      TruncateIOp, CondBranchOp, BranchOp, ReturnOp, ConstantOp,
                      IndexCastOp>();

    RewritePatternSet legalizePatterns(&getContext());
    legalizePatterns.add<DummyPattern>(&getContext());
    DenseSet<Operation *> legalizedOps;
    if (applyPartialConversion(getOperation(), target,
                               std::move(legalizePatterns))
            .failed())
      return failure();

    // Program conversion
    RewritePatternSet conversionPatterns(&getContext());
    conversionPatterns.add<ModuleOpConversion>(&getContext(), topLevelFunction,
                                               programOpOut);
    return applyOpPatternsAndFold(getOperation(),
                                  std::move(conversionPatterns));
  }

  /// 'Once' patterns are expected to take an additional LogicalResult&
  /// argument, to forward their result state (greedyPatternRewriteDriver
  /// results are skipped for Once patterns).
  template <typename TPattern, typename... PatternArgs>
  void addOncePattern(SmallVectorImpl<LoweringPattern> &patterns,
                      PatternArgs &&...args) {
    RewritePatternSet ps(&getContext());
    ps.add<TPattern>(&getContext(), partialPatternRes, args...);
    patterns.push_back(
        LoweringPattern{std::move(ps), LoweringPattern::Strategy::Once});
  }

  template <typename TPattern, typename... PatternArgs>
  void addGreedyPattern(SmallVectorImpl<LoweringPattern> &patterns,
                        PatternArgs &&...args) {
    RewritePatternSet ps(&getContext());
    ps.add<TPattern>(&getContext(), args...);
    patterns.push_back(
        LoweringPattern{std::move(ps), LoweringPattern::Strategy::Greedy});
  }

  LogicalResult runPartialPattern(RewritePatternSet &pattern, bool runOnce) {
    auto &nativePatternSet = pattern.getNativePatterns();
    assert(nativePatternSet.size() == 1 &&
           "Should only apply 1 partial lowering pattern at once");

    // During component creation, the function body is inlined into the
    // component body for further processing. However, proper control flow
    // will only be established later in the conversion process, so ensure
    // that rewriter optimizations (especially DCE) are disabled.
    GreedyRewriteConfig config;
    config.enableRegionSimplification = false;
    if (runOnce)
      config.maxIterations = 1;

    /// Can't return applyPatternsAndFoldGreedily. Root isn't
    /// necessarily erased so it will always return failed(). Instead,
    /// forward the 'succeeded' value from PartialLoweringPatternBase.
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(pattern),
                                       config);
    return partialPatternRes;
  }

private:
  LogicalResult partialPatternRes;
  std::shared_ptr<ProgramLoweringState> loweringState = nullptr;
};

void SCFToCalyxPass::runOnOperation() {
  std::string topLevelFunction;
  if (failed(setTopLevelFunction(getOperation(), topLevelFunction))) {
    signalPassFailure();
    return;
  }

  /// Start conversion
  calyx::ProgramOp programOp;
  if (failed(createProgram(topLevelFunction, &programOp))) {
    signalPassFailure();
    return;
  }
  assert(programOp.getOperation() != nullptr &&
         "programOp should have been set during module "
         "conversion, if module conversion succeeded.");
  loweringState =
      std::make_shared<ProgramLoweringState>(programOp, topLevelFunction);

  /// --------------------------------------------------------------------------
  /// If you are a developer, it may be helpful to add a
  /// 'getOperation()->dump()' call after the execution of each stage to
  /// view the transformations that's going on.
  /// --------------------------------------------------------------------------
  FuncMapping funcMap;
  SmallVector<LoweringPattern, 8> loweringPatterns;

  /// Creates a new Calyx component for each FuncOp in the inpurt module.
  addOncePattern<FuncOpConversion>(loweringPatterns, funcMap, *loweringState);

  /// This pattern creates registers for the function return values.
  addOncePattern<BuildReturnRegs>(loweringPatterns, funcMap, *loweringState);

  /// This pattern converts operations within basic blocks to Calyx library
  /// operators. Combinational operations are assigned inside a
  /// calyx::CombGroupOp, and sequential inside calyx::GroupOps.
  /// Sequential groups are registered with the Block* of which the operation
  /// originated from. This is used during control schedule generation. By
  /// having a distinct group for each operation, groups are analogous to SSA
  /// values in the source program.
  addOncePattern<BuildOpGroups>(loweringPatterns, funcMap, *loweringState);

  /// This pattern traverses the CFG of the program and generates a control
  /// schedule based on the calyx::GroupOp's which were registered for each
  /// basic block in the source function.
  addOncePattern<BuildControl>(loweringPatterns, funcMap, *loweringState);

  /// This pattern removes the source FuncOp which has now been converted into
  /// a Calyx component.
  addOncePattern<CleanupFuncOps>(loweringPatterns, funcMap, *loweringState);

  /// Sequentially apply each lowering pattern.
  for (auto &pat : loweringPatterns) {
    LogicalResult partialPatternRes = runPartialPattern(
        pat.pattern,
        /*runOnce=*/pat.strategy == LoweringPattern::Strategy::Once);
    if (succeeded(partialPatternRes))
      continue;
    signalPassFailure();
    return;
  }
}

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createSCFToCalyxPass() {
  return std::make_unique<SCFToCalyxPass>();
}

} // namespace circt
