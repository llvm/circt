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

#include "circt/Conversion/SCFToCalyx.h"
#include "../PassDetail.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxLoweringUtils.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include <variant>

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::cf;
using namespace mlir::func;

namespace circt {
namespace scftocalyx {

//===----------------------------------------------------------------------===//
// Utility types
//===----------------------------------------------------------------------===//

/// A mapping is maintained between a function operation and its corresponding
/// Calyx component.
using FuncMapping = DenseMap<FuncOp, calyx::ComponentOp>;

class ScfWhileOp : public calyx::WhileOpInterface<scf::WhileOp> {
public:
  explicit ScfWhileOp(scf::WhileOp op)
      : calyx::WhileOpInterface<scf::WhileOp>(op) {}

  Block::BlockArgListType getBodyArgs() override {
    return getOperation().getAfterArguments();
  }

  Block *getBodyBlock() override { return &getOperation().getAfter().front(); }

  Block *getConditionBlock() override {
    return &getOperation().getBefore().front();
  }

  Value getConditionValue() override {
    return getOperation().getConditionOp().getOperand(0);
  }

  Optional<uint64_t> getBound() override { return None; }
};

struct LoopScheduleable {
  /// While operation to schedule.
  ScfWhileOp whileOp;
  /// The group(s) to schedule before the while operation These groups should
  /// set the initial value(s) of the loop init_args register(s).
  SmallVector<calyx::GroupOp> initGroups;
};

struct WhileScheduleable : LoopScheduleable {};

/// A variant of types representing scheduleable operations.
using Scheduleable = std::variant<calyx::GroupOp, WhileScheduleable>;

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
    assert(opNames.find(op) == opNames.end() &&
           "A unique name was already set for op");
    opNames[op] = getUniqueName(prefix);
  }

  template <typename TLibraryOp>
  TLibraryOp getNewLibraryOpInstance(PatternRewriter &rewriter, Location loc,
                                     TypeRange resTypes) {
    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(compOp.getBody(), compOp.getBody()->begin());
    auto name = TLibraryOp::getOperationName().split(".").second;
    return rewriter.create<TLibraryOp>(loc, getUniqueName(name), resTypes);
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

  /// Register 'grp' as a group which performs block argument
  /// register transfer when transitioning from basic block from to to.
  void addBlockArgGroup(Block *from, Block *to, calyx::GroupOp grp) {
    blockArgGroups[from][to].push_back(grp);
  }

  /// Returns a list of groups to be evaluated to perform the block argument
  /// register assignments when transitioning from basic block 'from' to 'to'.
  ArrayRef<calyx::GroupOp> getBlockArgGroups(Block *from, Block *to) {
    return blockArgGroups[from][to];
  }

  /// Register reg as being the idx'th argument register for block.
  void addBlockArgReg(Block *block, calyx::RegisterOp reg, unsigned idx) {
    assert(blockArgRegs[block].count(idx) == 0);
    assert(idx < block->getArguments().size());
    blockArgRegs[block][idx] = reg;
  }

  /// Return a mapping of block argument indices to block argument registers.
  const DenseMap<unsigned, calyx::RegisterOp> &getBlockArgRegs(Block *block) {
    return blockArgRegs[block];
  }

  /// Register reg as being the idx'th iter_args register for 'whileOp'.
  void addWhileIterReg(ScfWhileOp whileOp, calyx::RegisterOp reg,
                       unsigned idx) {
    assert(whileIterRegs[whileOp.getOperation()].count(idx) == 0 &&
           "A register was already registered for the given while iter_arg "
           "index");
    assert(idx < whileOp.getBodyArgs().size());
    whileIterRegs[whileOp.getOperation()][idx] = reg;
  }

  /// Return a mapping of block argument indices to block argument registers.
  calyx::RegisterOp getWhileIterReg(ScfWhileOp whileOp, unsigned idx) {
    auto iterRegs = getWhileIterRegs(whileOp);
    auto it = iterRegs.find(idx);
    assert(it != iterRegs.end() &&
           "No iter arg register set for the provided index");
    return it->second;
  }

  /// Return a mapping of block argument indices to block argument registers.
  const DenseMap<unsigned, calyx::RegisterOp> &
  getWhileIterRegs(ScfWhileOp whileOp) {
    return whileIterRegs[whileOp.getOperation()];
  }

  /// Registers grp to be the while latch group of whileOp.
  void setWhileLatchGroup(ScfWhileOp whileOp, calyx::GroupOp grp) {
    assert(whileLatchGroups.count(whileOp.getOperation()) == 0 &&
           "A latch group was already set for this whileOp");
    whileLatchGroups[whileOp.getOperation()] = grp;
  }

  /// Retrieve the while latch group registered for whileOp.
  calyx::GroupOp getWhileLatchGroup(ScfWhileOp whileOp) {
    auto it = whileLatchGroups.find(whileOp.getOperation());
    assert(it != whileLatchGroups.end() &&
           "No while latch group was set for this whileOp");
    return it->second;
  }

  /// Registers a memory interface as being associated with a memory identified
  /// by 'memref'.
  void registerMemoryInterface(Value memref,
                               const calyx::MemoryInterface &memoryInterface) {
    assert(memref.getType().isa<MemRefType>());
    assert(memories.find(memref) == memories.end() &&
           "Memory already registered for memref");
    memories[memref] = memoryInterface;
  }

  /// Returns the memory interface registered for the given memref.
  calyx::MemoryInterface getMemoryInterface(Value memref) {
    assert(memref.getType().isa<MemRefType>());
    auto it = memories.find(memref);
    assert(it != memories.end() && "No memory registered for memref");
    return it->second;
  }

  /// If v is an input to any memory registered within this component, returns
  /// the memory. If not, returns null.
  Optional<calyx::MemoryInterface> isInputPortOfMemory(Value v) {
    for (auto &memIf : memories) {
      auto &mem = memIf.getSecond();
      if (mem.writeEn() == v || mem.writeData() == v ||
          llvm::any_of(mem.addrPorts(), [=](Value port) { return port == v; }))
        return {mem};
    }
    return {};
  }

  /// Assign a mapping between the source funcOp result indices and the
  /// corresponding output port indices of this componentOp.
  void setFuncOpResultMapping(const DenseMap<unsigned, unsigned> &mapping) {
    funcOpResultMapping = mapping;
  }

  /// Get the output port index of this component for which the funcReturnIdx of
  /// the original function maps to.
  unsigned getFuncOpResultMapping(unsigned funcReturnIdx) {
    auto it = funcOpResultMapping.find(funcReturnIdx);
    assert(it != funcOpResultMapping.end() &&
           "No component return port index recorded for the requested function "
           "return index");
    return it->second;
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

  /// BlockScheduleables is a list of scheduleables that should be
  /// sequentially executed when executing the associated basic block.
  DenseMap<mlir::Block *, SmallVector<Scheduleable>> blockScheduleables;

  /// A mapping from blocks to block argument registers.
  DenseMap<Block *, DenseMap<unsigned, calyx::RegisterOp>> blockArgRegs;

  /// Block arg groups is a list of groups that should be sequentially
  /// executed when passing control from the source to destination block.
  /// Block arg groups are executed before blockScheduleables (akin to a
  /// phi-node).
  DenseMap<Block *, DenseMap<Block *, SmallVector<calyx::GroupOp>>>
      blockArgGroups;

  /// A while latch group is a group that should be sequentially executed when
  /// finishing a while loop body. The execution of this group will write the
  /// yield'ed loop body values to the iteration argument registers.
  DenseMap<Operation *, calyx::GroupOp> whileLatchGroups;

  /// A mapping from while ops to iteration argument registers.
  DenseMap<Operation *, DenseMap<unsigned, calyx::RegisterOp>> whileIterRegs;

  /// A mapping from memref's to their corresponding Calyx memory interface.
  DenseMap<Value, calyx::MemoryInterface> memories;

  /// A mapping between the source funcOp result indices and the corresponding
  /// output port indices of this componentOp.
  DenseMap<unsigned, unsigned> funcOpResultMapping;
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
  rewriter.create<calyx::AssignOp>(loc, reg.in(), inputValue);
  rewriter.create<calyx::AssignOp>(
      loc, reg.write_en(),
      createConstant(loc, rewriter, state.getComponentOp(), 1, 1));
  rewriter.create<calyx::GroupDoneOp>(loc, reg.done());
}

/// Creates a new group that assigns the 'ops' values to the iter arg registers
/// of the 'whileOp'.
static calyx::GroupOp buildWhileIterArgAssignments(
    PatternRewriter &rewriter, ComponentLoweringState &state, Location loc,
    ScfWhileOp whileOp, Twine uniqueSuffix, MutableArrayRef<OpOperand> ops) {
  assert(whileOp.getOperation());
  /// Pass iteration arguments through registers. This follows closely
  /// to what is done for branch ops.
  auto groupName = "assign_" + uniqueSuffix;
  auto groupOp = calyx::createGroup<calyx::GroupOp>(
      rewriter, state.getComponentOp(), loc, groupName);
  /// Create register assignment for each iter_arg. a calyx::GroupDone signal
  /// is created for each register. These will be &'ed together in
  /// MultipleGroupDonePattern.
  for (auto &arg : ops) {
    auto reg = state.getWhileIterReg(whileOp, arg.getOperandNumber());
    buildAssignmentsForRegisterWrite(state, rewriter, groupOp, reg, arg.get());
  }
  return groupOp;
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

//===----------------------------------------------------------------------===//
// Partial lowering patterns
//===----------------------------------------------------------------------===//

/// FuncOpPartialLoweringPatterns are patterns which intend to match on FuncOps
/// and then perform their own walking of the IR. FuncOpPartialLoweringPatterns
/// have direct access to the ComponentLoweringState for the corresponding
/// component of the matched FuncOp.
class FuncOpPartialLoweringPattern : public PartialLoweringPattern<FuncOp> {
public:
  FuncOpPartialLoweringPattern(MLIRContext *context, LogicalResult &resRef,
                               FuncMapping &_funcMap, ProgramLoweringState &pls)
      : PartialLoweringPattern(context, resRef), funcMap(_funcMap), pls(pls) {}

  LogicalResult partiallyLower(FuncOp funcOp,
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
  PartiallyLowerFuncToComp(FuncOp funcOp, PatternRewriter &rewriter) const = 0;

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
  PartiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    /// We walk the operations of the funcOp to ensure that all def's have
    /// been visited before their uses.
    bool opBuiltSuccessfully = true;
    funcOp.walk([&](Operation *_op) {
      opBuiltSuccessfully &=
          TypeSwitch<mlir::Operation *, bool>(_op)
              .template Case<arith::ConstantOp, ReturnOp, BranchOpInterface,
                             /// SCF
                             scf::YieldOp,
                             /// memref
                             memref::AllocOp, memref::AllocaOp, memref::LoadOp,
                             memref::StoreOp,
                             /// standard arithmetic
                             AddIOp, SubIOp, CmpIOp, ShLIOp, ShRUIOp, ShRSIOp,
                             AndIOp, XOrIOp, OrIOp, ExtUIOp, ExtSIOp, TruncIOp,
                             MulIOp, DivUIOp, RemUIOp, IndexCastOp>(
                  [&](auto op) { return buildOp(rewriter, op).succeeded(); })
              .template Case<scf::WhileOp, FuncOp, scf::ConditionOp>([&](auto) {
                /// Skip: these special cases will be handled separately.
                return true;
              })
              .Default([&](auto op) {
                op->emitError() << "Unhandled operation during BuildOpGroups()";
                return false;
              });

      return opBuiltSuccessfully ? WalkResult::advance()
                                 : WalkResult::interrupt();
    });

    return success(opBuiltSuccessfully);
  }

private:
  /// Op builder specializations.
  LogicalResult buildOp(PatternRewriter &rewriter, scf::YieldOp yieldOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        BranchOpInterface brOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        arith::ConstantOp constOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AddIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, SubIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, MulIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, DivUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, RemUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShRUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShRSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShLIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AndIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, OrIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, XOrIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, CmpIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, TruncIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ExtUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ExtSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ReturnOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, IndexCastOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::AllocOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::AllocaOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::LoadOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::StoreOp op) const;

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
                                       op->getOperand(dstOp.index()));

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
    return calyx::createGroup<TGroupOp>(rewriter,
                                        getComponentState().getComponentOp(),
                                        op->getLoc(), groupName);
  }

  /// buildLibraryBinaryPipeOp will build a TCalyxLibBinaryPipeOp, to
  /// deal with MulIOp, DivUIOp and RemUIOp.
  template <typename TOpType, typename TSrcOp>
  LogicalResult buildLibraryBinaryPipeOp(PatternRewriter &rewriter, TSrcOp op,
                                         TOpType opPipe, Value out) const {
    StringRef opName = TSrcOp::getOperationName().split(".").second;
    Location loc = op.getLoc();
    Type width = op.getResult().getType();
    // Pass the result from the Operation to the Calyx primitive.
    op.getResult().replaceAllUsesWith(out);
    auto reg = createRegister(op.getLoc(), rewriter, *getComponent(),
                              width.getIntOrFloatBitWidth(),
                              getComponentState().getUniqueName(opName));
    // Operation pipelines are not combinational, so a GroupOp is required.
    auto group = createGroupForOp<calyx::GroupOp>(rewriter, op);
    getComponentState().addBlockScheduleable(op->getBlock(), group);

    rewriter.setInsertionPointToEnd(group.getBody());
    rewriter.create<calyx::AssignOp>(loc, opPipe.left(), op.getLhs());
    rewriter.create<calyx::AssignOp>(loc, opPipe.right(), op.getRhs());
    // Write the output to this register.
    rewriter.create<calyx::AssignOp>(loc, reg.in(), out);
    // The write enable port is high when the pipeline is done.
    rewriter.create<calyx::AssignOp>(loc, reg.write_en(), opPipe.done());
    rewriter.create<calyx::AssignOp>(
        loc, opPipe.go(), createConstant(loc, rewriter, *getComponent(), 1, 1));
    // The group is done when the register write is complete.
    rewriter.create<calyx::GroupDoneOp>(loc, reg.done());

    // Register the values for the pipeline.
    getComponentState().registerEvaluatingGroup(out, group);
    getComponentState().registerEvaluatingGroup(opPipe.left(), group);
    getComponentState().registerEvaluatingGroup(opPipe.right(), group);

    return success();
  }

  /// Creates assignments within the provided group to the address ports of the
  /// memoryOp based on the provided addressValues.
  void assignAddressPorts(PatternRewriter &rewriter, Location loc,
                          calyx::GroupInterface group,
                          calyx::MemoryInterface memoryInterface,
                          Operation::operand_range addressValues) const {
    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(group.getBody());
    auto addrPorts = memoryInterface.addrPorts();
    assert(addrPorts.size() == addressValues.size() &&
           "Mismatch between number of address ports of the provided memory "
           "and address assignment values");
    for (auto &address : enumerate(addressValues))
      rewriter.create<calyx::AssignOp>(loc, addrPorts[address.index()],
                                       address.value());
  }
};

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::LoadOp loadOp) const {
  Value memref = loadOp.memref();
  auto memoryInterface = getComponentState().getMemoryInterface(memref);
  if (calyx::noStoresToMemory(memref) && calyx::singleLoadFromMemory(memref)) {
    // Single load from memory; we do not need to write the
    // output to a register. This is essentially a "combinational read" under
    // current Calyx semantics with memory, and thus can be done in a
    // combinational group. Note that if any stores are done to this memory,
    // we require that the load and store be in separate non-combinational
    // groups to avoid reading and writing to the same memory in the same group.
    auto combGroup = createGroupForOp<calyx::CombGroupOp>(rewriter, loadOp);
    assignAddressPorts(rewriter, loadOp.getLoc(), combGroup, memoryInterface,
                       loadOp.getIndices());

    // We refrain from replacing the loadOp result with
    // memoryInterface.readData, since multiple loadOp's need to be converted
    // to a single memory's ReadData. If this replacement is done now, we lose
    // the link between which SSA memref::LoadOp values map to which groups for
    // loading a value from the Calyx memory. At this point of lowering, we
    // keep the memref::LoadOp SSA value, and do value replacement _after_
    // control has been generated (see LateSSAReplacement). This is *vital* for
    // things such as InlineCombGroups to be able to properly track which
    // memory assignment groups belong to which accesses.
    getComponentState().registerEvaluatingGroup(loadOp.getResult(), combGroup);
  } else {
    auto group = createGroupForOp<calyx::GroupOp>(rewriter, loadOp);
    assignAddressPorts(rewriter, loadOp.getLoc(), group, memoryInterface,
                       loadOp.getIndices());

    // Multiple loads from the same memory; In this case, we _may_ have a
    // structural hazard in the design we generate. To get around this, we
    // conservatively place a register in front of each load operation, and
    // replace all uses of the loaded value with the register output. Proper
    // handling of this requires the combinational group inliner/scheduler to
    // be aware of when a combinational expression references multiple loaded
    // values from the same memory, and then schedule assignments to temporary
    // registers to get around the structural hazard.
    auto reg = createRegister(loadOp.getLoc(), rewriter, *getComponent(),
                              loadOp.getMemRefType().getElementTypeBitWidth(),
                              getComponentState().getUniqueName("load"));
    buildAssignmentsForRegisterWrite(getComponentState(), rewriter, group, reg,
                                     memoryInterface.readData());
    loadOp.getResult().replaceAllUsesWith(reg.out());
    getComponentState().addBlockScheduleable(loadOp->getBlock(), group);
  }
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::StoreOp storeOp) const {
  auto memoryInterface =
      getComponentState().getMemoryInterface(storeOp.memref());
  auto group = createGroupForOp<calyx::GroupOp>(rewriter, storeOp);

  // This is a sequential group, so register it as being scheduleable for the
  // block.
  getComponentState().addBlockScheduleable(storeOp->getBlock(), group);
  assignAddressPorts(rewriter, storeOp.getLoc(), group, memoryInterface,
                     storeOp.getIndices());
  rewriter.setInsertionPointToEnd(group.getBody());
  rewriter.create<calyx::AssignOp>(
      storeOp.getLoc(), memoryInterface.writeData(), storeOp.getValueToStore());
  rewriter.create<calyx::AssignOp>(
      storeOp.getLoc(), memoryInterface.writeEn(),
      createConstant(storeOp.getLoc(), rewriter, *getComponent(), 1, 1));
  rewriter.create<calyx::GroupDoneOp>(storeOp.getLoc(), memoryInterface.done());

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     MulIOp mul) const {
  Location loc = mul.getLoc();
  Type width = mul.getResult().getType(), one = rewriter.getI1Type();
  auto mulPipe =
      getComponentState().getNewLibraryOpInstance<calyx::MultPipeLibOp>(
          rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::MultPipeLibOp>(rewriter, mul, mulPipe,
                                                        /*out=*/mulPipe.out());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     DivUIOp div) const {
  Location loc = div.getLoc();
  Type width = div.getResult().getType(), one = rewriter.getI1Type();
  auto divPipe =
      getComponentState().getNewLibraryOpInstance<calyx::DivPipeLibOp>(
          rewriter, loc, {one, one, one, width, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::DivPipeLibOp>(
      rewriter, div, divPipe,
      /*out=*/divPipe.out_quotient());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     RemUIOp rem) const {
  Location loc = rem.getLoc();
  Type width = rem.getResult().getType(), one = rewriter.getI1Type();
  auto remPipe =
      getComponentState().getNewLibraryOpInstance<calyx::DivPipeLibOp>(
          rewriter, loc, {one, one, one, width, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::DivPipeLibOp>(
      rewriter, rem, remPipe,
      /*out=*/remPipe.out_remainder());
}

template <typename TAllocOp>
static LogicalResult buildAllocOp(ComponentLoweringState &componentState,
                                  PatternRewriter &rewriter, TAllocOp allocOp) {
  rewriter.setInsertionPointToStart(componentState.getComponentOp().getBody());
  MemRefType memtype = allocOp.getType();
  SmallVector<int64_t> addrSizes;
  SmallVector<int64_t> sizes;
  for (int64_t dim : memtype.getShape()) {
    sizes.push_back(dim);
    addrSizes.push_back(calyx::handleZeroWidth(dim));
  }
  auto memoryOp = rewriter.create<calyx::MemoryOp>(
      allocOp.getLoc(), componentState.getUniqueName("mem"),
      memtype.getElementType().getIntOrFloatBitWidth(), sizes, addrSizes);
  // Externalize memories by default. This makes it easier for the native
  // compiler to provide initialized memories.
  memoryOp->setAttr("external",
                    IntegerAttr::get(rewriter.getI1Type(), llvm::APInt(1, 1)));
  componentState.registerMemoryInterface(allocOp.getResult(),
                                         calyx::MemoryInterface(memoryOp));
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::AllocOp allocOp) const {
  return buildAllocOp(getComponentState(), rewriter, allocOp);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::AllocaOp allocOp) const {
  return buildAllocOp(getComponentState(), rewriter, allocOp);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     scf::YieldOp yieldOp) const {
  if (yieldOp.getOperands().size() == 0)
    return success();
  auto whileOp = dyn_cast<scf::WhileOp>(yieldOp->getParentOp());
  assert(whileOp);
  ScfWhileOp whileOpInterface(whileOp);

  auto assignGroup = buildWhileIterArgAssignments(
      rewriter, getComponentState(), yieldOp.getLoc(), whileOpInterface,
      getComponentState().getUniqueName(whileOp) + "_latch",
      yieldOp->getOpOperands());
  getComponentState().setWhileLatchGroup(whileOpInterface, assignGroup);
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     BranchOpInterface brOp) const {
  /// Branch argument passing group creation
  /// Branch operands are passed through registers. In BuildBBRegs we
  /// created registers for all branch arguments of each block. We now
  /// create groups for assigning values to these registers.
  Block *srcBlock = brOp->getBlock();
  for (auto succBlock : enumerate(brOp->getSuccessors())) {
    auto succOperands = brOp.getSuccessorOperands(succBlock.index());
    if (succOperands.empty())
      continue;
    // Create operand passing group
    std::string groupName = progState().blockName(srcBlock) + "_to_" +
                            progState().blockName(succBlock.value());
    auto groupOp = calyx::createGroup<calyx::GroupOp>(rewriter, *getComponent(),
                                                      brOp.getLoc(), groupName);
    // Fetch block argument registers associated with the basic block
    auto dstBlockArgRegs =
        getComponentState().getBlockArgRegs(succBlock.value());
    // Create register assignment for each block argument
    for (auto arg : enumerate(succOperands.getForwardedOperands())) {
      auto reg = dstBlockArgRegs[arg.index()];
      buildAssignmentsForRegisterWrite(getComponentState(), rewriter, groupOp,
                                       reg, arg.value());
    }
    /// Register the group as a block argument group, to be executed
    /// when entering the successor block from this block (srcBlock).
    getComponentState().addBlockArgGroup(srcBlock, succBlock.value(), groupOp);
  }
  return success();
}

/// For each return statement, we create a new group for assigning to the
/// previously created return value registers.
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ReturnOp retOp) const {
  if (retOp.getNumOperands() == 0)
    return success();

  std::string groupName = getComponentState().getUniqueName("ret_assign");
  auto groupOp = calyx::createGroup<calyx::GroupOp>(rewriter, *getComponent(),
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
                                     arith::ConstantOp constOp) const {
  /// Move constant operations to the compOp body as hw::ConstantOp's.
  APInt value;
  calyx::matchConstantOp(constOp, value);
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
                                     ShRUIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::RshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShRSIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SrshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShLIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::LshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AndIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::AndLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     OrIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::OrLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     XOrIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::XorLibOp>(rewriter, op);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     CmpIOp op) const {
  switch (op.getPredicate()) {
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
  }
  llvm_unreachable("unsupported comparison predicate");
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     TruncIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SliceLibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ExtUIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::PadLibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ExtSIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::ExtSILibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     IndexCastOp op) const {
  Type sourceType = calyx::convIndexType(rewriter, op.getOperand().getType());
  Type targetType = calyx::convIndexType(rewriter, op.getResult().getType());
  unsigned targetBits = targetType.getIntOrFloatBitWidth();
  unsigned sourceBits = sourceType.getIntOrFloatBitWidth();
  LogicalResult res = success();

  if (targetBits == sourceBits) {
    /// Drop the index cast and replace uses of the target value with the source
    /// value.
    op.getResult().replaceAllUsesWith(op.getOperand());
  } else {
    /// pad/slice the source operand.
    if (sourceBits > targetBits)
      res = buildLibraryOp<calyx::CombGroupOp, calyx::SliceLibOp>(
          rewriter, op, {sourceType}, {targetType});
    else
      res = buildLibraryOp<calyx::CombGroupOp, calyx::PadLibOp>(
          rewriter, op, {sourceType}, {targetType});
  }
  rewriter.eraseOp(op);
  return res;
}

/// This pass rewrites memory accesses that have a width mismatch. Such
/// mismatches are due to index types being assumed 32-bit wide due to the lack
/// of a width inference pass.
class RewriteMemoryAccesses : public PartialLoweringPattern<calyx::AssignOp> {
public:
  RewriteMemoryAccesses(MLIRContext *context, LogicalResult &resRef,
                        ProgramLoweringState &pls)
      : PartialLoweringPattern(context, resRef), pls(pls) {}

  LogicalResult partiallyLower(calyx::AssignOp assignOp,
                               PatternRewriter &rewriter) const override {
    auto &state =
        pls.compLoweringState(assignOp->getParentOfType<calyx::ComponentOp>());

    auto dest = assignOp.dest();
    if (!state.isInputPortOfMemory(dest).hasValue())
      return success();

    auto src = assignOp.src();
    unsigned srcBits = src.getType().getIntOrFloatBitWidth();
    unsigned dstBits = dest.getType().getIntOrFloatBitWidth();
    if (srcBits == dstBits)
      return success();

    SmallVector<Type> types = {rewriter.getIntegerType(srcBits),
                               rewriter.getIntegerType(dstBits)};
    Operation *newOp;
    if (srcBits > dstBits) {
      newOp = state.getNewLibraryOpInstance<calyx::SliceLibOp>(
          rewriter, assignOp.getLoc(), types);
    } else {
      newOp = state.getNewLibraryOpInstance<calyx::PadLibOp>(
          rewriter, assignOp.getLoc(), types);
    }
    rewriter.setInsertionPoint(assignOp->getBlock(),
                               assignOp->getBlock()->begin());
    rewriter.create<calyx::AssignOp>(assignOp->getLoc(), newOp->getResult(0),
                                     src);
    assignOp.setOperand(1, newOp->getResult(1));

    return success();
  }

private:
  ProgramLoweringState &pls;
};

/// Connverts all index-typed operations and values to i32 values.
class ConvertIndexTypes : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    funcOp.walk([&](Block *block) {
      for (auto arg : block->getArguments())
        arg.setType(calyx::convIndexType(rewriter, arg.getType()));
    });

    funcOp.walk([&](Operation *op) {
      for (auto res : op->getResults()) {
        auto resType = res.getType();
        if (!resType.isIndex())
          continue;

        res.setType(calyx::convIndexType(rewriter, resType));
        if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
          APInt value;
          calyx::matchConstantOp(constOp, value);
          rewriter.setInsertionPoint(constOp);
          rewriter.replaceOpWithNewOp<arith::ConstantOp>(
              constOp, rewriter.getI32IntegerAttr(value.getSExtValue()));
        }
      }
    });
    return success();
  }
};

/// Inlines Calyx ExecuteRegionOp operations within their parent blocks.
/// An execution region op (ERO) is inlined by:
///  i  : add a sink basic block for all yield operations inside the
///       ERO to jump to
///  ii : Rewrite scf.yield calls inside the ERO to branch to the sink block
///  iii: inline the ERO region
/// TODO(#1850) evaluate the usefulness of this lowering pattern.
class InlineExecuteRegionOpPattern
    : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp execOp,
                                PatternRewriter &rewriter) const override {
    /// Determine type of "yield" operations inside the ERO.
    TypeRange yieldTypes = execOp.getResultTypes();

    /// Create sink basic block and rewrite uses of yield results to sink block
    /// arguments.
    rewriter.setInsertionPointAfter(execOp);
    auto *sinkBlock = rewriter.splitBlock(
        execOp->getBlock(),
        execOp.getOperation()->getIterator()->getNextNode()->getIterator());
    sinkBlock->addArguments(
        yieldTypes,
        SmallVector<Location, 4>(yieldTypes.size(), rewriter.getUnknownLoc()));
    for (auto res : enumerate(execOp.getResults()))
      res.value().replaceAllUsesWith(sinkBlock->getArgument(res.index()));

    /// Rewrite yield calls as branches.
    for (auto yieldOp :
         make_early_inc_range(execOp.getRegion().getOps<scf::YieldOp>())) {
      rewriter.setInsertionPointAfter(yieldOp);
      rewriter.replaceOpWithNewOp<BranchOp>(yieldOp, sinkBlock,
                                            yieldOp.getOperands());
    }

    /// Inline the regionOp.
    auto *preBlock = execOp->getBlock();
    auto *execOpEntryBlock = &execOp.getRegion().front();
    auto *postBlock = execOp->getBlock()->splitBlock(execOp);
    rewriter.inlineRegionBefore(execOp.getRegion(), postBlock);
    rewriter.mergeBlocks(postBlock, preBlock);
    rewriter.eraseOp(execOp);

    /// Finally, erase the unused entry block of the execOp region.
    rewriter.mergeBlocks(execOpEntryBlock, preBlock);

    return success();
  }
};

static void
appendPortsForExternalMemref(PatternRewriter &rewriter, StringRef memName,
                             Value memref, unsigned memoryID,
                             SmallVectorImpl<calyx::PortInfo> &inPorts,
                             SmallVectorImpl<calyx::PortInfo> &outPorts) {
  MemRefType memrefType = memref.getType().cast<MemRefType>();

  /// Ports constituting a memory interface are added a set of attributes under
  /// a "mem : {...}" dictionary. These attributes allows for deducing which
  /// top-level I/O signals constitutes a unique memory interface.
  auto getMemoryInterfaceAttr = [&](StringRef tag,
                                    Optional<unsigned> addrIdx = {}) {
    auto attrs = SmallVector<NamedAttribute>{
        /// "id" denotes a unique memory interface.
        rewriter.getNamedAttr("id", rewriter.getI32IntegerAttr(memoryID)),
        /// "tag" denotes the function of this signal.
        rewriter.getNamedAttr("tag", rewriter.getStringAttr(tag))};
    if (addrIdx.hasValue())
      /// "addr_idx" denotes the address index of this signal, for
      /// multi-dimensional memory interfaces.
      attrs.push_back(rewriter.getNamedAttr(
          "addr_idx", rewriter.getI32IntegerAttr(addrIdx.getValue())));

    return rewriter.getNamedAttr("mem", rewriter.getDictionaryAttr(attrs));
  };

  /// Read data
  inPorts.push_back(calyx::PortInfo{
      rewriter.getStringAttr(memName + "_read_data"),
      memrefType.getElementType(), calyx::Direction::Input,
      DictionaryAttr::get(rewriter.getContext(),
                          {getMemoryInterfaceAttr("read_data")})});

  /// Done
  inPorts.push_back(
      calyx::PortInfo{rewriter.getStringAttr(memName + "_done"),
                      rewriter.getI1Type(), calyx::Direction::Input,
                      DictionaryAttr::get(rewriter.getContext(),
                                          {getMemoryInterfaceAttr("done")})});

  /// Write data
  outPorts.push_back(calyx::PortInfo{
      rewriter.getStringAttr(memName + "_write_data"),
      memrefType.getElementType(), calyx::Direction::Output,
      DictionaryAttr::get(rewriter.getContext(),
                          {getMemoryInterfaceAttr("write_data")})});

  /// Memory address outputs
  for (auto dim : enumerate(memrefType.getShape())) {
    outPorts.push_back(calyx::PortInfo{
        rewriter.getStringAttr(memName + "_addr" + std::to_string(dim.index())),
        rewriter.getIntegerType(calyx::handleZeroWidth(dim.value())),
        calyx::Direction::Output,
        DictionaryAttr::get(rewriter.getContext(),
                            {getMemoryInterfaceAttr("addr", dim.index())})});
  }

  /// Write enable
  outPorts.push_back(calyx::PortInfo{
      rewriter.getStringAttr(memName + "_write_en"), rewriter.getI1Type(),
      calyx::Direction::Output,
      DictionaryAttr::get(rewriter.getContext(),
                          {getMemoryInterfaceAttr("write_en")})});
}

/// Creates a new Calyx component for each FuncOp in the program.
struct FuncOpConversion : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    /// Maintain a mapping between funcOp input arguments and the port index
    /// which the argument will eventually map to.
    DenseMap<Value, unsigned> funcOpArgRewrites;

    /// Maintain a mapping between funcOp output indexes and the component
    /// output port index which the return value will eventually map to.
    DenseMap<unsigned, unsigned> funcOpResultMapping;

    /// Maintain a mapping between an external memory argument (identified by a
    /// memref) and eventual component input- and output port indices that will
    /// map to the memory ports. The pair denotes the start index of the memory
    /// ports in the in- and output ports of the component. Ports are expected
    /// to be ordered in the same manner as they are added by
    /// appendPortsForExternalMemref.
    DenseMap<Value, std::pair<unsigned, unsigned>> extMemoryCompPortIndices;

    /// Create I/O ports. Maintain separate in/out port vectors to determine
    /// which port index each function argument will eventually map to.
    SmallVector<calyx::PortInfo> inPorts, outPorts;
    FunctionType funcType = funcOp.getFunctionType();
    unsigned extMemCounter = 0;
    for (auto &arg : enumerate(funcOp.getArguments())) {
      if (arg.value().getType().isa<MemRefType>()) {
        /// External memories
        auto memName =
            "ext_mem" + std::to_string(extMemoryCompPortIndices.size());
        extMemoryCompPortIndices[arg.value()] = {inPorts.size(),
                                                 outPorts.size()};
        appendPortsForExternalMemref(rewriter, memName, arg.value(),
                                     extMemCounter++, inPorts, outPorts);
      } else {
        /// Single-port arguments
        auto inName = "in" + std::to_string(arg.index());
        funcOpArgRewrites[arg.value()] = inPorts.size();
        inPorts.push_back(calyx::PortInfo{
            rewriter.getStringAttr(inName),
            calyx::convIndexType(rewriter, arg.value().getType()),
            calyx::Direction::Input,
            DictionaryAttr::get(rewriter.getContext(), {})});
      }
    }
    for (auto &res : enumerate(funcType.getResults())) {
      funcOpResultMapping[res.index()] = outPorts.size();
      outPorts.push_back(calyx::PortInfo{
          rewriter.getStringAttr("out" + std::to_string(res.index())),
          calyx::convIndexType(rewriter, res.value()), calyx::Direction::Output,
          DictionaryAttr::get(rewriter.getContext(), {})});
    }

    /// We've now recorded all necessary indices. Merge in- and output ports
    /// and add the required mandatory component ports.
    auto ports = inPorts;
    llvm::append_range(ports, outPorts);
    calyx::addMandatoryComponentPorts(rewriter, ports);

    /// Create a calyx::ComponentOp corresponding to the to-be-lowered function.
    auto compOp = rewriter.create<calyx::ComponentOp>(
        funcOp.getLoc(), rewriter.getStringAttr(funcOp.getSymName()), ports);

    /// Mark this component as the toplevel.
    compOp->setAttr("toplevel", rewriter.getUnitAttr());

    /// Store the function-to-component mapping.
    funcMap[funcOp] = compOp;
    auto &compState = progState().compLoweringState(compOp);
    compState.setFuncOpResultMapping(funcOpResultMapping);

    /// Rewrite funcOp SSA argument values to the CompOp arguments.
    for (auto &mapping : funcOpArgRewrites)
      mapping.getFirst().replaceAllUsesWith(
          compOp.getArgument(mapping.getSecond()));

    /// Register external memories
    for (auto extMemPortIndices : extMemoryCompPortIndices) {
      /// Create a mapping for the in- and output ports using the Calyx memory
      /// port structure.
      calyx::MemoryPortsImpl extMemPorts;
      unsigned inPortsIt = extMemPortIndices.getSecond().first;
      unsigned outPortsIt = extMemPortIndices.getSecond().second +
                            compOp.getInputPortInfo().size();
      extMemPorts.readData = compOp.getArgument(inPortsIt++);
      extMemPorts.done = compOp.getArgument(inPortsIt);
      extMemPorts.writeData = compOp.getArgument(outPortsIt++);
      unsigned nAddresses = extMemPortIndices.getFirst()
                                .getType()
                                .cast<MemRefType>()
                                .getShape()
                                .size();
      for (unsigned j = 0; j < nAddresses; ++j)
        extMemPorts.addrPorts.push_back(compOp.getArgument(outPortsIt++));
      extMemPorts.writeEn = compOp.getArgument(outPortsIt);

      /// Register the external memory ports as a memory interface within the
      /// component.
      compState.registerMemoryInterface(extMemPortIndices.getFirst(),
                                        calyx::MemoryInterface(extMemPorts));
    }

    return success();
  }
};

/// In BuildWhileGroups, a register is created for each iteration argumenet of
/// the while op. These registers are then written to on the while op
/// terminating yield operation alongside before executing the whileOp in the
/// schedule, to set the initial values of the argument registers.
class BuildWhileGroups : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    LogicalResult res = success();
    funcOp.walk([&](Operation *op) {
      // Only work on ops that support the ScfWhileOp.
      if (!isa<scf::WhileOp>(op))
        return WalkResult::advance();

      auto scfWhileOp = cast<scf::WhileOp>(op);
      ScfWhileOp whileOp(scfWhileOp);

      getComponentState().setUniqueName(whileOp.getOperation(), "while");

      /// Check for do-while loops.
      /// TODO(mortbopet) can we support these? for now, do not support loops
      /// where iterargs are changed in the 'before' region. scf.WhileOp also
      /// has support for different types of iter_args and return args which we
      /// also do not support; iter_args and while return values are placed in
      /// the same registers.
      for (auto barg :
           enumerate(scfWhileOp.getBefore().front().getArguments())) {
        auto condOp = scfWhileOp.getConditionOp().getArgs()[barg.index()];
        if (barg.value() != condOp) {
          res = whileOp.getOperation()->emitError()
                << progState().irName(barg.value())
                << " != " << progState().irName(condOp)
                << "do-while loops not supported; expected iter-args to "
                   "remain untransformed in the 'before' region of the "
                   "scf.while op.";
          return WalkResult::interrupt();
        }
      }

      /// Create iteration argument registers.
      /// The iteration argument registers will be referenced:
      /// - In the "before" part of the while loop, calculating the conditional,
      /// - In the "after" part of the while loop,
      /// - Outside the while loop, rewriting the while loop return values.
      for (auto arg : enumerate(whileOp.getBodyArgs())) {
        std::string name =
            getComponentState().getUniqueName(whileOp.getOperation()).str() +
            "_arg" + std::to_string(arg.index());
        auto reg =
            createRegister(arg.value().getLoc(), rewriter, *getComponent(),
                           arg.value().getType().getIntOrFloatBitWidth(), name);
        getComponentState().addWhileIterReg(whileOp, reg, arg.index());
        arg.value().replaceAllUsesWith(reg.out());

        /// Also replace uses in the "before" region of the while loop
        whileOp.getConditionBlock()
            ->getArgument(arg.index())
            .replaceAllUsesWith(reg.out());
      }

      /// Create iter args initial value assignment group(s), one per register.
      SmallVector<calyx::GroupOp> initGroups;
      auto numOperands = whileOp.getOperation()->getNumOperands();
      for (size_t i = 0; i < numOperands; ++i) {
        auto initGroupOp = buildWhileIterArgAssignments(
            rewriter, getComponentState(), whileOp.getOperation()->getLoc(),
            whileOp,
            getComponentState().getUniqueName(whileOp.getOperation()) +
                "_init_" + std::to_string(i),
            whileOp.getOperation()->getOpOperand(i));
        initGroups.push_back(initGroupOp);
      }

      getComponentState().addBlockScheduleable(
          whileOp.getOperation()->getBlock(),
          WhileScheduleable{{whileOp, initGroups}});
      return WalkResult::advance();
    });
    return res;
  }
};

/// Builds registers for each block argument in the program.
class BuildBBRegs : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    funcOp.walk([&](Block *block) {
      /// Do not register component input values.
      if (block == &block->getParent()->front())
        return;

      for (auto arg : enumerate(block->getArguments())) {
        Type argType = arg.value().getType();
        assert(argType.isa<IntegerType>() && "unsupported block argument type");
        unsigned width = argType.getIntOrFloatBitWidth();
        std::string name =
            progState().blockName(block) + "_arg" + std::to_string(arg.index());
        auto reg = createRegister(arg.value().getLoc(), rewriter,
                                  *getComponent(), width, name);
        getComponentState().addBlockArgReg(block, reg, arg.index());
        arg.value().replaceAllUsesWith(reg.out());
      }
    });
    return success();
  }
};

/// Builds registers for the return statement of the program and constant
/// assignments to the component return value.
class BuildReturnRegs : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {

    for (auto argType : enumerate(funcOp.getResultTypes())) {
      auto convArgType = calyx::convIndexType(rewriter, argType.value());
      assert(convArgType.isa<IntegerType>() && "unsupported return type");
      unsigned width = convArgType.getIntOrFloatBitWidth();
      std::string name = "ret_arg" + std::to_string(argType.index());
      auto reg = createRegister(funcOp.getLoc(), rewriter, *getComponent(),
                                width, name);
      getComponentState().addReturnReg(reg, argType.index());

      rewriter.setInsertionPointToStart(getComponent()->getWiresOp().getBody());
      rewriter.create<calyx::AssignOp>(
          funcOp->getLoc(),
          calyx::getComponentOutput(
              *getComponent(),
              getComponentState().getFuncOpResultMapping(argType.index())),
          reg.out());
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
  PartiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    auto *entryBlock = &funcOp.getBlocks().front();
    rewriter.setInsertionPointToStart(getComponent()->getControlOp().getBody());
    auto topLevelSeqOp = rewriter.create<calyx::SeqOp>(funcOp.getLoc());
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
      auto seqOp = rewriter.create<calyx::SeqOp>(loc);
      parentCtrlBlock = seqOp.getBody();
    }

    for (auto &group : compBlockScheduleables) {
      rewriter.setInsertionPointToEnd(parentCtrlBlock);
      if (auto groupPtr = std::get_if<calyx::GroupOp>(&group); groupPtr) {
        rewriter.create<calyx::EnableOp>(groupPtr->getLoc(),
                                         groupPtr->sym_name());
      } else if (auto whileSchedPtr = std::get_if<WhileScheduleable>(&group);
                 whileSchedPtr) {
        auto &whileOp = whileSchedPtr->whileOp;

        auto whileCtrlOp =
            buildWhileCtrlOp(whileOp, whileSchedPtr->initGroups, rewriter);
        rewriter.setInsertionPointToEnd(whileCtrlOp.getBody());
        auto whileBodyOp =
            rewriter.create<calyx::SeqOp>(whileOp.getOperation()->getLoc());
        auto *whileBodyOpBlock = whileBodyOp.getBody();

        /// Only schedule the 'after' block. The 'before' block is
        /// implicitly scheduled when evaluating the while condition.
        LogicalResult res = buildCFGControl(path, rewriter, whileBodyOpBlock,
                                            block, whileOp.getBodyBlock());

        // Insert loop-latch at the end of the while group
        rewriter.setInsertionPointToEnd(whileBodyOpBlock);
        calyx::GroupOp whileLatchGroup =
            getComponentState().getWhileLatchGroup(whileOp);
        rewriter.create<calyx::EnableOp>(whileLatchGroup.getLoc(),
                                         whileLatchGroup.getName());

        if (res.failed())
          return res;
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
    /// Schedule any registered block arguments to be executed before the body
    /// of the branch.
    rewriter.setInsertionPointToEnd(parentCtrlBlock);
    auto preSeqOp = rewriter.create<calyx::SeqOp>(loc);
    rewriter.setInsertionPointToEnd(preSeqOp.getBody());
    for (auto barg : getComponentState().getBlockArgGroups(from, to))
      rewriter.create<calyx::EnableOp>(barg.getLoc(), barg.sym_name());

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
        /// TODO(mortbopet): we could choose to support ie. std.switch, but it
        /// would probably be easier to just require it to be lowered
        /// beforehand.
        assert(nSuccessors == 2 &&
               "only conditional branches supported for now...");
        /// Wrap each branch inside an if/else.
        auto cond = brOp->getOperand(0);
        auto condGroup =
            getComponentState().getEvaluatingGroup<calyx::CombGroupOp>(cond);
        auto symbolAttr = FlatSymbolRefAttr::get(
            StringAttr::get(getContext(), condGroup.sym_name()));

        auto ifOp = rewriter.create<calyx::IfOp>(
            brOp->getLoc(), cond, symbolAttr, /*initializeElseBody=*/true);
        rewriter.setInsertionPointToStart(ifOp.getThenBody());
        auto thenSeqOp = rewriter.create<calyx::SeqOp>(brOp.getLoc());
        rewriter.setInsertionPointToStart(ifOp.getElseBody());
        auto elseSeqOp = rewriter.create<calyx::SeqOp>(brOp.getLoc());

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

  calyx::WhileOp buildWhileCtrlOp(ScfWhileOp whileOp,
                                  SmallVector<calyx::GroupOp> initGroups,
                                  PatternRewriter &rewriter) const {
    Location loc = whileOp.getLoc();
    /// Insert while iter arg initialization group(s). Emit a
    /// parallel group to assign one or more registers all at once.
    {
      PatternRewriter::InsertionGuard g(rewriter);
      auto parOp = rewriter.create<calyx::ParOp>(loc);
      rewriter.setInsertionPointToStart(parOp.getBody());
      for (calyx::GroupOp group : initGroups)
        rewriter.create<calyx::EnableOp>(group.getLoc(), group.getName());
    }

    /// Insert the while op itself.
    auto cond = whileOp.getConditionValue();
    auto condGroup =
        getComponentState().getEvaluatingGroup<calyx::CombGroupOp>(cond);
    auto symbolAttr = FlatSymbolRefAttr::get(
        StringAttr::get(getContext(), condGroup.sym_name()));
    return rewriter.create<calyx::WhileOp>(loc, cond, symbolAttr);
  }
};

/// This pass recursively inlines use-def chains of combinational logic (from
/// non-stateful groups) into groups referenced in the control schedule.
class InlineCombGroups
    : public PartialLoweringPattern<calyx::GroupInterface,
                                    OpInterfaceRewritePattern> {
public:
  InlineCombGroups(MLIRContext *context, LogicalResult &resRef,
                   ProgramLoweringState &pls)
      : PartialLoweringPattern(context, resRef), pls(pls) {}

  LogicalResult partiallyLower(calyx::GroupInterface originGroup,
                               PatternRewriter &rewriter) const override {
    auto &state = pls.compLoweringState(
        originGroup->getParentOfType<calyx::ComponentOp>());

    /// Filter groups which are not part of the control schedule.
    if (SymbolTable::symbolKnownUseEmpty(originGroup.symName(),
                                         state.getComponentOp().getControlOp()))
      return success();

    /// Maintain a set of the groups which we've inlined so far. The group
    /// itself is implicitly inlined.
    llvm::SmallSetVector<Operation *, 8> inlinedGroups;
    inlinedGroups.insert(originGroup);

    /// Starting from the matched originGroup, we traverse use-def chains of
    /// combinational logic, and inline assignments from the defining
    /// combinational groups.
    recurseInlineCombGroups(
        rewriter, state, inlinedGroups, originGroup, originGroup,
        /*disable inlining of the originGroup itself*/ false);
    return success();
  }

private:
  void
  recurseInlineCombGroups(PatternRewriter &rewriter,
                          ComponentLoweringState &state,
                          llvm::SmallSetVector<Operation *, 8> &inlinedGroups,
                          calyx::GroupInterface originGroup,
                          calyx::GroupInterface recGroup, bool doInline) const {
    inlinedGroups.insert(recGroup);
    for (auto assignOp : recGroup.getBody()->getOps<calyx::AssignOp>()) {
      if (doInline) {
        /// Inline the assignment into the originGroup.
        auto clonedAssignOp = rewriter.clone(*assignOp.getOperation());
        clonedAssignOp->moveBefore(originGroup.getBody(),
                                   originGroup.getBody()->end());
      }
      Value src = assignOp.src();

      /// Things which stop recursive inlining (or in other words, what
      /// breaks combinational paths).
      /// - Component inputs
      /// - Register and memory reads
      /// - Constant ops (constant ops are not evaluated by any group)
      /// - Multiplication pipelines are sequential.
      /// - 'While' return values (these are registers, however, 'while'
      ///   return values have at the current point of conversion not yet
      ///   been rewritten to their register outputs, see comment in
      ///   LateSSAReplacement)
      if (src.isa<BlockArgument>() ||
          isa<calyx::RegisterOp, calyx::MemoryOp, hw::ConstantOp,
              arith::ConstantOp, calyx::MultPipeLibOp, calyx::DivPipeLibOp,
              scf::WhileOp>(src.getDefiningOp()))
        continue;

      auto srcCombGroup = dyn_cast<calyx::CombGroupOp>(
          state.getEvaluatingGroup(src).getOperation());
      if (!srcCombGroup)
        continue;
      if (inlinedGroups.count(srcCombGroup))
        continue;

      recurseInlineCombGroups(rewriter, state, inlinedGroups, originGroup,
                              srcCombGroup, true);
    }
  }

  ProgramLoweringState &pls;
};

/// LateSSAReplacement contains various functions for replacing SSA values that
/// were not replaced during op construction.
class LateSSAReplacement : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult PartiallyLowerFuncToComp(FuncOp funcOp,
                                         PatternRewriter &) const override {
    funcOp.walk([&](scf::WhileOp op) {
      /// The yielded values returned from the while op will be present in the
      /// iterargs registers post execution of the loop.
      /// This is done now, as opposed to during BuildWhileGroups since if the
      /// results of the whileOp were replaced before
      /// BuildOpGroups/BuildControl, the whileOp would get dead-code
      /// eliminated.
      ScfWhileOp whileOp(op);
      for (auto res : getComponentState().getWhileIterRegs(whileOp))
        whileOp.getOperation()->getResults()[res.first].replaceAllUsesWith(
            res.second.out());
    });

    funcOp.walk([&](memref::LoadOp loadOp) {
      if (calyx::singleLoadFromMemory(loadOp)) {
        /// In buildOpGroups we did not replace loadOp's results, to ensure a
        /// link between evaluating groups (which fix the input addresses of a
        /// memory op) and a readData result. Now, we may replace these SSA
        /// values with their memoryOp readData output.
        loadOp.getResult().replaceAllUsesWith(
            getComponentState().getMemoryInterface(loadOp.memref()).readData());
      }
    });

    return success();
  }
};

/// Erases FuncOp operations.
class CleanupFuncOps : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    rewriter.eraseOp(funcOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Simplification patterns
//===----------------------------------------------------------------------===//

/// Removes calyx::CombGroupOps which are unused. These correspond to
/// combinational groups created during op building that, after conversion,
/// have either been inlined into calyx::GroupOps or are referenced by an
/// if/while with statement.
/// We do not eliminate unused calyx::GroupOps; this should never happen, and is
/// considered an error. In these cases, the program will be invalidated when
/// the Calyx verifiers execute.
struct EliminateUnusedCombGroups : mlir::OpRewritePattern<calyx::CombGroupOp> {
  using mlir::OpRewritePattern<calyx::CombGroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(calyx::CombGroupOp combGroupOp,
                                PatternRewriter &rewriter) const override {
    auto control =
        combGroupOp->getParentOfType<calyx::ComponentOp>().getControlOp();
    if (!SymbolTable::symbolKnownUseEmpty(combGroupOp.sym_nameAttr(), control))
      return failure();

    rewriter.eraseOp(combGroupOp);
    return success();
  }
};

/// GroupDoneOp's are terminator operations and should therefore be the last
/// operator in a group. During group construction, we always append assignments
/// to the end of a group, resulting in group_done ops migrating away from the
/// terminator position. This pattern moves such ops to the end of their group.
struct NonTerminatingGroupDonePattern
    : mlir::OpRewritePattern<calyx::GroupDoneOp> {
  using mlir::OpRewritePattern<calyx::GroupDoneOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(calyx::GroupDoneOp groupDoneOp,
                                PatternRewriter & /*rewriter*/) const override {
    Block *block = groupDoneOp->getBlock();
    if (&block->back() == groupDoneOp)
      return failure();

    groupDoneOp->moveBefore(groupDoneOp->getBlock(),
                            groupDoneOp->getBlock()->end());
    return success();
  }
};

/// When building groups which contain accesses to multiple sequential
/// components, a group_done op is created for each of these. This pattern
/// and's each of the group_done values into a single group_done.
struct MultipleGroupDonePattern : mlir::OpRewritePattern<calyx::GroupOp> {
  using mlir::OpRewritePattern<calyx::GroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(calyx::GroupOp groupOp,
                                PatternRewriter &rewriter) const override {
    auto groupDoneOps = SmallVector<calyx::GroupDoneOp>(
        groupOp.getBody()->getOps<calyx::GroupDoneOp>());

    if (groupDoneOps.size() <= 1)
      return failure();

    /// 'and' all of the calyx::GroupDoneOp's.
    rewriter.setInsertionPointToEnd(groupDoneOps[0]->getBlock());
    SmallVector<Value> doneOpSrcs;
    llvm::transform(groupDoneOps, std::back_inserter(doneOpSrcs),
                    [](calyx::GroupDoneOp op) { return op.src(); });
    Value allDone =
        rewriter.create<comb::AndOp>(groupDoneOps.front().getLoc(), doneOpSrcs);

    /// Create a group done op with the complex expression as a guard.
    rewriter.create<calyx::GroupDoneOp>(
        groupOp.getLoc(),
        rewriter.create<hw::ConstantOp>(groupOp.getLoc(), APInt(1, 1)),
        allDone);
    for (auto groupDoneOp : groupDoneOps)
      rewriter.eraseOp(groupDoneOp);

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
      auto funcOps = moduleOp.getOps<FuncOp>();
      if (std::distance(funcOps.begin(), funcOps.end()) == 1)
        topLevelFunction = (*funcOps.begin()).getSymName().str();
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
    target.addIllegalDialect<FuncDialect>();
    target.addIllegalDialect<ArithmeticDialect>();
    target.addLegalOp<AddIOp, SubIOp, CmpIOp, ShLIOp, ShRUIOp, ShRSIOp, AndIOp,
                      XOrIOp, OrIOp, ExtUIOp, TruncIOp, CondBranchOp, BranchOp,
                      MulIOp, DivUIOp, RemUIOp, ReturnOp, arith::ConstantOp,
                      IndexCastOp, FuncOp, ExtSIOp>();

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
    assert(pattern.getNativePatterns().size() == 1 &&
           "Should only apply 1 partial lowering pattern at once");

    // During component creation, the function body is inlined into the
    // component body for further processing. However, proper control flow
    // will only be established later in the conversion process, so ensure
    // that rewriter optimizations (especially DCE) are disabled.
    GreedyRewriteConfig config;
    config.enableRegionSimplification = false;
    if (runOnce)
      config.maxIterations = 0;

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

  /// This pass inlines scf.ExecuteRegionOp's by adding control-flow.
  addGreedyPattern<InlineExecuteRegionOpPattern>(loweringPatterns);

  /// This pattern converts all index typed values to an i32 integer.
  addOncePattern<ConvertIndexTypes>(loweringPatterns, funcMap, *loweringState);

  /// This pattern creates registers for all basic-block arguments.
  addOncePattern<BuildBBRegs>(loweringPatterns, funcMap, *loweringState);

  /// This pattern creates registers for the function return values.
  addOncePattern<BuildReturnRegs>(loweringPatterns, funcMap, *loweringState);

  /// This pattern creates registers for iteration arguments of scf.while
  /// operations. Additionally, creates a group for assigning the initial
  /// value of the iteration argument registers.
  addOncePattern<BuildWhileGroups>(loweringPatterns, funcMap, *loweringState);

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

  /// This pass recursively inlines use-def chains of combinational logic (from
  /// non-stateful groups) into groups referenced in the control schedule.
  addOncePattern<InlineCombGroups>(loweringPatterns, *loweringState);

  /// This pattern performs various SSA replacements that must be done
  /// after control generation.
  addOncePattern<LateSSAReplacement>(loweringPatterns, funcMap, *loweringState);

  /// Eliminate any unused combinational groups. This is done before
  /// RewriteMemoryAccesses to avoid inferring slice components for groups that
  /// will be removed.
  addGreedyPattern<EliminateUnusedCombGroups>(loweringPatterns);

  /// This pattern rewrites accesses to memories which are too wide due to
  /// index types being converted to a fixed-width integer type.
  addOncePattern<RewriteMemoryAccesses>(loweringPatterns, *loweringState);

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

  //===----------------------------------------------------------------------===//
  // Cleanup patterns
  //===----------------------------------------------------------------------===//
  RewritePatternSet cleanupPatterns(&getContext());
  cleanupPatterns.add<MultipleGroupDonePattern, NonTerminatingGroupDonePattern>(
      &getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(cleanupPatterns)))) {
    signalPassFailure();
    return;
  }

  if (ciderSourceLocationMetadata) {
    // Debugging information for the Cider debugger.
    // Reference: https://docs.calyxir.org/debug/cider.html
    SmallVector<Attribute, 16> sourceLocations;
    getOperation()->walk([&](calyx::ComponentOp component) {
      return getCiderSourceLocationMetadata(component, sourceLocations);
    });

    MLIRContext *context = getOperation()->getContext();
    getOperation()->setAttr("calyx.metadata",
                            ArrayAttr::get(context, sourceLocations));
  }
}

} // namespace scftocalyx

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createSCFToCalyxPass() {
  return std::make_unique<scftocalyx::SCFToCalyxPass>();
}

} // namespace circt
