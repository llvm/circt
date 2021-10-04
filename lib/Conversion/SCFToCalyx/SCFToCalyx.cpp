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
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

struct WhileScheduleable {
  /// While operation to schedule.
  scf::WhileOp whileOp;
  /// The group to schedule before the while operation This group should set the
  /// initial values of the loop init_args registers.
  calyx::GroupOp initGroup;
};

/// A variant of types representing scheduleable operations.
using Scheduleable = std::variant<calyx::GroupOp, WhileScheduleable>;

/// A structure representing a set of ports which act as a memory interface.
struct CalyxMemoryPorts {
  Value readData;
  Value done;
  Value writeData;
  SmallVector<Value> addrPorts;
  Value writeEn;
};

/// The various lowering passes are agnostic wrt. whether working with a
/// calyx::MemoryOp (internally allocated memory) or external memory (through
/// CalyxMemoryPort). This is achieved through the following
/// CalyxMemoryInterface for accessing either a calyx::MemoryOp or a
/// CalyxMemoryPorts struct.
struct CalyxMemoryInterface {
  CalyxMemoryInterface() {}
  explicit CalyxMemoryInterface(const CalyxMemoryPorts &ports) : impl(ports) {}
  explicit CalyxMemoryInterface(calyx::MemoryOp memOp) : impl(memOp) {}

#define memoryInterfaceGetter(portName, TRet)                                  \
  TRet portName() {                                                            \
    if (auto memOp = std::get_if<calyx::MemoryOp>(&impl); memOp)               \
      return memOp->portName();                                                \
    else                                                                       \
      return std::get<CalyxMemoryPorts>(impl).portName;                        \
  }

  memoryInterfaceGetter(readData, Value);
  memoryInterfaceGetter(done, Value);
  memoryInterfaceGetter(writeData, Value);
  memoryInterfaceGetter(writeEn, Value);
  memoryInterfaceGetter(addrPorts, ValueRange);
#undef memoryInterfaceGetter

private:
  std::variant<calyx::MemoryOp, CalyxMemoryPorts> impl;
};

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Tries to match a constant value defined by op. If the match was
/// successful, returns true and binds the constant to 'value'. If unsuccessful,
/// the value is unmodified.
static bool matchConstantOp(Operation *op, APInt &value) {
  return mlir::detail::constant_int_op_binder(&value).match(op);
}

/// Returns true if there exists only a single memref::LoadOp which loads from
/// the memory referenced by loadOp.
static bool singleLoadFromMemory(memref::LoadOp loadOp) {
  return llvm::count_if(loadOp.memref().getUses(), [](auto &user) {
           return dyn_cast<memref::LoadOp>(user.getOwner());
         }) <= 1;
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
  return rewriter.create<TGroup>(loc, uniqueName.str());
}

/// Get the index'th output port of compOp.
static Value getComponentOutput(calyx::ComponentOp compOp,
                                unsigned outPortIdx) {
  size_t resIdx = compOp.getInputPortInfo().size() + outPortIdx;
  assert(compOp.getNumArguments() > resIdx &&
         "Exceeded number of arguments in the Component");
  return compOp.getArgument(resIdx);
}

/// If the provided type is an index type, converts it to i32, else, returns the
/// unmodified type.
static Type convIndexType(PatternRewriter &rewriter, Type type) {
  if (type.isIndex())
    return rewriter.getI32Type();
  return type;
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
    return rewriter.create<TLibraryOp>(
        loc, FlatSymbolRefAttr::get(rewriter.getContext(), getUniqueName(name)),
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
  void addWhileIterReg(scf::WhileOp whileOp, calyx::RegisterOp reg,
                       unsigned idx) {
    assert(whileIterRegs[whileOp].count(idx) == 0 &&
           "A register was already registered for the given while iter_arg "
           "index");
    assert(idx < whileOp.getAfterArguments().size());
    whileIterRegs[whileOp][idx] = reg;
  }

  /// Return a mapping of block argument indices to block argument registers.
  calyx::RegisterOp getWhileIterReg(scf::WhileOp whileOp, unsigned idx) {
    auto iterRegs = getWhileIterRegs(whileOp);
    auto it = iterRegs.find(idx);
    assert(it != iterRegs.end() &&
           "No iter arg register set for the provided index");
    return it->second;
  }

  /// Return a mapping of block argument indices to block argument registers.
  const DenseMap<unsigned, calyx::RegisterOp> &
  getWhileIterRegs(scf::WhileOp whileOp) {
    return whileIterRegs[whileOp];
  }

  /// Registers grp to be the while latch group of whileOp.
  void setWhileLatchGroup(scf::WhileOp whileOp, calyx::GroupOp grp) {
    assert(whileLatchGroups.count(whileOp) == 0 &&
           "A latch group was already set for this whileOp");
    whileLatchGroups[whileOp] = grp;
  }

  /// Retrieve the while latch group registered for whileOp.
  calyx::GroupOp getWhileLatchGroup(scf::WhileOp whileOp) {
    auto it = whileLatchGroups.find(whileOp);
    assert(it != whileLatchGroups.end() &&
           "No while latch group was set for this whileOp");
    return it->second;
  }

  /// Registers a memory interface as being associated with a memory identified
  /// by 'memref'.
  void registerMemoryInterface(Value memref,
                               const CalyxMemoryInterface &memoryInterface) {
    assert(memref.getType().isa<MemRefType>());
    assert(memories.find(memref) == memories.end() &&
           "Memory already registered for memref");
    memories[memref] = memoryInterface;
  }

  /// Returns the memory interface registered for the given memref.
  CalyxMemoryInterface getMemoryInterface(Value memref) {
    assert(memref.getType().isa<MemRefType>());
    auto it = memories.find(memref);
    assert(it != memories.end() && "No memory registered for memref");
    return it->second;
  }

  /// If v is an input to any memory registered within this component, returns
  /// the memory. If not, returns null.
  Optional<CalyxMemoryInterface> isInputPortOfMemory(Value v) {
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

  /// A mapping of currently available constants in this component.
  DenseMap<APInt, Value> constants;

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
  DenseMap<Value, CalyxMemoryInterface> memories;

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
  rewriter.create<calyx::AssignOp>(loc, reg.write_en(),
                                   state.getConstant(rewriter, loc, 1, 1));
  rewriter.create<calyx::GroupDoneOp>(loc, reg.donePort());
}

static calyx::GroupOp buildWhileIterArgAssignments(
    PatternRewriter &rewriter, ComponentLoweringState &state, Location loc,
    scf::WhileOp whileOp, Twine uniqueSuffix, ValueRange ops) {
  assert(whileOp);
  /// Pass iteration arguments through registers. This follows closely
  /// to what is done for branch ops.
  auto groupName = "assign_" + uniqueSuffix;
  auto groupOp = createGroup<calyx::GroupOp>(rewriter, state.getComponentOp(),
                                             loc, groupName);
  // Create register assignment for each iter_arg. a calyx::GroupDone signal
  // is created for each register. This is later cleaned up in
  // GroupDoneCleanupPattern.
  for (auto arg : enumerate(ops)) {
    auto reg = state.getWhileIterReg(whileOp, arg.index());
    buildAssignmentsForRegisterWrite(state, rewriter, groupOp, reg,
                                     arg.value());
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

/// Creates a new register within the component associated to 'compState'.
static calyx::RegisterOp createReg(ComponentLoweringState &compState,
                                   PatternRewriter &rewriter, Location loc,
                                   Twine prefix, size_t width) {
  IRRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(compState.getComponentOp().getBody());
  return rewriter.create<calyx::RegisterOp>(loc, (prefix + "_reg").str(),
                                            width);
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
              .template Case<ConstantOp, ReturnOp, BranchOpInterface,
                             /// SCF
                             scf::YieldOp,
                             /// memref
                             memref::AllocOp, memref::AllocaOp, memref::LoadOp,
                             memref::StoreOp,
                             /// standard arithmetic
                             AddIOp, SubIOp, CmpIOp, ShiftLeftOp,
                             UnsignedShiftRightOp, SignedShiftRightOp, AndOp,
                             XOrOp, OrOp, ZeroExtendIOp, TruncateIOp,
                             IndexCastOp>(
                  [&](auto op) { return buildOp(rewriter, op).succeeded(); })
              .template Case<scf::WhileOp, mlir::FuncOp, scf::ConditionOp>(
                  [&](auto) {
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
    return createGroup<TGroupOp>(rewriter, getComponentState().getComponentOp(),
                                 block->front().getLoc(), groupName);
  }

  /// Creates assignments within the provided group to the address ports of the
  /// memoryOp based on the provided addressValues.
  void assignAddressPorts(PatternRewriter &rewriter, Location loc,
                          calyx::GroupInterface group,
                          CalyxMemoryInterface memoryInterface,
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
  auto memoryInterface =
      getComponentState().getMemoryInterface(loadOp.memref());
  if (singleLoadFromMemory(loadOp)) {
    /// Single load from memory; Combinational case - we do not have to consider
    /// adding registers in front of the memory.
    auto combGroup = createGroupForOp<calyx::CombGroupOp>(rewriter, loadOp);
    assignAddressPorts(rewriter, loadOp.getLoc(), combGroup, memoryInterface,
                       loadOp.getIndices());

    /// We refrain from replacing the loadOp result with
    /// memoryInterface.readData, since multiple loadOp's need to be converted
    /// to a single memory's ReadData. If this replacement is done now, we lose
    /// the link between which SSA memref::LoadOp values map to which groups for
    /// loading a value from the Calyx memory. At this point of lowering, we
    /// keep the memref::LoadOp SSA value, and do value replacement _after_
    /// control has been generated (see LateSSAReplacement). This is *vital* for
    /// things such as InlineCombGroups to be able to properly track which
    /// memory assignment groups belong to which accesses.
    getComponentState().registerEvaluatingGroup(loadOp.getResult(), combGroup);
  } else {
    auto group = createGroupForOp<calyx::GroupOp>(rewriter, loadOp);
    assignAddressPorts(rewriter, loadOp.getLoc(), group, memoryInterface,
                       loadOp.getIndices());

    /// Multiple loads from the same memory; In this case, we _may_ have a
    /// structural hazard in the design we generate. To get around this, we
    /// conservatively place a register in front of each load operation, and
    /// replace all uses of the loaded value with the register output. Proper
    /// handling of this requires the combinational group inliner/scheduler to
    /// be aware of when a combinational expression references multiple loaded
    /// values from the same memory, and then schedule assignments to temporary
    /// registers to get around the structural hazard.
    auto reg = createReg(getComponentState(), rewriter, loadOp.getLoc(),
                         getComponentState().getUniqueName("load"),
                         loadOp.getMemRefType().getElementTypeBitWidth());
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

  /// This is a sequential group, so register it as being scheduleable for the
  /// block.
  getComponentState().addBlockScheduleable(storeOp->getBlock(),
                                           cast<calyx::GroupOp>(group));
  assignAddressPorts(rewriter, storeOp.getLoc(), group, memoryInterface,
                     storeOp.getIndices());
  rewriter.setInsertionPointToEnd(group.getBody());
  rewriter.create<calyx::AssignOp>(
      storeOp.getLoc(), memoryInterface.writeData(), storeOp.getValueToStore());
  rewriter.create<calyx::AssignOp>(
      storeOp.getLoc(), memoryInterface.writeEn(),
      getComponentState().getConstant(rewriter, storeOp.getLoc(), 1, 1));
  rewriter.create<calyx::GroupDoneOp>(storeOp.getLoc(), memoryInterface.done());
  return success();
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
    addrSizes.push_back(llvm::Log2_64_Ceil(dim));
  }
  auto memoryOp = rewriter.create<calyx::MemoryOp>(
      allocOp.getLoc(), componentState.getUniqueName("mem"),
      memtype.getElementType().getIntOrFloatBitWidth(), sizes, addrSizes);
  componentState.registerMemoryInterface(allocOp.getResult(),
                                         CalyxMemoryInterface(memoryOp));
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
  yieldOp.getOperands();
  auto assignGroup = buildWhileIterArgAssignments(
      rewriter, getComponentState(), yieldOp.getLoc(), whileOp,
      getComponentState().getUniqueName(whileOp) + "_latch",
      yieldOp.getOperands());
  getComponentState().setWhileLatchGroup(whileOp, assignGroup);
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
    if (!succOperands.hasValue() || succOperands.getValue().size() == 0)
      continue;
    // Create operand passing group
    std::string groupName = progState().blockName(srcBlock) + "_to_" +
                            progState().blockName(succBlock.value());
    auto groupOp = createGroup<calyx::GroupOp>(rewriter, *getComponent(),
                                               brOp.getLoc(), groupName);
    // Fetch block argument registers associated with the basic block
    auto dstBlockArgRegs =
        getComponentState().getBlockArgRegs(succBlock.value());
    // Create register assignment for each block argument
    for (auto arg : enumerate(succOperands.getValue())) {
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
  }
  llvm_unreachable("unsupported comparison predicate");
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

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     IndexCastOp op) const {
  Type sourceType = convIndexType(rewriter, op.getOperand().getType());
  Type targetType = convIndexType(rewriter, op.getResult().getType());
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
    auto sliceOp = state.getNewLibraryOpInstance<calyx::SliceLibOp>(
        rewriter, assignOp.getLoc(), types);
    rewriter.setInsertionPoint(assignOp->getBlock(),
                               assignOp->getBlock()->begin());
    rewriter.create<calyx::AssignOp>(assignOp->getLoc(), sliceOp.getResult(0),
                                     src);
    assignOp.setOperand(1, sliceOp.getResult(1));

    return success();
  }

private:
  ProgramLoweringState &pls;
};

/// Connverts all index-typed operations and values to i32 values.
class ConvertIndexTypes : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    funcOp.walk([&](Block *block) {
      for (auto arg : block->getArguments())
        arg.setType(convIndexType(rewriter, arg.getType()));
    });

    funcOp.walk([&](Operation *op) {
      for (auto res : op->getResults()) {
        auto resType = res.getType();
        if (!resType.isIndex())
          continue;

        res.setType(convIndexType(rewriter, resType));
        if (auto constOp = dyn_cast<ConstantOp>(op)) {
          APInt value;
          matchConstantOp(constOp, value);
          rewriter.setInsertionPoint(constOp);
          rewriter.replaceOpWithNewOp<ConstantOp>(
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
    sinkBlock->addArguments(yieldTypes);
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
                             Value memref,
                             SmallVectorImpl<calyx::PortInfo> &inPorts,
                             SmallVectorImpl<calyx::PortInfo> &outPorts) {
  MemRefType memrefType = memref.getType().cast<MemRefType>();

  /// Read data
  inPorts.push_back(
      calyx::PortInfo{rewriter.getStringAttr(memName + "_read_data"),
                      memrefType.getElementType(), calyx::Direction::Input,
                      DictionaryAttr::get(rewriter.getContext(), {})});

  /// Done
  inPorts.push_back(calyx::PortInfo{
      rewriter.getStringAttr(memName + "_done"), rewriter.getI1Type(),
      calyx::Direction::Input, DictionaryAttr::get(rewriter.getContext(), {})});

  /// Write data
  outPorts.push_back(
      calyx::PortInfo{rewriter.getStringAttr(memName + "_write_data"),
                      memrefType.getElementType(), calyx::Direction::Output,
                      DictionaryAttr::get(rewriter.getContext(), {})});

  /// Memory address outputs
  for (auto dim : enumerate(memrefType.getShape())) {
    outPorts.push_back(calyx::PortInfo{
        rewriter.getStringAttr(memName + "_addr" + std::to_string(dim.index())),
        rewriter.getIntegerType(dim.value()), calyx::Direction::Output,
        DictionaryAttr::get(rewriter.getContext(), {})});
  }

  /// Write enable
  outPorts.push_back(
      calyx::PortInfo{rewriter.getStringAttr(memName + "_write_en"),
                      rewriter.getI1Type(), calyx::Direction::Output,
                      DictionaryAttr::get(rewriter.getContext(), {})});
}

/// Creates a new Calyx component for each FuncOp in the program.
struct FuncOpConversion : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
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
    FunctionType funcType = funcOp.getType();
    for (auto &arg : enumerate(funcOp.getArguments())) {
      if (arg.value().getType().isa<MemRefType>()) {
        /// External memories
        auto memName =
            "ext_mem" + std::to_string(extMemoryCompPortIndices.size());
        extMemoryCompPortIndices[arg.value()] = {inPorts.size(),
                                                 outPorts.size()};
        appendPortsForExternalMemref(rewriter, memName, arg.value(), inPorts,
                                     outPorts);
      } else {
        /// Single-port arguments
        auto inName = "in" + std::to_string(arg.index());
        funcOpArgRewrites[arg.value()] = inPorts.size();
        inPorts.push_back(
            calyx::PortInfo{rewriter.getStringAttr(inName),
                            convIndexType(rewriter, arg.value().getType()),
                            calyx::Direction::Input,
                            DictionaryAttr::get(rewriter.getContext(), {})});
      }
    }
    for (auto &res : enumerate(funcType.getResults())) {
      funcOpResultMapping[res.index()] = outPorts.size();
      outPorts.push_back(calyx::PortInfo{
          rewriter.getStringAttr("out" + std::to_string(res.index())),
          convIndexType(rewriter, res.value()), calyx::Direction::Output,
          DictionaryAttr::get(rewriter.getContext(), {})});
    }

    /// We've now recorded all necessary indices. Merge in- and output ports
    /// and add the required mandatory component ports.
    auto ports = inPorts;
    llvm::append_range(ports, outPorts);
    addMandatoryComponentPorts(rewriter, ports);

    /// Create a calyx::ComponentOp corresponding to the to-be-lowered function.
    auto compOp = rewriter.create<calyx::ComponentOp>(
        funcOp.getLoc(), rewriter.getStringAttr(funcOp.sym_name()), ports);

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
      CalyxMemoryPorts extMemPorts;
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
                                        CalyxMemoryInterface(extMemPorts));
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
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    LogicalResult res = success();
    funcOp.walk([&](scf::WhileOp whileOp) {
      getComponentState().setUniqueName(whileOp.getOperation(), "while");

      /// Check for do-while loops.
      /// TODO(mortbopet) can we support these? for now, do not support loops
      /// where iterargs are changed in the 'before' region. scf.WhileOp also
      /// has support for different types of iter_args and return args which we
      /// also do not support; iter_args and while return values are placed in
      /// the same registers.
      for (auto barg : enumerate(whileOp.before().front().getArguments())) {
        auto condOp = whileOp.getConditionOp().args()[barg.index()];
        if (barg.value() != condOp) {
          res = whileOp.emitError()
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
      for (auto arg : enumerate(whileOp.getAfterArguments())) {
        std::string name = getComponentState().getUniqueName(whileOp).str() +
                           "_arg" + std::to_string(arg.index());
        auto reg =
            createReg(getComponentState(), rewriter, arg.value().getLoc(), name,
                      arg.value().getType().getIntOrFloatBitWidth());
        getComponentState().addWhileIterReg(whileOp, reg, arg.index());
        arg.value().replaceAllUsesWith(reg.out());

        /// Also replace uses in the "before" region of the while loop
        whileOp.before()
            .front()
            .getArgument(arg.index())
            .replaceAllUsesWith(reg.out());
      }

      /// Create iter args initial value assignment group
      auto initGroupOp = buildWhileIterArgAssignments(
          rewriter, getComponentState(), whileOp.getLoc(), whileOp,
          getComponentState().getUniqueName(whileOp) + "_init",
          whileOp.getOperands());

      /// Add the while op to the list of scheduleable things in the current
      /// block.
      getComponentState().addBlockScheduleable(
          whileOp->getBlock(), WhileScheduleable{whileOp, initGroupOp});
      return WalkResult::advance();
    });
    return res;
  }
};

/// Builds registers for each block argument in the program.
class BuildBBRegs : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
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
        auto reg = createReg(getComponentState(), rewriter,
                             arg.value().getLoc(), name, width);
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
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {

    for (auto argType : enumerate(funcOp.getType().getResults())) {
      auto convArgType = convIndexType(rewriter, argType.value());
      assert(convArgType.isa<IntegerType>() && "unsupported return type");
      unsigned width = convArgType.getIntOrFloatBitWidth();
      std::string name = "ret_arg" + std::to_string(argType.index());
      auto reg = createReg(getComponentState(), rewriter, funcOp.getLoc(), name,
                           width);
      getComponentState().addReturnReg(reg, argType.index());

      rewriter.setInsertionPointToStart(getComponent()->getWiresOp().getBody());
      rewriter.create<calyx::AssignOp>(
          funcOp->getLoc(),
          getComponentOutput(
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
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
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
        rewriter.create<calyx::EnableOp>(loc, groupPtr->sym_name());
      } else if (auto whileSchedPtr = std::get_if<WhileScheduleable>(&group);
                 whileSchedPtr) {
        auto &whileOp = whileSchedPtr->whileOp;

        /// Insert while iter arg initialization group.
        rewriter.create<calyx::EnableOp>(loc,
                                         whileSchedPtr->initGroup.getName());

        auto cond = whileOp.getConditionOp().getOperand(0);
        auto condGroup =
            getComponentState().getEvaluatingGroup<calyx::CombGroupOp>(cond);
        auto symbolAttr = FlatSymbolRefAttr::get(
            StringAttr::get(getContext(), condGroup.sym_name()));
        auto whileCtrlOp =
            rewriter.create<calyx::WhileOp>(loc, cond, symbolAttr);
        rewriter.setInsertionPointToEnd(whileCtrlOp.getBody());
        auto whileSeqOp = rewriter.create<calyx::SeqOp>(whileOp.getLoc());

        /// Only schedule the after block. The 'before' block is
        /// implicitly scheduled when evaluating the while condition.
        LogicalResult res =
            buildCFGControl(path, rewriter, whileSeqOp.getBody(), block,
                            &whileOp.after().front());
        // Insert loop-latch at the end of the while group
        rewriter.setInsertionPointToEnd(whileSeqOp.getBody());
        rewriter.create<calyx::EnableOp>(
            loc, getComponentState().getWhileLatchGroup(whileOp).getName());
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
      rewriter.create<calyx::EnableOp>(loc, barg.sym_name());

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
      auto src = assignOp.src();
      auto srcDefOp = src.getDefiningOp();

      /// Things which stop recursive inlining (or in other words, what
      /// breaks combinational paths).
      /// - Component inputs
      /// - Register and memory reads
      /// - Constant ops (constant ops are not evaluated by any group)
      /// - 'While' return values (these are registers, however, 'while'
      ///   return values have at the current point of conversion not yet
      ///   been rewritten to their register outputs, see comment in
      ///   LateSSAReplacement)
      if (src.isa<BlockArgument>() ||
          isa<calyx::RegisterOp, calyx::MemoryOp, hw::ConstantOp, ConstantOp,
              scf::WhileOp>(srcDefOp))
        continue;

      auto srcCombGroup = state.getEvaluatingGroup<calyx::CombGroupOp>(src);
      assert(srcCombGroup && "expected combinational group");
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

  LogicalResult PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                                         PatternRewriter &) const override {
    funcOp.walk([&](scf::WhileOp whileOp) {
      /// The yielded values returned from the while op will be present in the
      /// iterargs registers post execution of the loop.
      /// This is done now, as opposed to during BuildWhileGroups since if the
      /// results of the whileOp were replaced before
      /// BuildOpGroups/BuildControl, the whileOp would get dead-code
      /// eliminated.
      for (auto res : getComponentState().getWhileIterRegs(whileOp))
        whileOp.getResults()[res.first].replaceAllUsesWith(res.second.out());
    });

    funcOp.walk([&](memref::LoadOp loadOp) {
      if (singleLoadFromMemory(loadOp)) {
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
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
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

    /// Create an and-tree of all calyx::GroupDoneOp's.
    rewriter.setInsertionPointToEnd(groupDoneOps[0]->getBlock());
    Value acc = groupDoneOps[0].src();
    for (auto groupDoneOp : llvm::makeArrayRef(groupDoneOps).drop_front(1)) {
      auto newAndOp = rewriter.create<comb::AndOp>(groupDoneOp.getLoc(), acc,
                                                   groupDoneOp.src());
      acc = newAndOp.getResult();
    }

    /// Create a group done op with the complex expression as a guard.
    rewriter.create<calyx::GroupDoneOp>(
        groupOp.getLoc(),
        rewriter.create<hw::ConstantOp>(groupOp.getLoc(), APInt(1, 1)), acc);
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
                      PatternArgs &&... args) {
    RewritePatternSet ps(&getContext());
    ps.add<TPattern>(&getContext(), partialPatternRes, args...);
    patterns.push_back(
        LoweringPattern{std::move(ps), LoweringPattern::Strategy::Once});
  }

  template <typename TPattern, typename... PatternArgs>
  void addGreedyPattern(SmallVectorImpl<LoweringPattern> &patterns,
                        PatternArgs &&... args) {
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

  getOperation().dump();
}

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createSCFToCalyxPass() {
  return std::make_unique<SCFToCalyxPass>();
}

} // namespace circt
