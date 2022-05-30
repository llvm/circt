//===- CalyxLoweringUtils.h - Calyx lowering utility methods ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines various lowering utility methods for converting to
// and from Calyx programs.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H
#define CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"

#include <variant>

namespace circt {
namespace calyx {

// Walks the control of this component, and appends source information for leaf
// nodes. It also appends a position attribute that connects the source location
// metadata to the corresponding control operation.
WalkResult
getCiderSourceLocationMetadata(calyx::ComponentOp component,
                               SmallVectorImpl<Attribute> &sourceLocations);

// Tries to match a constant value defined by op. If the match was
// successful, returns true and binds the constant to 'value'. If unsuccessful,
// the value is unmodified.
bool matchConstantOp(Operation *op, APInt &value);

// Returns true if there exists only a single memref::LoadOp which loads from
// the memory referenced by loadOp.
bool singleLoadFromMemory(Value memoryReference);

// Returns true if there are no memref::StoreOp uses with the referenced
// memory.
bool noStoresToMemory(Value memoryReference);

// Get the index'th output port of compOp.
Value getComponentOutput(calyx::ComponentOp compOp, unsigned outPortIdx);

// If the provided type is an index type, converts it to i32, else, returns the
// unmodified type.
Type convIndexType(PatternRewriter &rewriter, Type type);

// Creates a new calyx::CombGroupOp or calyx::GroupOp group within compOp.
template <typename TGroup>
TGroup createGroup(PatternRewriter &rewriter, calyx::ComponentOp compOp,
                   Location loc, Twine uniqueName) {
  mlir::IRRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(compOp.getWiresOp().getBody());
  return rewriter.create<TGroup>(loc, uniqueName.str());
}

// A structure representing a set of ports which act as a memory interface for
// external memories.
struct MemoryPortsImpl {
  Value readData;
  Value done;
  Value writeData;
  SmallVector<Value> addrPorts;
  Value writeEn;
};

// Represents the interface of memory in Calyx. The various lowering passes
// are agnostic wrt. whether working with a calyx::MemoryOp (internally
// allocated memory) or MemoryPortsImpl (external memory).
struct MemoryInterface {
  MemoryInterface();
  explicit MemoryInterface(const MemoryPortsImpl &ports);
  explicit MemoryInterface(calyx::MemoryOp memOp);

  // Getter methods for each memory interface port.
  Value readData();
  Value done();
  Value writeData();
  Value writeEn();
  ValueRange addrPorts();

private:
  std::variant<calyx::MemoryOp, MemoryPortsImpl> impl;
};

// Provide a common interface for loop operations that need to be lowered to
// Calyx.
class LoopInterface {
public:
  virtual ~LoopInterface() = default;

  // Returns the arguments to this while operation.
  virtual Block::BlockArgListType getBodyArgs() = 0;

  // Returns body of this while operation.
  virtual Block *getBodyBlock() = 0;

  // Returns the Block in which the condition exists.
  virtual Block *getConditionBlock() = 0;

  // Returns the condition as a Value.
  virtual Value getConditionValue() = 0;

  // Returns the number of iterations the while loop will conduct if known.
  virtual Optional<uint64_t> getBound() = 0;
};

// Provides an interface for the control flow `while` operation across different
// dialects.
template <typename T>
class WhileOpInterface : LoopInterface {
  static_assert(std::is_convertible_v<T, Operation *>);

public:
  explicit WhileOpInterface(T op) : impl(op) {}
  explicit WhileOpInterface(Operation *op) : impl(dyn_cast_or_null<T>(op)) {}

  // Returns the operation.
  T getOperation() { return impl; }

  // Returns the source location of the operation.
  Location getLoc() { return impl->getLoc(); }

private:
  T impl;
};

//===----------------------------------------------------------------------===//
// Lowering state classes
//===----------------------------------------------------------------------===//

// An interface for the Calyx component lowering state.
template <typename Loop>
class ComponentLoweringStateInterface {
  static_assert(std::is_base_of_v<LoopInterface, Loop>);

public:
  ComponentLoweringStateInterface(calyx::ComponentOp component)
      : component(component) {}

  virtual ~ComponentLoweringStateInterface() = default;

  /// Register reg as being the idx'th iter_args register for 'op'.
  virtual void addWhileIterReg(Loop op, calyx::RegisterOp reg,
                               unsigned idx) = 0;

  /// Return a mapping of block argument indices to block argument.
  virtual calyx::RegisterOp getWhileIterReg(Loop op, unsigned idx) = 0;

  /// Return a mapping of block argument indices to block argument.
  virtual const DenseMap<unsigned, calyx::RegisterOp> &
  getWhileIterRegs(Loop op) = 0;

  /// Registers grp to be the while latch group of `op`.
  virtual void setWhileLatchGroup(Loop op, calyx::GroupOp group) = 0;

  /// Retrieve the while latch group registered for `op`.
  virtual calyx::GroupOp getWhileLatchGroup(Loop op) = 0;

  /// Returns the calyx::ComponentOp associated with this lowering state.
  calyx::ComponentOp getComponentOp() { return component; }

  /// Register reg as being the idx'th argument register for block. This is
  /// necessary for the `BuildBBReg` pass.
  void addBlockArgReg(Block *block, calyx::RegisterOp reg, unsigned idx) {
    assert(blockArgRegs[block].count(idx) == 0);
    assert(idx < block->getArguments().size());
    blockArgRegs[block][idx] = reg;
  }

  /// Return a mapping of block argument indices to block argument registers.
  /// This is necessary for the `BuildBBReg` pass.
  const DenseMap<unsigned, calyx::RegisterOp> &getBlockArgRegs(Block *block) {
    return blockArgRegs[block];
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
    mlir::IRRewriter::InsertionGuard guard(rewriter);
    Block *body = component.getBody();
    rewriter.setInsertionPoint(body, body->begin());
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

  /// Creates register assignment operations within the provided groupOp.
  void buildAssignmentsForRegisterWrite(PatternRewriter &rewriter,
                                        calyx::GroupOp groupOp,
                                        calyx::RegisterOp &reg,
                                        Value inputValue) {
    mlir::IRRewriter::InsertionGuard guard(rewriter);
    auto loc = inputValue.getLoc();
    rewriter.setInsertionPointToEnd(groupOp.getBody());
    rewriter.create<calyx::AssignOp>(loc, reg.in(), inputValue);
    rewriter.create<calyx::AssignOp>(
        loc, reg.write_en(),
        createConstant(loc, rewriter, getComponentOp(), 1, 1));
    rewriter.create<calyx::GroupDoneOp>(loc, reg.done());
  }

  /// Creates a new group that assigns the 'ops' values to the iter arg
  /// registers of the 'whileOp'.
  calyx::GroupOp buildWhileIterArgAssignments(PatternRewriter &rewriter,
                                              Loop op, Twine uniqueSuffix,
                                              MutableArrayRef<OpOperand> ops) {
    /// Pass iteration arguments through registers. This follows closely
    /// to what is done for branch ops.
    auto groupName = "assign_" + uniqueSuffix;
    auto groupOp = calyx::createGroup<calyx::GroupOp>(
        rewriter, getComponentOp(), op.getLoc(), groupName);
    /// Create register assignment for each iter_arg. a calyx::GroupDone signal
    /// is created for each register. These will be &'ed together in
    /// MultipleGroupDonePattern.
    for (auto &arg : ops) {
      auto reg = getWhileIterReg(op, arg.getOperandNumber());
      buildAssignmentsForRegisterWrite(rewriter, groupOp, reg, arg.get());
    }
    return groupOp;
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
  /// The component which this lowering state is associated to.
  calyx::ComponentOp component;

  /// A mapping from blocks to block argument registers.
  DenseMap<Block *, DenseMap<unsigned, calyx::RegisterOp>> blockArgRegs;

  /// Block arg groups is a list of groups that should be sequentially
  /// executed when passing control from the source to destination block.
  /// Block arg groups are executed before blockScheduleables (akin to a
  /// phi-node).
  DenseMap<Block *, DenseMap<Block *, SmallVector<calyx::GroupOp>>>
      blockArgGroups;

  /// A mapping of string prefixes and the current uniqueness counter for that
  /// prefix. Used to generate unique names.
  std::map<std::string, unsigned> prefixIdMap;

  /// A mapping from Operations and previously assigned unique name of the op.
  std::map<Operation *, std::string> opNames;

  /// A mapping between SSA values and the groups which assign them.
  DenseMap<Value, calyx::GroupInterface> valueGroupAssigns;

  /// A mapping from return value indexes to return value registers.
  DenseMap<unsigned, calyx::RegisterOp> returnRegs;

  /// A mapping from memref's to their corresponding Calyx memory interface.
  DenseMap<Value, calyx::MemoryInterface> memories;

  /// A mapping between the source funcOp result indices and the corresponding
  /// output port indices of this componentOp.
  DenseMap<unsigned, unsigned> funcOpResultMapping;
};

/// An interface for conversion passes that lower Calyx programs.
class ProgramLoweringStateInterface {
public:
  explicit ProgramLoweringStateInterface(calyx::ProgramOp program,
                                         StringRef topLevelFunction);

  /// Returns a meaningful name for a value within the program scope.
  template <typename ValueOrBlock>
  std::string irName(ValueOrBlock &v) {
    std::string s;
    llvm::raw_string_ostream os(s);
    mlir::AsmState asmState(program);
    v.printAsOperand(os, asmState);
    return s;
  }

  /// Returns a meaningful name for a block within the program scope (removes
  /// the ^ prefix from block names).
  std::string blockName(Block *b);

  /// Returns the current program.
  calyx::ProgramOp getProgram();

  /// Returns the name of the top-level function in the source program.
  StringRef getTopLevelFunction() const;

private:
  /// The name of this top-level function.
  StringRef topLevelFunction;
  /// The program associated with this state.
  calyx::ProgramOp program;
};

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

struct ModuleOpConversion : public OpRewritePattern<mlir::ModuleOp> {
  ModuleOpConversion(MLIRContext *context, StringRef topLevelFunction,
                     calyx::ProgramOp *programOpOutput);

  LogicalResult matchAndRewrite(mlir::ModuleOp moduleOp,
                                PatternRewriter &rewriter) const override;

private:
  calyx::ProgramOp *programOpOutput;
  StringRef topLevelFunction;
};

} // namespace calyx
} // namespace circt

#endif // CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H
