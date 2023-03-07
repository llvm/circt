//===- IMConstProp.cpp - Intermodule ConstProp and DCE ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements SCCP:
// https://www.cs.wustl.edu/~cytron/531Pages/f11/Resources/Papers/cprop.pdf
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/APInt.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace circt;
using namespace firrtl;

#define DEBUG_TYPE "IMCP"

/// Return true if this is a wire or register.
static bool isWireOrReg(Operation *op) {
  return isa<WireOp, RegResetOp, RegOp>(op);
}

/// Return true if this is an aggregate indexer.
static bool isAggregate(Operation *op) {
  return isa<SubindexOp, SubaccessOp, SubfieldOp>(op);
}

/// Return true if this is a wire or register we're allowed to delete.
static bool isDeletableWireOrRegOrNode(Operation *op) {
  return (isWireOrReg(op) || isa<NodeOp>(op)) && AnnotationSet(op).empty() &&
         !hasDontTouch(op) && hasDroppableName(op);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a single lattice value. A lattive value corresponds to
/// the various different states that a value in the SCCP dataflow analysis can
/// take. See 'Kind' below for more details on the different states a value can
/// take.
class LatticeValue {
  enum Kind {
    /// A value with a yet-to-be-determined value. This state may be changed to
    /// anything, it hasn't been processed by IMConstProp.
    Unknown,

    /// A value with an unknown, but live state.  This is used for wires so that
    /// they can start in a defined state and the solver won't have unanalyzed
    /// parts of the IR.
    Unwritten,

    /// A value that is known to be a constant. This state may be changed to
    /// overdefined.
    Constant,

    /// A value that cannot statically be determined to be a constant. This
    /// state cannot be changed.
    Overdefined
  };

public:
  /// Initialize a lattice value with "Unknown".
  /*implicit*/ LatticeValue() : valueAndTag(nullptr, Kind::Unknown) {}
  /// Initialize a lattice value with a constant.
  /*implicit*/ LatticeValue(IntegerAttr attr)
      : valueAndTag(attr, Kind::Constant) {}

  static LatticeValue getOverdefined() {
    LatticeValue result;
    result.markOverdefined();
    return result;
  }

  static LatticeValue getUnwritten() {
    LatticeValue result;
    result.markUnwritten();
    return result;
  }

  bool isUnknown() const { return valueAndTag.getInt() == Kind::Unknown; }
  bool isUnwritten() const { return valueAndTag.getInt() == Kind::Unwritten; }
  bool isConstant() const { return valueAndTag.getInt() == Kind::Constant; }
  bool isOverdefined() const {
    return valueAndTag.getInt() == Kind::Overdefined;
  }

  /// Mark the lattice value as overdefined.
  void markOverdefined() {
    valueAndTag.setPointerAndInt(nullptr, Kind::Overdefined);
  }

  /// Mark the lattice value as unwritten.
  void markUnwritten() {
    valueAndTag.setPointerAndInt(nullptr, Kind::Unwritten);
  }

  /// Mark the lattice value as constant.
  void markConstant(IntegerAttr value) {
    valueAndTag.setPointerAndInt(value, Kind::Constant);
  }

  /// If this lattice is constant or invalid value, return the attribute.
  /// Returns nullptr otherwise.
  Attribute getValue() const { return valueAndTag.getPointer(); }

  /// If this is in the constant state, return the IntegerAttr.
  IntegerAttr getConstant() const {
    assert(isConstant());
    return getValue().dyn_cast_or_null<IntegerAttr>();
  }

  /// Merge in the value of the 'rhs' lattice into this one. Returns true if the
  /// lattice value changed.
  bool mergeIn(LatticeValue rhs) {
    // If we are already overdefined, or rhs is unknown, there is nothing to do.
    if (isOverdefined() || rhs.isUnknown())
      return false;

    // If we are unknown, just take the value of rhs.
    if (isUnknown()) {
      valueAndTag = rhs.valueAndTag;
      return true;
    }

    // Unwritten values don't add value.
    if (rhs.isUnwritten())
      return false;

    // In unwritten, only promote.
    if (isUnwritten()) {
      valueAndTag = rhs.valueAndTag;
      return true;
    }

    // Otherwise, if this value doesn't match rhs go straight to overdefined.
    // This happens when we merge "3" and "4" from two different instance sites
    // for example.
    if (valueAndTag != rhs.valueAndTag) {
      markOverdefined();
      return true;
    }
    return false;
  }

  bool operator==(const LatticeValue &other) const {
    return valueAndTag == other.valueAndTag;
  }
  bool operator!=(const LatticeValue &other) const {
    return valueAndTag != other.valueAndTag;
  }

private:
  /// The attribute value if this is a constant and the tag for the element
  /// kind.  The attribute is always an IntegerAttr.
  llvm::PointerIntPair<Attribute, 2, Kind> valueAndTag;
};
} // end anonymous namespace

LLVM_ATTRIBUTE_USED
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const LatticeValue &lattice) {
  if (lattice.isUnknown())
    return os << "<Unknown>";
  if (lattice.isOverdefined())
    return os << "<Overdefined>";
  if (lattice.isUnwritten())
    return os << "<Unwritten>";
  return os << "<" << lattice.getConstant() << ">";
}

namespace {
struct IMConstPropPass : public IMConstPropBase<IMConstPropPass> {

  void runOnOperation() override;
  void rewriteModuleBody(FModuleOp module);

  /// Returns true if the given block is executable.
  bool isBlockExecutable(Block *block) const {
    return executableBlocks.count(block);
  }

  bool isOverdefined(Value value) const {
    auto it = latticeValues.find(value);
    return it != latticeValues.end() && it->second.isOverdefined();
  }

  /// Mark the given value as overdefined. This means that we cannot refine a
  /// specific constant for this value.
  void markOverdefined(Value value) {
    auto &entry = latticeValues[value];
    if (!entry.isOverdefined()) {
      LLVM_DEBUG({
        logger.getOStream() << "Setting overdefined : (" << value << ")\n";
      });
      entry.markOverdefined();
      changedLatticeValueWorklist.push_back(value);
    }
  }

  /// Mark the given value as overdefined. This means that we cannot refine a
  /// specific constant for this value.
  void markUnwritten(Value value) {
    auto &entry = latticeValues[value];
    if (!entry.isUnwritten()) {
      LLVM_DEBUG({
        logger.getOStream() << "Setting unwritten : (" << value << ")\n";
      });
      entry.markUnwritten();
      changedLatticeValueWorklist.push_back(value);
    }
  }

  /// Merge information from the 'from' lattice value into value.  If it
  /// changes, then users of the value are added to the worklist for
  /// revisitation.
  void mergeLatticeValue(Value value, LatticeValue &valueEntry,
                         LatticeValue source) {
    if (valueEntry.mergeIn(source)) {
      LLVM_DEBUG({
        logger.getOStream()
            << "Changed to " << valueEntry << " : (" << value << ")\n";
      });
      changedLatticeValueWorklist.push_back(value);
    }
  }
  void mergeLatticeValue(Value value, LatticeValue source) {
    // Don't even do a map lookup if from has no info in it.
    if (source.isUnknown())
      return;
    mergeLatticeValue(value, latticeValues[value], source);
  }
  void mergeLatticeValue(Value result, Value from) {
    // If 'from' hasn't been computed yet, then it is unknown, don't do
    // anything.
    auto it = latticeValues.find(from);
    if (it == latticeValues.end())
      return;
    mergeLatticeValue(result, it->second);
  }

  /// setLatticeValue - This is used when a new LatticeValue is computed for
  /// the result of the specified value that replaces any previous knowledge,
  /// e.g. because a fold() function on an op returned a new thing.  This should
  /// not be used on operations that have multiple contributors to it, e.g.
  /// wires or ports.
  void setLatticeValue(Value value, LatticeValue source) {
    // Don't even do a map lookup if from has no info in it.
    if (source.isUnknown())
      return;

    // If we've changed this value then revisit all the users.
    auto &valueEntry = latticeValues[value];
    if (valueEntry != source) {
      changedLatticeValueWorklist.push_back(value);
      valueEntry = source;
    }
  }

  /// Return the lattice value for the specified SSA value, extended to the
  /// width of the specified destType.  If allowTruncation is true, then this
  /// allows truncating the lattice value to the specified type.
  LatticeValue getExtendedLatticeValue(Value value, FIRRTLBaseType destType,
                                       bool allowTruncation = false);

  /// Mark the given block as executable.
  void markBlockExecutable(Block *block);
  void markWireOrRegOp(Operation *wireOrReg);
  void markMemOp(MemOp mem);

  void markInvalidValueOp(InvalidValueOp invalid);
  void markConstantOp(ConstantOp constant);
  void markSpecialConstantOp(SpecialConstantOp specialConstant);
  void markInstanceOp(InstanceOp instance);

  void visitConnectLike(FConnectLike connect);
  void visitRegResetOp(RegResetOp regReset);
  void visitRefSend(RefSendOp send);
  void visitRefResolve(RefResolveOp resolve);
  void visitNode(NodeOp node);
  void visitOperation(Operation *op);

private:
  /// This is the current instance graph for the Circuit.
  InstanceGraph *instanceGraph = nullptr;

  /// This keeps track of the current state of each tracked value.
  DenseMap<Value, LatticeValue> latticeValues;

  /// The set of blocks that are known to execute, or are intrinsically live.
  SmallPtrSet<Block *, 16> executableBlocks;

  /// A worklist of values whose LatticeValue recently changed, indicating the
  /// users need to be reprocessed.
  SmallVector<Value, 64> changedLatticeValueWorklist;

  /// This keeps track of users the instance results that correspond to output
  /// ports.
  DenseMap<BlockArgument, llvm::TinyPtrVector<Value>>
      resultPortToInstanceResultMapping;

#ifndef NDEBUG
  /// A logger used to emit information during the application process.
  llvm::ScopedPrinter logger{llvm::dbgs()};
#endif
};
} // end anonymous namespace

// TODO: handle annotations: [[OptimizableExtModuleAnnotation]]
void IMConstPropPass::runOnOperation() {
  auto circuit = getOperation();
  LLVM_DEBUG(
      { logger.startLine() << "IMConstProp : " << circuit.getName() << "\n"; });

  instanceGraph = &getAnalysis<InstanceGraph>();

  // Mark the input ports of public modules as being overdefined.
  for (auto module : circuit.getBodyBlock()->getOps<FModuleOp>()) {
    if (module.isPublic()) {
      markBlockExecutable(module.getBodyBlock());
      for (auto port : module.getBodyBlock()->getArguments())
        markOverdefined(port);
    }
  }

  // If a value changed lattice state then reprocess any of its users.
  while (!changedLatticeValueWorklist.empty()) {
    Value changedVal = changedLatticeValueWorklist.pop_back_val();
    for (Operation *user : changedVal.getUsers()) {
      if (isBlockExecutable(user->getBlock()))
        visitOperation(user);
    }
  }

  // Rewrite any constants in the modules.
  mlir::parallelForEach(circuit.getContext(),
                        circuit.getBodyBlock()->getOps<FModuleOp>(),
                        [&](auto op) { rewriteModuleBody(op); });

  // Clean up our state for next time.
  instanceGraph = nullptr;
  latticeValues.clear();
  executableBlocks.clear();
  resultPortToInstanceResultMapping.clear();
}

/// Return the lattice value for the specified SSA value, extended to the width
/// of the specified destType.  If allowTruncation is true, then this allows
/// truncating the lattice value to the specified type.
LatticeValue IMConstPropPass::getExtendedLatticeValue(Value value,
                                                      FIRRTLBaseType destType,
                                                      bool allowTruncation) {
  // If 'value' hasn't been computed yet, then it is unknown.
  auto it = latticeValues.find(value);
  if (it == latticeValues.end())
    return LatticeValue();

  auto result = it->second;
  // Unknown/overdefined stay whatever they are.
  if (result.isUnknown() || result.isOverdefined() || result.isUnwritten())
    return result;

  auto constant = result.getConstant();

  // If this is a BoolAttr then we are dealing with a special constant.
  if (auto boolAttr = constant.dyn_cast<BoolAttr>()) {
    // No extOrTrunc necessary for clock or reset types.
    return LatticeValue(boolAttr);
  }

  // If destType is wider than the source constant type, extend it.
  auto resultConstant = result.getConstant().getAPSInt();
  auto destWidth = destType.getBitWidthOrSentinel();
  if (destWidth == -1) // We don't support unknown width FIRRTL.
    return LatticeValue::getOverdefined();
  if (resultConstant.getBitWidth() == (unsigned)destWidth)
    return result; // Already the right width, we're done.

  // Otherwise, extend the constant using the signedness of the source.
  resultConstant = extOrTruncZeroWidth(resultConstant, destWidth);
  return LatticeValue(IntegerAttr::get(destType.getContext(), resultConstant));
}

/// Mark a block executable if it isn't already.  This does an initial scan of
/// the block, processing nullary operations like wires, instances, and
/// constants that only get processed once.
void IMConstPropPass::markBlockExecutable(Block *block) {
  if (!executableBlocks.insert(block).second)
    return; // Already executable.

  // Mark block arguments, which are module ports, with don't touch as
  // overdefined.
  for (auto ba : block->getArguments())
    if (hasDontTouch(ba))
      markOverdefined(ba);

  for (auto &op : *block) {
    // Handle each of the special operations in the firrtl dialect.
    if (isWireOrReg(&op))
      markWireOrRegOp(&op);
    else if (isAggregate(&op))
      markOverdefined(op.getResult(0));
    else if (auto constant = dyn_cast<ConstantOp>(op))
      markConstantOp(constant);
    else if (auto specialConstant = dyn_cast<SpecialConstantOp>(op))
      markSpecialConstantOp(specialConstant);
    else if (auto invalid = dyn_cast<InvalidValueOp>(op))
      markInvalidValueOp(invalid);
    else if (auto instance = dyn_cast<InstanceOp>(op))
      markInstanceOp(instance);
    else if (auto mem = dyn_cast<MemOp>(op))
      markMemOp(mem);
    else if (auto cast = dyn_cast<mlir::UnrealizedConversionCastOp>(op))
      for (auto result : cast.getResults())
        markOverdefined(result);
    else if (auto verbatim = dyn_cast<VerbatimExprOp>(op))
      markOverdefined(verbatim.getResult());
    else if (auto verbatim = dyn_cast<VerbatimWireOp>(op))
      markOverdefined(verbatim.getResult());
  }
}

void IMConstPropPass::markWireOrRegOp(Operation *wireOrReg) {
  // If the wire/reg/node has a non-ground type, then it is too complex for us
  // to handle, mark it as overdefined.
  // TODO: Eventually add a field-sensitive model.
  auto resultValue = wireOrReg->getResult(0);
  auto type = resultValue.getType().dyn_cast<FIRRTLBaseType>();
  if (!type || !type.getPassiveType().isGround())
    return markOverdefined(resultValue);

  if (hasDontTouch(wireOrReg))
    return markOverdefined(resultValue);

  // Otherwise, this starts out as UnWritten and is upgraded by connects.
  markUnwritten(resultValue);
}

void IMConstPropPass::markMemOp(MemOp mem) {
  for (auto result : mem.getResults())
    markOverdefined(result);
}

void IMConstPropPass::markConstantOp(ConstantOp constant) {
  mergeLatticeValue(constant, LatticeValue(constant.getValueAttr()));
}

void IMConstPropPass::markSpecialConstantOp(SpecialConstantOp specialConstant) {
  mergeLatticeValue(specialConstant,
                    LatticeValue(specialConstant.getValueAttr()));
}

void IMConstPropPass::markInvalidValueOp(InvalidValueOp invalid) {
  markOverdefined(invalid.getResult());
}

/// Instances have no operands, so they are visited exactly once when their
/// enclosing block is marked live.  This sets up the def-use edges for ports.
void IMConstPropPass::markInstanceOp(InstanceOp instance) {
  // Get the module being reference or a null pointer if this is an extmodule.
  Operation *op = instanceGraph->getReferencedModule(instance);

  // If this is an extmodule, just remember that any results and inouts are
  // overdefined.
  if (!isa<FModuleOp>(op)) {
    auto module = dyn_cast<FModuleLike>(op);
    for (size_t resultNo = 0, e = instance.getNumResults(); resultNo != e;
         ++resultNo) {
      auto portVal = instance.getResult(resultNo);
      // If this is an input to the extmodule, we can ignore it.
      if (module.getPortDirection(resultNo) == Direction::In)
        continue;

      // Otherwise this is a result from it or an inout, mark it as overdefined.
      markOverdefined(portVal);
    }
    return;
  }

  // Otherwise this is a defined module.
  auto fModule = cast<FModuleOp>(op);
  markBlockExecutable(fModule.getBodyBlock());

  // Ok, it is a normal internal module reference.  Populate
  // resultPortToInstanceResultMapping, and forward any already-computed values.
  for (size_t resultNo = 0, e = instance.getNumResults(); resultNo != e;
       ++resultNo) {
    auto instancePortVal = instance.getResult(resultNo);
    // If this is an input to the instance, it will
    // get handled when any connects to it are processed.
    if (fModule.getPortDirection(resultNo) == Direction::In)
      continue;
    // We only support simple values so far.
    auto portType = instancePortVal.getType().dyn_cast<FIRRTLBaseType>();
    if (portType && !portType.isGround()) {
      // TODO: Add field sensitivity.
      markOverdefined(instancePortVal);
      continue;
    }

    // Otherwise we have a result from the instance.  We need to forward results
    // from the body to this instance result's SSA value, so remember it.
    BlockArgument modulePortVal = fModule.getArgument(resultNo);

    resultPortToInstanceResultMapping[modulePortVal].push_back(instancePortVal);

    // If there is already a value known for modulePortVal make sure to forward
    // it here.
    mergeLatticeValue(instancePortVal, modulePortVal);
  }
}

void IMConstPropPass::visitConnectLike(FConnectLike connect) {
  // Mark foreign types as overdefined.
  auto destTypeFIRRTL = connect.getDest().getType().dyn_cast<FIRRTLType>();
  if (!destTypeFIRRTL) {
    markOverdefined(connect.getSrc());
    return markOverdefined(connect.getDest());
  }
  auto destType = getBaseType(destTypeFIRRTL).getPassiveType();

  // Handle implicit extensions.
  auto srcValue = getExtendedLatticeValue(connect.getSrc(), destType);
  if (srcValue.isUnknown())
    return;

  // Driving result ports propagates the value to each instance using the
  // module.
  if (auto blockArg = connect.getDest().dyn_cast<BlockArgument>()) {
    for (auto userOfResultPort : resultPortToInstanceResultMapping[blockArg])
      mergeLatticeValue(userOfResultPort, srcValue);
    // Output ports are wire-like and may have users.
    return mergeLatticeValue(connect.getDest(), srcValue);
  }

  auto dest = connect.getDest().cast<mlir::OpResult>();

  // For wires and registers, we drive the value of the wire itself, which
  // automatically propagates to users.
  if (isWireOrReg(dest.getOwner()))
    return mergeLatticeValue(connect.getDest(), srcValue);

  // Driving an instance argument port drives the corresponding argument of the
  // referenced module.
  if (auto instance = dest.getDefiningOp<InstanceOp>()) {
    // Update the dest, when its an instance op.
    mergeLatticeValue(connect.getDest(), srcValue);
    auto module =
        dyn_cast<FModuleOp>(*instanceGraph->getReferencedModule(instance));
    if (!module)
      return;

    BlockArgument modulePortVal = module.getArgument(dest.getResultNumber());
    return mergeLatticeValue(modulePortVal, srcValue);
  }

  // Driving a memory result is ignored because these are always treated as
  // overdefined.
  if (auto subfield = dest.getDefiningOp<SubfieldOp>()) {
    if (subfield.getOperand().getDefiningOp<MemOp>())
      return;
  }

  // Skip if the dest is an aggregate value. Aggregate values are firstly marked
  // overdefined.
  if (isAggregate(dest.getOwner()))
    return;

  connect.emitError("connectlike operation unhandled by IMConstProp")
          .attachNote(connect.getDest().getLoc())
      << "connect destination is here";
}

void IMConstPropPass::visitRegResetOp(RegResetOp regReset) {
  // If the reg has a non-ground type, then it is too complex for us to handle,
  // mark it as overdefined.
  // TODO: Eventually add a field-sensitive model.
  if (!regReset.getType().getPassiveType().isGround())
    return markOverdefined(regReset);

  // The reset value may be known - if so, merge it in if the enable is greater
  // than invalid.
  auto srcValue =
      getExtendedLatticeValue(regReset.getResetValue(), regReset.getType(),
                              /*allowTruncation=*/true);
  auto enable = getExtendedLatticeValue(regReset.getResetSignal(),
                                        regReset.getResetSignal().getType(),
                                        /*allowTruncation=*/true);
  if (enable.isOverdefined() ||
      (enable.isConstant() && !enable.getConstant().getValue().isZero()))
    mergeLatticeValue(regReset, srcValue);
}

void IMConstPropPass::visitRefSend(RefSendOp send) {
  // Send connects the base value (source) to the result (dest).
  return mergeLatticeValue(send.getResult(), send.getBase());
}

void IMConstPropPass::visitRefResolve(RefResolveOp resolve) {
  // Resolve connects the ref value (source) to result (dest).
  // If writes are ever supported, this will need to work differently!
  return mergeLatticeValue(resolve.getResult(), resolve.getRef());
}

void IMConstPropPass::visitNode(NodeOp node) {
  // Nodes don't fold if they have interesting names, but they should still
  // propagate values.
  if (hasDontTouch(node.getResult()) ||
      (node.getAnnotationsAttr() && !node.getAnnotationsAttr().empty()))
    return markOverdefined(node);

  return mergeLatticeValue(node.getResult(), node.getInput());
}

/// This method is invoked when an operand of the specified op changes its
/// lattice value state and when the block containing the operation is first
/// noticed as being alive.
///
/// This should update the lattice value state for any result values.
///
void IMConstPropPass::visitOperation(Operation *op) {
  // If this is a operation with special handling, handle it specially.
  if (auto connectLikeOp = dyn_cast<FConnectLike>(op))
    return visitConnectLike(connectLikeOp);
  if (auto regResetOp = dyn_cast<RegResetOp>(op))
    return visitRegResetOp(regResetOp);
  if (auto sendOp = dyn_cast<RefSendOp>(op))
    return visitRefSend(sendOp);
  if (auto resolveOp = dyn_cast<RefResolveOp>(op))
    return visitRefResolve(resolveOp);
  if (auto nodeOp = dyn_cast<NodeOp>(op))
    return visitNode(nodeOp);

  // The clock operand of regop changing doesn't change its result value.
  if (isa<RegOp>(op))
    return;
  // TODO: Handle 'when' operations.

  // If all of the results of this operation are already overdefined (or if
  // there are no results) then bail out early: we've converged.
  auto isOverdefinedFn = [&](Value value) { return isOverdefined(value); };
  if (llvm::all_of(op->getResults(), isOverdefinedFn))
    return;

  // Collect all of the constant operands feeding into this operation. If any
  // are not ready to be resolved, bail out and wait for them to resolve.
  SmallVector<Attribute, 8> operandConstants;
  operandConstants.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    auto &operandLattice = latticeValues[operand];

    // If the operand is an unknown value, then we generally don't want to
    // process it - we want to wait until the value is resolved to by the SCCP
    // algorithm.
    if (operandLattice.isUnknown())
      return;

    // Otherwise, it must be constant, invalid, or overdefined.  Translate them
    // into attributes that the fold hook can look at.
    if (operandLattice.isConstant())
      operandConstants.push_back(operandLattice.getValue());
    else
      operandConstants.push_back({});
  }

  // Simulate the result of folding this operation to a constant. If folding
  // fails or was an in-place fold, mark the results as overdefined.
  SmallVector<OpFoldResult, 8> foldResults;
  foldResults.reserve(op->getNumResults());
  if (failed(op->fold(operandConstants, foldResults))) {
    LLVM_DEBUG({
      logger.startLine() << "Folding Failed operation : '" << op->getName()
                         << "\n";
      op->dump();
    });
    for (auto value : op->getResults())
      markOverdefined(value);
    return;
  }

  LLVM_DEBUG({
    logger.getOStream() << "\n";
    logger.startLine() << "Folding operation : '" << op->getName() << "\n";
    op->dump();
    logger.getOStream() << "( ";
    for (auto cst : operandConstants)
      if (!cst)
        logger.getOStream() << "{} ";
      else
        logger.getOStream() << cst << " ";
    logger.unindent();
    logger.getOStream() << ") -> { ";
    logger.indent();
    for (auto &r : foldResults) {
      logger.getOStream() << r << " ";
    }
    logger.unindent();
    logger.getOStream() << "}\n";
  });

  // Fold functions in general are allowed to do in-place updates, but FIRRTL
  // does not do this and supporting it costs more.
  assert(!foldResults.empty() &&
         "FIRRTL fold functions shouldn't do in-place updates!");

  // Merge the fold results into the lattice for this operation.
  assert(foldResults.size() == op->getNumResults() && "invalid result size");
  for (unsigned i = 0, e = foldResults.size(); i != e; ++i) {
    // Merge in the result of the fold, either a constant or a value.
    LatticeValue resultLattice;
    OpFoldResult foldResult = foldResults[i];
    if (Attribute foldAttr = foldResult.dyn_cast<Attribute>()) {
      if (auto intAttr = foldAttr.dyn_cast<IntegerAttr>())
        resultLattice = LatticeValue(intAttr);
      else // Treat non integer constants as overdefined.
        resultLattice = LatticeValue::getOverdefined();
    } else { // Folding to an operand results in its value.
      resultLattice = latticeValues[foldResult.get<Value>()];
    }

    // We do not "merge" the lattice value in, we set it.  This is because the
    // fold functions can produce different values over time, e.g. in the
    // presence of InvalidValue operands that get resolved to other constants.
    setLatticeValue(op->getResult(i), resultLattice);
  }
}

void IMConstPropPass::rewriteModuleBody(FModuleOp module) {
  auto *body = module.getBodyBlock();
  // If a module is unreachable, just ignore it.
  if (!executableBlocks.count(body))
    return;

  auto builder = OpBuilder::atBlockBegin(body);

  // Unique constants per <Const,Type> pair, inserted at entry
  DenseMap<std::pair<Attribute, Type>, Operation *> constPool;
  auto getConst = [&](Attribute constantValue, Type type, Location loc) {
    auto constIt = constPool.find({constantValue, type});
    if (constIt != constPool.end()) {
      auto *cst = constIt->second;
      // Add location to the constant
      cst->setLoc(builder.getFusedLoc(cst->getLoc(), loc));
      return cst->getResult(0);
    }
    auto savedIP = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(body);
    auto *cst = module->getDialect()->materializeConstant(
        builder, constantValue, type, loc);
    builder.restoreInsertionPoint(savedIP);
    assert(cst && "all FIRRTL constants can be materialized");
    constPool.insert({{constantValue, type}, cst});
    return cst->getResult(0);
  };

  // If the lattice value for the specified value is a constant update it and
  // return true.  Otherwise return false.
  auto replaceValueIfPossible = [&](Value value) -> bool {
    // Lambda to replace all uses of this value a replacement, unless this is
    // the destination of a connect.  We leave connects alone to avoid upsetting
    // flow, i.e., to avoid trying to connect to a constant.
    auto replaceIfNotConnect = [&value](Value replacement) {
      value.replaceUsesWithIf(replacement, [](OpOperand &operand) {
        return !isa<FConnectLike>(operand.getOwner()) ||
               operand.getOperandNumber() != 0;
      });
    };

    auto it = latticeValues.find(value);
    if (it == latticeValues.end() || it->second.isOverdefined() ||
        it->second.isUnknown())
      return false;
    if (it->second.isUnwritten()) {
      if (auto reg = value.getDefiningOp<RegOp>()) {
        // Registers can get replaced with a unique (new) invalid value
        auto invalid =
            builder.create<InvalidValueOp>(reg.getLoc(), reg.getType());
        replaceIfNotConnect(invalid);
        return true;
      }
      return false;
    }

    // Cannot materialize constants for non-base types.
    if (!value.getType().isa<FIRRTLBaseType>())
      return false;

    auto cstValue =
        getConst(it->second.getValue(), value.getType(), value.getLoc());

    replaceIfNotConnect(cstValue);
    return true;
  };

  // Constant propagate any ports that are always constant.
  for (auto &port : body->getArguments())
    replaceValueIfPossible(port);

  // TODO: Walk 'when's preorder with `walk`.

  // Walk the IR bottom-up when folding.  We often fold entire chains of
  // operations into constants, which make the intermediate nodes dead.  Going
  // bottom up eliminates the users of the intermediate ops, allowing us to
  // aggressively delete them.
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*body))) {
    // Connects to values that we found to be constant can be dropped.
    if (auto connect = dyn_cast<FConnectLike>(op)) {
      if (auto *destOp = connect.getDest().getDefiningOp()) {
        if (isDeletableWireOrRegOrNode(destOp) &&
            !isOverdefined(connect.getDest())) {
          connect.erase();
          ++numErasedOp;
        }
      }
      continue;
    }

    // We only fold single-result ops and instances in practice, because they
    // are the expressions.
    if (op.getNumResults() != 1 && !isa<InstanceOp>(op))
      continue;

    // If this operation is already dead, then go ahead and remove it.
    if (op.use_empty() &&
        (wouldOpBeTriviallyDead(&op) || isDeletableWireOrRegOrNode(&op))) {
      LLVM_DEBUG({ logger.getOStream() << "Trivially dead : " << op << "\n"; });
      op.erase();
      continue;
    }

    // Don't "refold" constants.  TODO: Unique in the module entry block.
    if (isa<ConstantOp, SpecialConstantOp, InvalidValueOp>(op))
      continue;

    // If the op had any constants folded, replace them.
    builder.setInsertionPoint(&op);
    bool foldedAny = false;
    for (auto result : op.getResults())
      foldedAny |= replaceValueIfPossible(result);

    if (foldedAny)
      ++numFoldedOp;

    // If the operation folded to a constant then we can probably nuke it.
    if (foldedAny && op.use_empty() &&
        (wouldOpBeTriviallyDead(&op) || isDeletableWireOrRegOrNode(&op))) {
      LLVM_DEBUG({ logger.getOStream() << "Made dead : " << op << "\n"; });
      op.erase();
      ++numErasedOp;
      continue;
    }
  }
}

std::unique_ptr<mlir::Pass> circt::firrtl::createIMConstPropPass() {
  return std::make_unique<IMConstPropPass>();
}
