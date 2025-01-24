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

#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/APInt.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ScopedPrinter.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_IMCONSTPROP
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

#define DEBUG_TYPE "IMCP"

/// Return true if this is a wire or register.
static bool isWireOrReg(Operation *op) {
  return isa<WireOp, RegResetOp, RegOp>(op);
}

/// Return true if this is an aggregate indexer.
static bool isAggregate(Operation *op) {
  return isa<SubindexOp, SubaccessOp, SubfieldOp, OpenSubfieldOp,
             OpenSubindexOp, RefSubOp>(op);
}

// Return true if this forwards input to output.
// Implies has appropriate visit method that propagates changes.
static bool isNodeLike(Operation *op) {
  return isa<NodeOp, RefResolveOp, RefSendOp>(op);
}

/// Return true if this is a wire or register we're allowed to delete.
static bool isDeletableWireOrRegOrNode(Operation *op) {
  if (!isWireOrReg(op) && !isa<NodeOp>(op))
    return false;

  // Always allow deleting wires of probe-type.
  if (type_isa<RefType>(op->getResult(0).getType()))
    return true;

  // Otherwise, don't delete if has anything keeping it around or unknown.
  return AnnotationSet(op).empty() && !hasDontTouch(op) &&
         hasDroppableName(op) && !cast<Forceable>(op).isForceable();
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
  /*implicit*/ LatticeValue(StringAttr attr)
      : valueAndTag(attr, Kind::Constant) {}

  static LatticeValue getOverdefined() {
    LatticeValue result;
    result.markOverdefined();
    return result;
  }

  bool isUnknown() const { return valueAndTag.getInt() == Kind::Unknown; }
  bool isConstant() const { return valueAndTag.getInt() == Kind::Constant; }
  bool isOverdefined() const {
    return valueAndTag.getInt() == Kind::Overdefined;
  }

  /// Mark the lattice value as overdefined.
  void markOverdefined() {
    valueAndTag.setPointerAndInt(nullptr, Kind::Overdefined);
  }

  /// Mark the lattice value as constant.
  void markConstant(IntegerAttr value) {
    valueAndTag.setPointerAndInt(value, Kind::Constant);
  }

  /// If this lattice is constant or invalid value, return the attribute.
  /// Returns nullptr otherwise.
  Attribute getValue() const { return valueAndTag.getPointer(); }

  /// If this is in the constant state, return the attribute.
  Attribute getConstant() const {
    assert(isConstant());
    return getValue();
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
  /// kind.  The attribute is an IntegerAttr (or BoolAttr) or StringAttr.
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
  return os << "<" << lattice.getConstant() << ">";
}

namespace {
struct IMConstPropPass
    : public circt::firrtl::impl::IMConstPropBase<IMConstPropPass> {

  void runOnOperation() override;
  void rewriteModuleBody(FModuleOp module);

  /// Returns true if the given block is executable.
  bool isBlockExecutable(Block *block) const {
    return executableBlocks.count(block);
  }

  bool isOverdefined(FieldRef value) const {
    auto it = latticeValues.find(value);
    return it != latticeValues.end() && it->second.isOverdefined();
  }

  // Mark the given value as overdefined. If the value is an aggregate,
  // we mark all ground elements as overdefined.
  void markOverdefined(Value value) {
    FieldRef fieldRef = getOrCacheFieldRefFromValue(value);
    auto firrtlType = type_dyn_cast<FIRRTLType>(value.getType());
    if (!firrtlType || type_isa<PropertyType>(firrtlType)) {
      markOverdefined(fieldRef);
      return;
    }

    walkGroundTypes(firrtlType, [&](uint64_t fieldID, auto, auto) {
      markOverdefined(fieldRef.getSubField(fieldID));
    });
  }

  /// Mark the given value as overdefined. This means that we cannot refine a
  /// specific constant for this value.
  void markOverdefined(FieldRef value) {
    auto &entry = latticeValues[value];
    if (!entry.isOverdefined()) {
      LLVM_DEBUG({
        logger.getOStream()
            << "Setting overdefined : (" << getFieldName(value).first << ")\n";
      });
      entry.markOverdefined();
      changedLatticeValueWorklist.push_back(value);
    }
  }

  /// Merge information from the 'from' lattice value into value.  If it
  /// changes, then users of the value are added to the worklist for
  /// revisitation.
  void mergeLatticeValue(FieldRef value, LatticeValue &valueEntry,
                         LatticeValue source) {
    if (valueEntry.mergeIn(source)) {
      LLVM_DEBUG({
        logger.getOStream()
            << "Changed to " << valueEntry << " : (" << value << ")\n";
      });
      changedLatticeValueWorklist.push_back(value);
    }
  }

  void mergeLatticeValue(FieldRef value, LatticeValue source) {
    // Don't even do a map lookup if from has no info in it.
    if (source.isUnknown())
      return;
    mergeLatticeValue(value, latticeValues[value], source);
  }

  void mergeLatticeValue(FieldRef result, FieldRef from) {
    // If 'from' hasn't been computed yet, then it is unknown, don't do
    // anything.
    auto it = latticeValues.find(from);
    if (it == latticeValues.end())
      return;
    mergeLatticeValue(result, it->second);
  }

  void mergeLatticeValue(Value result, Value from) {
    FieldRef fieldRefFrom = getOrCacheFieldRefFromValue(from);
    FieldRef fieldRefResult = getOrCacheFieldRefFromValue(result);
    if (!type_isa<FIRRTLType>(result.getType()))
      return mergeLatticeValue(fieldRefResult, fieldRefFrom);
    // Special-handle PropertyType's, walkGroundType's doesn't support.
    if (type_isa<PropertyType>(result.getType()))
      return mergeLatticeValue(fieldRefResult, fieldRefFrom);
    walkGroundTypes(type_cast<FIRRTLType>(result.getType()),
                    [&](uint64_t fieldID, auto, auto) {
                      mergeLatticeValue(fieldRefResult.getSubField(fieldID),
                                        fieldRefFrom.getSubField(fieldID));
                    });
  }

  /// setLatticeValue - This is used when a new LatticeValue is computed for
  /// the result of the specified value that replaces any previous knowledge,
  /// e.g. because a fold() function on an op returned a new thing.  This should
  /// not be used on operations that have multiple contributors to it, e.g.
  /// wires or ports.
  void setLatticeValue(FieldRef value, LatticeValue source) {
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

  // This function returns a field ref of the given value. This function caches
  // the result to avoid extra IR traversal if the value is an aggregate
  // element.
  FieldRef getOrCacheFieldRefFromValue(Value value) {
    if (!value.getDefiningOp() || !isAggregate(value.getDefiningOp()))
      return FieldRef(value, 0);
    auto &fieldRef = valueToFieldRef[value];
    if (fieldRef)
      return fieldRef;
    return fieldRef = getFieldRefFromValue(value);
  }

  /// Return the lattice value for the specified SSA value, extended to the
  /// width of the specified destType.  If allowTruncation is true, then this
  /// allows truncating the lattice value to the specified type.
  LatticeValue getExtendedLatticeValue(FieldRef value, FIRRTLType destType,
                                       bool allowTruncation = false);

  /// Mark the given block as executable.
  void markBlockExecutable(Block *block);
  void markWireOp(WireOp wireOrReg);
  void markMemOp(MemOp mem);

  void markInvalidValueOp(InvalidValueOp invalid);
  void markAggregateConstantOp(AggregateConstantOp constant);
  void markInstanceOp(InstanceOp instance);
  void markObjectOp(ObjectOp object);
  template <typename OpTy>
  void markConstantValueOp(OpTy op);

  void visitConnectLike(FConnectLike connect, FieldRef changedFieldRef);
  void visitRefSend(RefSendOp send, FieldRef changedFieldRef);
  void visitRefResolve(RefResolveOp resolve, FieldRef changedFieldRef);
  void mergeOnlyChangedLatticeValue(Value dest, Value src,
                                    FieldRef changedFieldRef);
  void visitNode(NodeOp node, FieldRef changedFieldRef);
  void visitOperation(Operation *op, FieldRef changedFieldRef);

private:
  /// This is the current instance graph for the Circuit.
  InstanceGraph *instanceGraph = nullptr;

  /// This keeps track of the current state of each tracked value.
  DenseMap<FieldRef, LatticeValue> latticeValues;

  /// The set of blocks that are known to execute, or are intrinsically live.
  SmallPtrSet<Block *, 16> executableBlocks;

  /// A worklist of values whose LatticeValue recently changed, indicating the
  /// users need to be reprocessed.
  SmallVector<FieldRef, 64> changedLatticeValueWorklist;

  // A map to give operations to be reprocessed.
  DenseMap<FieldRef, llvm::TinyPtrVector<Operation *>> fieldRefToUsers;

  // A map to cache results of getFieldRefFromValue since it's costly traverse
  // the IR.
  llvm::DenseMap<Value, FieldRef> valueToFieldRef;

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
    FieldRef changedFieldRef = changedLatticeValueWorklist.pop_back_val();
    for (Operation *user : fieldRefToUsers[changedFieldRef]) {
      if (isBlockExecutable(user->getBlock()))
        visitOperation(user, changedFieldRef);
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
  assert(changedLatticeValueWorklist.empty());
  fieldRefToUsers.clear();
  valueToFieldRef.clear();
  resultPortToInstanceResultMapping.clear();
}

/// Return the lattice value for the specified SSA value, extended to the width
/// of the specified destType.  If allowTruncation is true, then this allows
/// truncating the lattice value to the specified type.
LatticeValue IMConstPropPass::getExtendedLatticeValue(FieldRef value,
                                                      FIRRTLType destType,
                                                      bool allowTruncation) {
  // If 'value' hasn't been computed yet, then it is unknown.
  auto it = latticeValues.find(value);
  if (it == latticeValues.end())
    return LatticeValue();

  auto result = it->second;
  // Unknown/overdefined stay whatever they are.
  if (result.isUnknown() || result.isOverdefined())
    return result;

  // No extOrTrunc for property types.  Return what we have.
  if (isa<PropertyType>(destType))
    return result;

  auto constant = result.getConstant();

  // If not property, only support integers.
  auto intAttr = dyn_cast<IntegerAttr>(constant);
  assert(intAttr && "unsupported lattice attribute kind");
  if (!intAttr)
    return result;

  // No extOrTrunc necessary for bools.
  if (auto boolAttr = dyn_cast<BoolAttr>(intAttr))
    return result;

  // Non-base (or non-ref) types are overdefined.
  auto baseType = getBaseType(destType);
  if (!baseType)
    return LatticeValue::getOverdefined();

  // If destType is wider than the source constant type, extend it.
  auto resultConstant = intAttr.getAPSInt();
  auto destWidth = baseType.getBitWidthOrSentinel();
  if (destWidth == -1) // We don't support unknown width FIRRTL.
    return LatticeValue::getOverdefined();
  if (resultConstant.getBitWidth() == (unsigned)destWidth)
    return result; // Already the right width, we're done.

  // Otherwise, extend the constant using the signedness of the source.
  resultConstant = extOrTruncZeroWidth(resultConstant, destWidth);
  return LatticeValue(IntegerAttr::get(destType.getContext(), resultConstant));
}

// NOLINTBEGIN(misc-no-recursion)
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
    TypeSwitch<Operation *>(&op)
        .Case<RegOp, RegResetOp>(
            [&](auto reg) { markOverdefined(op.getResult(0)); })
        .Case<WireOp>([&](auto wire) { markWireOp(wire); })
        .Case<ConstantOp, SpecialConstantOp, StringConstantOp,
              FIntegerConstantOp, BoolConstantOp>(
            [&](auto constOp) { markConstantValueOp(constOp); })
        .Case<AggregateConstantOp>(
            [&](auto aggConstOp) { markAggregateConstantOp(aggConstOp); })
        .Case<InvalidValueOp>(
            [&](auto invalid) { markInvalidValueOp(invalid); })
        .Case<InstanceOp>([&](auto instance) { markInstanceOp(instance); })
        .Case<ObjectOp>([&](auto obj) { markObjectOp(obj); })
        .Case<MemOp>([&](auto mem) { markMemOp(mem); })
        .Case<LayerBlockOp>(
            [&](auto layer) { markBlockExecutable(layer.getBody(0)); })
        .Default([&](auto _) {
          if (isa<mlir::UnrealizedConversionCastOp, VerbatimExprOp,
                  VerbatimWireOp, SubaccessOp>(op) ||
              op.getNumOperands() == 0) {
            // Mark operations whose results cannot be tracked as overdefined.
            // Mark unhandled operations with no operand as well since otherwise
            // they will remain unknown states until the end.
            for (auto result : op.getResults())
              markOverdefined(result);
          } else if (
              // Operations that are handled when propagating values, or chasing
              // indexing.
              !isAggregate(&op) && !isNodeLike(&op) && op.getNumResults() > 0) {
            // If an unknown operation has an aggregate operand, mark results as
            // overdefined since we cannot track the dataflow. Similarly if the
            // operations create aggregate values, we mark them overdefined.

            // TODO: We should handle aggregate operations such as
            // vector_create, bundle_create or vector operations.

            bool hasAggregateOperand =
                llvm::any_of(op.getOperandTypes(), [](Type type) {
                  return type_isa<FVectorType, BundleType>(type);
                });

            for (auto result : op.getResults())
              if (hasAggregateOperand ||
                  type_isa<FVectorType, BundleType>(result.getType()))
                markOverdefined(result);
          }
        });

    // This tracks a dependency from field refs to operations which need
    // to be added to worklist when lattice values change.
    if (!isAggregate(&op)) {
      for (auto operand : op.getOperands()) {
        auto fieldRef = getOrCacheFieldRefFromValue(operand);
        auto firrtlType = type_dyn_cast<FIRRTLType>(operand.getType());
        if (!firrtlType)
          continue;
        // Special-handle PropertyType's, walkGroundTypes doesn't support.
        if (type_isa<PropertyType>(firrtlType)) {
          fieldRefToUsers[fieldRef].push_back(&op);
          continue;
        }
        walkGroundTypes(firrtlType, [&](uint64_t fieldID, auto type, auto) {
          fieldRefToUsers[fieldRef.getSubField(fieldID)].push_back(&op);
        });
      }
    }
  }
}
// NOLINTEND(misc-no-recursion)

void IMConstPropPass::markWireOp(WireOp wire) {
  auto type = type_dyn_cast<FIRRTLType>(wire.getResult().getType());
  if (!type || hasDontTouch(wire.getResult()) || wire.isForceable()) {
    for (auto result : wire.getResults())
      markOverdefined(result);
    return;
  }

  // Otherwise, this starts out as unknown and is upgraded by connects.
}

void IMConstPropPass::markMemOp(MemOp mem) {
  for (auto result : mem.getResults())
    markOverdefined(result);
}

template <typename OpTy>
void IMConstPropPass::markConstantValueOp(OpTy op) {
  mergeLatticeValue(getOrCacheFieldRefFromValue(op),
                    LatticeValue(op.getValueAttr()));
}

void IMConstPropPass::markAggregateConstantOp(AggregateConstantOp constant) {
  walkGroundTypes(constant.getType(), [&](uint64_t fieldID, auto, auto) {
    mergeLatticeValue(FieldRef(constant, fieldID),
                      LatticeValue(cast<IntegerAttr>(
                          constant.getAttributeFromFieldID(fieldID))));
  });
}

void IMConstPropPass::markInvalidValueOp(InvalidValueOp invalid) {
  markOverdefined(invalid.getResult());
}

/// Instances have no operands, so they are visited exactly once when their
/// enclosing block is marked live.  This sets up the def-use edges for ports.
void IMConstPropPass::markInstanceOp(InstanceOp instance) {
  // Get the module being reference or a null pointer if this is an extmodule.
  Operation *op = instance.getReferencedModule(*instanceGraph);

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

    // Otherwise we have a result from the instance.  We need to forward results
    // from the body to this instance result's SSA value, so remember it.
    BlockArgument modulePortVal = fModule.getArgument(resultNo);

    resultPortToInstanceResultMapping[modulePortVal].push_back(instancePortVal);

    // If there is already a value known for modulePortVal make sure to forward
    // it here.
    mergeLatticeValue(instancePortVal, modulePortVal);
  }
}

void IMConstPropPass::markObjectOp(ObjectOp obj) {
  // Mark overdefined for now, not supported.
  markOverdefined(obj);
}

static std::optional<uint64_t>
getFieldIDOffset(FieldRef changedFieldRef, Type connectionType,
                 FieldRef connectedValueFieldRef) {
  assert(!type_isa<RefType>(connectionType));
  if (changedFieldRef.getValue() != connectedValueFieldRef.getValue())
    return {};
  if (changedFieldRef.getFieldID() >= connectedValueFieldRef.getFieldID() &&
      changedFieldRef.getFieldID() <=
          hw::FieldIdImpl::getMaxFieldID(connectionType) +
              connectedValueFieldRef.getFieldID())
    return changedFieldRef.getFieldID() - connectedValueFieldRef.getFieldID();
  return {};
}

void IMConstPropPass::mergeOnlyChangedLatticeValue(Value dest, Value src,
                                                   FieldRef changedFieldRef) {

  // Operate on inner type for refs.
  auto destType = dest.getType();
  if (auto refType = type_dyn_cast<RefType>(destType))
    destType = refType.getType();

  if (!isa<FIRRTLType>(destType)) {
    // If the dest is not FIRRTL type, conservatively mark
    // all of them overdefined.
    markOverdefined(src);
    return markOverdefined(dest);
  }

  auto fieldRefSrc = getOrCacheFieldRefFromValue(src);
  auto fieldRefDest = getOrCacheFieldRefFromValue(dest);

  // If a changed field ref is included the source value, find an offset in the
  // connection.
  if (auto srcOffset = getFieldIDOffset(changedFieldRef, destType, fieldRefSrc))
    mergeLatticeValue(fieldRefDest.getSubField(*srcOffset),
                      fieldRefSrc.getSubField(*srcOffset));

  // If a changed field ref is included the dest value, find an offset in the
  // connection.
  if (auto destOffset =
          getFieldIDOffset(changedFieldRef, destType, fieldRefDest))
    mergeLatticeValue(fieldRefDest.getSubField(*destOffset),
                      fieldRefSrc.getSubField(*destOffset));
}

void IMConstPropPass::visitConnectLike(FConnectLike connect,
                                       FieldRef changedFieldRef) {
  // Operate on inner type for refs.
  auto destType = connect.getDest().getType();
  if (auto refType = type_dyn_cast<RefType>(destType))
    destType = refType.getType();

  // Mark foreign types as overdefined.
  if (!isa<FIRRTLType>(destType)) {
    markOverdefined(connect.getSrc());
    return markOverdefined(connect.getDest());
  }

  auto fieldRefSrc = getOrCacheFieldRefFromValue(connect.getSrc());
  auto fieldRefDest = getOrCacheFieldRefFromValue(connect.getDest());
  if (auto subaccess = fieldRefDest.getValue().getDefiningOp<SubaccessOp>()) {
    // If the destination is subaccess, we give up to precisely track
    // lattice values and mark entire aggregate as overdefined. This code
    // should be dead unless we stop lowering of subaccess in LowerTypes.
    Value parent = subaccess.getInput();
    while (parent.getDefiningOp() &&
           parent.getDefiningOp()->getNumOperands() > 0)
      parent = parent.getDefiningOp()->getOperand(0);
    return markOverdefined(parent);
  }

  auto propagateElementLattice = [&](uint64_t fieldID, FIRRTLType destType) {
    auto fieldRefDestConnected = fieldRefDest.getSubField(fieldID);
    assert(!firrtl::type_isa<FIRRTLBaseType>(destType) ||
           firrtl::type_cast<FIRRTLBaseType>(destType).isGround());

    // Handle implicit extensions.
    auto srcValue =
        getExtendedLatticeValue(fieldRefSrc.getSubField(fieldID), destType);
    if (srcValue.isUnknown())
      return;

    // Driving result ports propagates the value to each instance using the
    // module.
    if (auto blockArg = dyn_cast<BlockArgument>(fieldRefDest.getValue())) {
      for (auto userOfResultPort : resultPortToInstanceResultMapping[blockArg])
        mergeLatticeValue(
            FieldRef(userOfResultPort, fieldRefDestConnected.getFieldID()),
            srcValue);
      // Output ports are wire-like and may have users.
      return mergeLatticeValue(fieldRefDestConnected, srcValue);
    }

    auto dest = cast<mlir::OpResult>(fieldRefDest.getValue());

    // For wires and registers, we drive the value of the wire itself, which
    // automatically propagates to users.
    if (isWireOrReg(dest.getOwner()))
      return mergeLatticeValue(fieldRefDestConnected, srcValue);

    // Driving an instance argument port drives the corresponding argument
    // of the referenced module.
    if (auto instance = dest.getDefiningOp<InstanceOp>()) {
      // Update the dest, when its an instance op.
      mergeLatticeValue(fieldRefDestConnected, srcValue);
      auto mod = instance.getReferencedModule<FModuleOp>(*instanceGraph);
      if (!mod)
        return;

      BlockArgument modulePortVal = mod.getArgument(dest.getResultNumber());

      return mergeLatticeValue(
          FieldRef(modulePortVal, fieldRefDestConnected.getFieldID()),
          srcValue);
    }

    // Driving a memory result is ignored because these are always treated
    // as overdefined.
    if (dest.getDefiningOp<MemOp>())
      return;

    // For now, don't support const prop into object fields.
    if (isa_and_nonnull<ObjectSubfieldOp>(dest.getDefiningOp()))
      return;

    connect.emitError("connectlike operation unhandled by IMConstProp")
            .attachNote(connect.getDest().getLoc())
        << "connect destination is here";
  };

  if (auto srcOffset = getFieldIDOffset(changedFieldRef, destType, fieldRefSrc))
    propagateElementLattice(
        *srcOffset,
        firrtl::type_cast<FIRRTLType>(
            hw::FieldIdImpl::getFinalTypeByFieldID(destType, *srcOffset)));

  if (auto relativeDest =
          getFieldIDOffset(changedFieldRef, destType, fieldRefDest))
    propagateElementLattice(
        *relativeDest,
        firrtl::type_cast<FIRRTLType>(
            hw::FieldIdImpl::getFinalTypeByFieldID(destType, *relativeDest)));
}

void IMConstPropPass::visitRefSend(RefSendOp send, FieldRef changedFieldRef) {
  // Send connects the base value (source) to the result (dest).
  return mergeOnlyChangedLatticeValue(send.getResult(), send.getBase(),
                                      changedFieldRef);
}

void IMConstPropPass::visitRefResolve(RefResolveOp resolve,
                                      FieldRef changedFieldRef) {
  // Resolve connects the ref value (source) to result (dest).
  // If writes are ever supported, this will need to work differently!
  return mergeOnlyChangedLatticeValue(resolve.getResult(), resolve.getRef(),
                                      changedFieldRef);
}

void IMConstPropPass::visitNode(NodeOp node, FieldRef changedFieldRef) {
  if (hasDontTouch(node.getResult()) || node.isForceable()) {
    for (auto result : node.getResults())
      markOverdefined(result);
    return;
  }

  return mergeOnlyChangedLatticeValue(node.getResult(), node.getInput(),
                                      changedFieldRef);
}

/// This method is invoked when an operand of the specified op changes its
/// lattice value state and when the block containing the operation is first
/// noticed as being alive.
///
/// This should update the lattice value state for any result values.
///
void IMConstPropPass::visitOperation(Operation *op, FieldRef changedField) {
  // If this is a operation with special handling, handle it specially.
  if (auto connectLikeOp = dyn_cast<FConnectLike>(op))
    return visitConnectLike(connectLikeOp, changedField);
  if (auto sendOp = dyn_cast<RefSendOp>(op))
    return visitRefSend(sendOp, changedField);
  if (auto resolveOp = dyn_cast<RefResolveOp>(op))
    return visitRefResolve(resolveOp, changedField);
  if (auto nodeOp = dyn_cast<NodeOp>(op))
    return visitNode(nodeOp, changedField);

  // The clock operand of regop changing doesn't change its result value.  All
  // other registers are over-defined. Aggregate operations also doesn't change
  // its result value.
  if (isa<RegOp, RegResetOp>(op) || isAggregate(op))
    return;
  // TODO: Handle 'when' operations.

  // If all of the results of this operation are already overdefined (or if
  // there are no results) then bail out early: we've converged.
  auto isOverdefinedFn = [&](Value value) {
    return isOverdefined(getOrCacheFieldRefFromValue(value));
  };
  if (llvm::all_of(op->getResults(), isOverdefinedFn))
    return;

  // To prevent regressions, mark values as overdefined when they are defined
  // by operations with a large number of operands.
  if (op->getNumOperands() > 128) {
    for (auto value : op->getResults())
      markOverdefined(value);
    return;
  }

  // Collect all of the constant operands feeding into this operation. If any
  // are not ready to be resolved, bail out and wait for them to resolve.
  SmallVector<Attribute, 8> operandConstants;
  operandConstants.reserve(op->getNumOperands());
  bool hasUnknown = false;
  for (Value operand : op->getOperands()) {

    auto &operandLattice = latticeValues[getOrCacheFieldRefFromValue(operand)];

    // If the operand is an unknown value, then we generally don't want to
    // process it - we want to wait until the value is resolved to by the SCCP
    // algorithm.
    if (operandLattice.isUnknown())
      hasUnknown = true;

    // Otherwise, it must be constant, invalid, or overdefined.  Translate them
    // into attributes that the fold hook can look at.
    if (operandLattice.isConstant())
      operandConstants.push_back(operandLattice.getValue());
    else
      operandConstants.push_back({});
  }

  // Simulate the result of folding this operation to a constant. If folding
  // fails mark the results as overdefined.
  SmallVector<OpFoldResult, 8> foldResults;
  foldResults.reserve(op->getNumResults());
  if (failed(op->fold(operandConstants, foldResults))) {
    LLVM_DEBUG({
      logger.startLine() << "Folding Failed operation : '" << op->getName()
                         << "\n";
      op->dump();
    });
    // If we had unknown arguments, hold off on overdefining
    if (!hasUnknown)
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

  // If the folding was in-place, keep going.  This is surprising, but since
  // only folder that will do in-place updates is the commutative folder, we
  // aren't going to stop.  We don't update the results, since they didn't
  // change, the op just got shuffled around.
  if (foldResults.empty())
    return visitOperation(op, changedField);

  // Merge the fold results into the lattice for this operation.
  assert(foldResults.size() == op->getNumResults() && "invalid result size");
  for (unsigned i = 0, e = foldResults.size(); i != e; ++i) {
    // Merge in the result of the fold, either a constant or a value.
    LatticeValue resultLattice;
    OpFoldResult foldResult = foldResults[i];
    if (Attribute foldAttr = dyn_cast<Attribute>(foldResult)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(foldAttr))
        resultLattice = LatticeValue(intAttr);
      else if (auto strAttr = dyn_cast<StringAttr>(foldAttr))
        resultLattice = LatticeValue(strAttr);
      else // Treat unsupported constants as overdefined.
        resultLattice = LatticeValue::getOverdefined();
    } else { // Folding to an operand results in its value.
      resultLattice =
          latticeValues[getOrCacheFieldRefFromValue(cast<Value>(foldResult))];
    }

    mergeLatticeValue(getOrCacheFieldRefFromValue(op->getResult(i)),
                      resultLattice);
  }
}

void IMConstPropPass::rewriteModuleBody(FModuleOp module) {
  auto *body = module.getBodyBlock();
  // If a module is unreachable, just ignore it.
  if (!executableBlocks.count(body))
    return;

  auto builder = OpBuilder::atBlockBegin(body);

  // Separate the constants we insert from the instructions we are folding and
  // processing. Leave these as-is until we're done.
  auto cursor = builder.create<firrtl::ConstantOp>(module.getLoc(), APSInt(1));
  builder.setInsertionPoint(cursor);

  // Unique constants per <Const,Type> pair, inserted at entry
  DenseMap<std::pair<Attribute, Type>, Operation *> constPool;

  std::function<Value(Attribute, Type, Location)> getConst =
      [&](Attribute constantValue, Type type, Location loc) -> Value {
    auto constIt = constPool.find({constantValue, type});
    if (constIt != constPool.end()) {
      auto *cst = constIt->second;
      // Add location to the constant
      cst->setLoc(builder.getFusedLoc({cst->getLoc(), loc}));
      return cst->getResult(0);
    }
    OpBuilder::InsertionGuard x(builder);
    builder.setInsertionPoint(cursor);

    // Materialize reftype "constants" by materializing the constant
    // and probing it.
    Operation *cst;
    if (auto refType = type_dyn_cast<RefType>(type)) {
      assert(!type_cast<RefType>(type).getForceable() &&
             "Attempting to materialize rwprobe of constant, shouldn't happen");
      auto inner = getConst(constantValue, refType.getType(), loc);
      assert(inner);
      cst = builder.create<RefSendOp>(loc, inner);
    } else
      cst = module->getDialect()->materializeConstant(builder, constantValue,
                                                      type, loc);
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

    // TODO: Replace entire aggregate.
    auto it = latticeValues.find(getFieldRefFromValue(value));
    if (it == latticeValues.end() || it->second.isOverdefined() ||
        it->second.isUnknown())
      return false;

    // Cannot materialize constants for certain types.
    // TODO: Let materializeConstant tell us what it supports instead of this.
    // Presently it asserts on unsupported combinations, so check this here.
    if (!type_isa<FIRRTLBaseType, RefType, FIntegerType, StringType, BoolType>(
            value.getType()))
      return false;

    auto cstValue =
        getConst(it->second.getValue(), value.getType(), value.getLoc());

    replaceIfNotConnect(cstValue);
    return true;
  };

  // Constant propagate any ports that are always constant.
  for (auto &port : body->getArguments())
    replaceValueIfPossible(port);

  // Walk the IR bottom-up when folding.  We often fold entire chains of
  // operations into constants, which make the intermediate nodes dead.  Going
  // bottom up eliminates the users of the intermediate ops, allowing us to
  // aggressively delete them.
  //
  // TODO: Handle WhenOps correctly.
  bool aboveCursor = false;
  module.walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
      [&](Operation *op) {
        auto dropIfDead = [&](Operation *op, const Twine &debugPrefix) {
          if (op->use_empty() &&
              (wouldOpBeTriviallyDead(op) || isDeletableWireOrRegOrNode(op))) {
            LLVM_DEBUG(
                { logger.getOStream() << debugPrefix << " : " << op << "\n"; });
            ++numErasedOp;
            op->erase();
            return true;
          }
          return false;
        };

        if (aboveCursor) {
          // Drop dead constants we materialized.
          dropIfDead(op, "Trivially dead materialized constant");
          return WalkResult::advance();
        }
        // Stop once hit the generated constants.
        if (op == cursor) {
          cursor.erase();
          aboveCursor = true;
          return WalkResult::advance();
        }

        // Connects to values that we found to be constant can be dropped.
        if (auto connect = dyn_cast<FConnectLike>(op)) {
          if (auto *destOp = connect.getDest().getDefiningOp()) {
            auto fieldRef = getOrCacheFieldRefFromValue(connect.getDest());
            // Don't remove a field-level connection even if the src value is
            // constant. If other elements of the aggregate value are not
            // constant, the aggregate value cannot be replaced. We can forward
            // the constant to its users, so IMDCE (or SV/HW canonicalizer)
            // should remove the aggregate if entire aggregate is dead.
            auto type = type_dyn_cast<FIRRTLType>(connect.getDest().getType());
            if (!type)
              return WalkResult::advance();
            auto baseType = type_dyn_cast<FIRRTLBaseType>(type);
            if (baseType && !baseType.isGround())
              return WalkResult::advance();
            if (isDeletableWireOrRegOrNode(destOp) &&
                !isOverdefined(fieldRef)) {
              connect.erase();
              ++numErasedOp;
            }
          }
          return WalkResult::advance();
        }

        // We only fold single-result ops and instances in practice, because
        // they are the expressions.
        if (op->getNumResults() != 1 && !isa<InstanceOp>(op))
          return WalkResult::advance();

        // If this operation is already dead, then go ahead and remove it.
        if (dropIfDead(op, "Trivially dead"))
          return WalkResult::advance();

        // Don't "fold" constants (into equivalent), also because they
        // may have name hints we'd like to preserve.
        if (op->hasTrait<mlir::OpTrait::ConstantLike>())
          return WalkResult::advance();

        // If the op had any constants folded, replace them.
        builder.setInsertionPoint(op);
        bool foldedAny = false;
        for (auto result : op->getResults())
          foldedAny |= replaceValueIfPossible(result);

        if (foldedAny)
          ++numFoldedOp;

        // If the operation folded to a constant then we can probably nuke it.
        if (foldedAny && dropIfDead(op, "Made dead"))
          return WalkResult::advance();

        return WalkResult::advance();
      });
}

std::unique_ptr<mlir::Pass> circt::firrtl::createIMConstPropPass() {
  return std::make_unique<IMConstPropPass>();
}
