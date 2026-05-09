//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "circt/Support/UnusedOpPruner.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-combine-drives"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_COMBINEDRIVESPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;
using hw::HWModuleOp;
using llvm::SmallMapVector;
using llvm::SmallSetVector;
using llvm::SpecificBumpPtrAllocator;

/// Determine the number of elements in a type. This returns the number of bits
/// in an integer, the number of elements in an array, or the number of fields
/// in a struct. Returns zero for everything else.
static unsigned getLength(Type type) {
  return TypeSwitch<Type, unsigned>(cast<RefType>(type).getNestedType())
      .Case<IntegerType>([](auto type) { return type.getWidth(); })
      .Case<hw::ArrayType>([](auto type) { return type.getNumElements(); })
      .Case<hw::StructType>([](auto type) { return type.getElements().size(); })
      .Default([](auto) { return 0; });
}

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

namespace {
struct Signal;

/// A value representing a slice of a larger aggregate value. Does not track
/// that larger value directly. Instead this struct only tracks the offset and
/// length of the slice within that larger value.
struct ValueSlice {
  Value value;
  unsigned offset = 0;
  unsigned length = 0;
};

/// A drive assigning a slice of a larger signal. Does not track that larger
/// signal directly. Instead this struct only tracks the offset and length of
/// the slice in the signal that is being assigned.
struct DriveSlice {
  /// The drive op assigning a value. This may be null if no current drive op
  /// exists.
  DriveOp op;
  /// The value being assigned. Usually this is equal to `op.getValue()`, but
  /// may hold something like a default signal value if this slice was created
  /// to fill a gap in drives and no drive op exists.
  Value value;
  /// The offset within the larger signal that is being assigned.
  unsigned offset = 0;
  /// The number of elements starting at the offset that are being assigned.
  unsigned length = 0;
};

/// A slice of a signal. Keeps a pointer to the full `Signal`, alongside the
/// offset and length of the elements within that signal. Operations like
/// `llhd.sig.extract` use this struct to track which exact bits of a signal are
/// being targeted.
///
/// Note the difference to `ValueSlice` and `DriveSlice`: this struct *directly*
/// tracks the signal being sliced, while the other two structs track the result
/// of the slicing, but not the signal being sliced directly.
struct SignalSlice {
  Signal *signal = nullptr;
  unsigned offset = 0;
  unsigned length = 0;

  explicit operator bool() const { return signal != nullptr; }
};

/// A signal that can be sliced and projected into. Arrays and structs track
/// their elements and fields as separate subsignals. Operations such as
/// `llhd.sig`, `llhd.sig.array_get`, and `llhd.sig.struct_extract` create new
/// `Signal`s, since they each represent an independent signal. Operations such
/// as `llhd.sig.extract` and `llhd.sig.array_slice` *do not* create new
/// `Signal`s; instead they simply adjust the offset and length of the
/// `SignalSlice` pointing to an existing signal. This is an important
/// distinction: operations that descend into subfields of an aggregate create
/// new `Signal`s corresponding to those subfields, while operations that merely
/// slice an aggregate into a smaller aggregate do not create new `Signal`s.
struct Signal {
  /// The SSA value representing the signal. This is how we first encountered
  /// this signal in the IR. The goal of the pass is to combine drives to any
  /// subsignals and slices into a single drive to this value.
  Value value;
  /// The parent aggregate signal that contains this signal.
  Signal *parent = nullptr;
  /// Index of the field within the parent.
  unsigned indexInParent = 0;
  /// The signals corresponding to individual subfields of this signal, if this
  /// signal is an aggregate.
  SmallVector<Signal *> subsignals;
  /// The SSA values representing this signal or slices of it. This likely also
  /// contains `value`. The slice's `value` field corresponds to the result of
  /// slicing this signal. The offset and length are referring to elements of
  /// this signal.
  SmallVector<ValueSlice> slices;
  /// The drives that assign a single value to the entire signal. There may be
  /// multiple drives with different delay and enable operands.
  SmallVector<DriveOp, 2> completeDrives;

  /// Create a root signal.
  explicit Signal(Value root) : value(root) {}
  /// Create a subsignal representing a single field of a parent signal.
  Signal(Value value, Signal *parent, unsigned indexInParent)
      : value(value), parent(parent), indexInParent(indexInParent) {}
};

/// Tracks projections within a module and combines multiple drives to aggregate
/// fields into single drives of the entire aggregate value.
struct ModuleContext {
  ModuleContext(HWModuleOp moduleOp) : moduleOp(moduleOp) {}

  // Utilities to trace the result of a projection op back to the root signal
  // being projected into.
  SignalSlice traceProjection(Value value);
  SignalSlice traceProjectionImpl(Value value);
  Signal *internSignal(Value root);
  Signal *internSignal(Value value, Signal *parent, unsigned index);

  // Utilities to aggregate drives to a signal.
  void aggregateDrives(Signal &signal);
  void addDefaultDriveSlices(Signal &signal,
                             SmallVectorImpl<DriveSlice> &slices);
  void aggregateDriveSlices(Signal &signal, Value driveDelay, Value driveEnable,
                            ArrayRef<DriveSlice> slices);

  /// The module within which we are combining drives.
  HWModuleOp moduleOp;
  /// The signal slice targeted by each projection op in the module.
  DenseMap<Value, SignalSlice> projections;
  /// The root signals that have interesting projections targeting them.
  SmallVector<Signal *> rootSignals;
  /// Helper to clean up unused ops.
  UnusedOpPruner pruner;

private:
  using SignalKey = std::pair<PointerUnion<Value, Signal *>, unsigned>;
  SpecificBumpPtrAllocator<Signal> signalAlloc;
  DenseMap<SignalKey, Signal *> internedSignals;
};
} // namespace

/// Print a signal.
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const Signal &signal) {
  if (signal.parent)
    return os << *signal.parent << "[" << signal.indexInParent << "]";
  signal.value.printAsOperand(os, OpPrintingFlags().useLocalScope());
  return os;
}

/// Print a signal slice.
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os, SignalSlice slice) {
  if (!slice)
    return os << "<null-slice>";
  return os << *slice.signal << "[" << slice.offset << ".."
            << (slice.offset + slice.length) << "]";
}

//===----------------------------------------------------------------------===//
// Projection Tracing
//===----------------------------------------------------------------------===//

/// Trace the result of a projection op back to the root signal being projected
/// into. This returns the slice within the parent signal that the projection
/// targets.
SignalSlice ModuleContext::traceProjection(Value value) {
  // Check if we have already resolved this projection.
  if (auto it = projections.find(value); it != projections.end())
    return it->second;

  // Otherwise trace the projection back to the root signal.
  auto projection = traceProjectionImpl(value);
  if (projection)
    projection.signal->slices.push_back(
        ValueSlice{value, projection.offset, projection.length});
  projections.insert({value, projection});
  LLVM_DEBUG(llvm::dbgs() << "- Traced " << value << " to " << projection
                          << "\n");
  return projection;
}

/// Uncached version of `traceProjection`.
SignalSlice ModuleContext::traceProjectionImpl(Value value) {
  // Handle reprojection operations like `llhd.sig.extract` and
  // `llhd.sig.array_slice`. These don't descend into a specific subfield of the
  // input aggregate. Instead, they adjust the offset and length of the slice of
  // bits or elements targeted by the input aggregate.
  if (auto op = value.getDefiningOp<SigExtractOp>()) {
    auto slice = traceProjection(op.getInput());
    if (!slice)
      return {};
    IntegerAttr offsetAttr;
    if (!matchPattern(op.getLowBit(), m_Constant(&offsetAttr)))
      return {};
    slice.offset += offsetAttr.getValue().getZExtValue();
    slice.length = getLength(value.getType());
    return slice;
  }

  if (auto op = value.getDefiningOp<SigArraySliceOp>()) {
    auto slice = traceProjection(op.getInput());
    if (!slice)
      return {};
    IntegerAttr offsetAttr;
    if (!matchPattern(op.getLowIndex(), m_Constant(&offsetAttr)))
      return {};
    slice.offset += offsetAttr.getValue().getZExtValue();
    slice.length = getLength(value.getType());
    return slice;
  }

  // Handle proper field projections like `llhd.sig.struct_extract` and
  // `llhd.sig.array_get`. These descend into one specific subfield of the input
  // aggregate and return a new handle for that specific subsignal.
  if (auto op = value.getDefiningOp<SigArrayGetOp>()) {
    auto input = traceProjection(op.getInput());
    if (!input)
      return {};
    IntegerAttr indexAttr;
    if (!matchPattern(op.getIndex(), m_Constant(&indexAttr)))
      return {};
    unsigned offset = input.offset + indexAttr.getValue().getZExtValue();
    SignalSlice slice;
    slice.signal = internSignal(value, input.signal, offset);
    slice.length = getLength(value.getType());
    return slice;
  }

  if (auto op = value.getDefiningOp<SigStructExtractOp>()) {
    auto structType = cast<hw::StructType>(
        cast<RefType>(op.getInput().getType()).getNestedType());
    auto input = traceProjection(op.getInput());
    if (!input)
      return {};
    assert(input.offset == 0);
    assert(input.length == structType.getElements().size());
    unsigned index = *structType.getFieldIndex(op.getFieldAttr());
    SignalSlice slice;
    slice.signal = internSignal(value, input.signal, index);
    slice.length = getLength(value.getType());
    return slice;
  }

  // Otherwise create a root node for this signal.
  SignalSlice slice;
  slice.signal = internSignal(value);
  slice.length = getLength(value.getType());
  return slice;
}

/// Return the `Signal` corresponding to the given root value. Create one if it
/// does not yet exist. This ensures that aliasing projections all collapse to
/// the same underlying signals.
Signal *ModuleContext::internSignal(Value root) {
  auto &slot = internedSignals[{root, 0}];
  if (!slot) {
    slot = new (signalAlloc.Allocate()) Signal(root);
    rootSignals.push_back(slot);
  }
  return slot;
}

/// Return the `Signal` corresponding to the given parent signal and index
/// within the parent. Create one if it does not yet exist. This ensures that
/// aliasing projections all collapse to the same underlying signals.
Signal *ModuleContext::internSignal(Value value, Signal *parent,
                                    unsigned index) {
  auto &slot = internedSignals[{parent, index}];
  if (!slot) {
    slot = new (signalAlloc.Allocate()) Signal(value, parent, index);
    parent->subsignals.push_back(slot);
  }
  return slot;
}

//===----------------------------------------------------------------------===//
// Drive Aggregation
//===----------------------------------------------------------------------===//

/// Try to combine separate drives to slices or projections of a signal into one
/// drive of the entire aggregate value. This only works if the drives target
/// consecutive and non-overlapping parts of the signal. This recursively
/// aggregates drives to any subsignals first, and then tries to aggregate
/// drives for this signal.
void ModuleContext::aggregateDrives(Signal &signal) {
  // First try to aggregate drives to our subsignals. This handles signals in a
  // depth-first manner, first trying to combine drives to leaf fields to be
  // combined into a single aggregate drive before processing the parent. We
  // collect the different combinations of delay and enable operands of the
  // drives as separate vectors of drive slices.
  SmallMapVector<std::pair<Value, Value>, SmallVector<DriveSlice>, 2> drives;
  SmallPtrSet<Operation *, 8> knownDrives;
  auto addDrive = [&](DriveOp op, unsigned offset, unsigned length) {
    knownDrives.insert(op);
    drives[{op.getTime(), op.getEnable()}].push_back(
        DriveSlice{op, op.getValue(), offset, length});
  };
  for (auto *subsignal : signal.subsignals) {
    aggregateDrives(*subsignal);

    // The above call to `aggregateDrives` has populated the signal's
    // `completeDrives` with the drive ops that assign a full value to the
    // signal. Use those to seed the drive slices. Each of these drives to a
    // subsignal assign a single element of the current signal. We indicate the
    // fact that this is a single scalar element as opposed to a length-1 slice
    // of the aggregate by setting the drive slice's length field to 0.
    for (auto driveOp : subsignal->completeDrives)
      addDrive(driveOp, subsignal->indexInParent, 0);
  }

  // Gather all drives targeting this signal or slices of it directly.
  for (auto slice : signal.slices) {
    for (auto &use : slice.value.getUses()) {
      auto driveOp = dyn_cast<DriveOp>(use.getOwner());
      if (driveOp && use.getOperandNumber() == 0 &&
          driveOp->getBlock() == slice.value.getParentBlock())
        addDrive(driveOp, slice.offset, slice.length);
    }
  }

  // Check if all uses of this signal are probes or drives we are aware of. If
  // this is true we know that we can drive undriven slices of the signal with
  // its default value without breaking semantics.
  SmallSetVector<Value, 8> worklist;
  worklist.insert(signal.value);
  bool hasUnknownUses = false;
  while (!worklist.empty() && !hasUnknownUses) {
    auto value = worklist.pop_back_val();
    for (auto *user : value.getUsers()) {
      if (isa<ProbeOp>(user))
        continue;
      if (isa<DriveOp>(user) && knownDrives.contains(user))
        continue;
      if (isa<SigExtractOp, SigStructExtractOp, SigArrayGetOp, SigArraySliceOp>(
              user)) {
        worklist.insert(user->getResult(0));
        continue;
      }
      hasUnknownUses = true;
      break;
    }
  }

  // If the signal has no unknown uses and all drives have the same delay and
  // condition, we can use the signal's default value to fill in undriven
  // slices.
  if (!hasUnknownUses && drives.size() == 1) {
    auto &slices = drives.begin()->second;
    addDefaultDriveSlices(signal, slices);
  }

  // Combine driven values that uniquely cover the entire signal without gaps or
  // overlaps.
  for (auto &[key, slices] : drives) {
    llvm::sort(slices, [](auto &a, auto &b) { return a.offset < b.offset; });
    aggregateDriveSlices(signal, key.first, key.second, slices);
  }
}

/// Fill gaps in a list of drive slices with parts of the signal's default
/// value. The slices do not have to be sorted. The filler slices are appended
/// to `slices` directly.
void ModuleContext::addDefaultDriveSlices(Signal &signal,
                                          SmallVectorImpl<DriveSlice> &slices) {
  auto type = cast<RefType>(signal.value.getType()).getNestedType();

  // Sort the slices such that we can find gaps easily.
  llvm::sort(slices, [](auto &a, auto &b) { return a.offset < b.offset; });

  // A helper function to add to `gapSlices` to fill in gaps as we encounter
  // them. Structs require each field to be listed separately, since there are
  // no struct slices.
  bool anyOverlaps = false;
  bool needSeparateFields = isa<hw::StructType>(type);
  SmallVector<DriveSlice> gapSlices;
  auto fillGap = [&](unsigned from, unsigned to) {
    if (from == to)
      return;
    if (from > to) {
      anyOverlaps = true;
      return;
    }
    if (needSeparateFields) {
      for (auto idx = from; idx < to; ++idx)
        gapSlices.push_back(DriveSlice{DriveOp{}, Value{}, idx, 0});
    } else {
      gapSlices.push_back(DriveSlice{DriveOp{}, Value{}, from, to - from});
    }
  };

  // Go through the slices and keep track of the offset at which we expect the
  // slice to start. If a slice starts beyond that offset, there is a gap which
  // we can fill with a chunk of the signal's default value.
  unsigned expectedOffset = 0;
  for (auto slice : slices) {
    fillGap(expectedOffset, slice.offset);
    expectedOffset = slice.offset + std::max<unsigned>(1, slice.length);
    if (anyOverlaps)
      return;
  }
  fillGap(expectedOffset, getLength(signal.value.getType()));

  // If we have seen any overlapping slices, don't bother filling in gaps
  // because we'll later give up on combining the drives anyway.
  if (anyOverlaps || gapSlices.empty())
    return;

  // Dig up the default value for the signal.
  //
  // This is technically only valid if the signal's default value is a constant.
  // Otherwise its value may have changed between the signal's initialization
  // and now. But checking for const-ness is tricky because we might use
  // bitcasts or other aggregate creation ops to build up the constant. We
  // currently never create non-constant signal values, so this is fine for now.
  // We'll want to revisit this at a later point, though.
  //
  // This currently does not work for nested signals. To support those, we
  // potentially have to walk up our parent signals to find an actual `llhd.sig`
  // op, and then descend back down, extracting subfields.
  auto signalOp = signal.value.getDefiningOp<SignalOp>();
  if (!signalOp)
    return;
  auto defaultValue = signalOp.getInit();

  // Create drives with the default value for the gaps we've filled in.
  ImplicitLocOpBuilder builder(signal.value.getLoc(),
                               signal.value.getContext());
  builder.setInsertionPointAfterValue(signal.value);

  for (auto &slice : gapSlices) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Filling gap " << signal << "[" << slice.offset << ".."
               << (slice.offset + slice.length) << "] with initial value\n");

    // Handle integers.
    if (auto intType = dyn_cast<IntegerType>(type)) {
      assert(slice.length > 0);
      slice.value =
          comb::ExtractOp::create(builder, builder.getIntegerType(slice.length),
                                  defaultValue, slice.offset);
      continue;
    }

    // Handle structs.
    if (auto structType = dyn_cast<hw::StructType>(type)) {
      assert(slice.length == 0);
      slice.value = hw::StructExtractOp::create(
          builder, defaultValue, structType.getElements()[slice.offset]);
      continue;
    }

    // Handle arrays.
    if (auto arrayType = dyn_cast<hw::ArrayType>(type)) {
      assert(slice.length > 0);
      auto offset = hw::ConstantOp::create(
          builder,
          APInt(llvm::Log2_64_Ceil(arrayType.getNumElements()), slice.offset));
      slice.value = hw::ArraySliceOp::create(
          builder, hw::ArrayType::get(arrayType.getElementType(), slice.length),
          defaultValue, offset);
      continue;
    }
  }

  // Add the gap fillers to the list of slices. These will be resorted later and
  // will then form a consecutive non-overlapping assignment to the entire
  // signal.
  slices.append(gapSlices.begin(), gapSlices.end());
}

/// Combine multiple drive slices into a single drive of the aggregate value.
/// The slices must be sorted by offset with the lowest offset first.
void ModuleContext::aggregateDriveSlices(Signal &signal, Value driveDelay,
                                         Value driveEnable,
                                         ArrayRef<DriveSlice> slices) {
  // Check whether the slices are consecutive and non-overlapping.
  unsigned expectedOffset = 0;
  for (auto slice : slices) {
    assert(slice.value && "all slices must have an assigned value");
    if (slice.offset != expectedOffset) {
      expectedOffset = -1;
      break;
    }
    // Individual subsignals are represented with length 0, since these describe
    // an individual field and not a slice of the aggregate (`array<1xi42>` vs.
    // `i42`). Therefore we have to count length 0 fields as single elements.
    expectedOffset += std::max<unsigned>(1, slice.length);
  }
  if (expectedOffset != getLength(signal.value.getType())) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Signal " << signal << " not completely driven\n");
    return;
  }

  // If we get here we cover the entire signal. If we already have a single
  // drive, simply mark that as this signal's single drive. Otherwise we have to
  // do some actual work.
  if (slices.size() == 1 && slices[0].length != 0 && slices[0].op) {
    signal.completeDrives.push_back(slices[0].op);
    return;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "- Aggregating " << signal << " drives (delay ";
    driveDelay.printAsOperand(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    if (driveEnable) {
      llvm::dbgs() << " if ";
      driveEnable.printAsOperand(llvm::dbgs(),
                                 OpPrintingFlags().useLocalScope());
    }
    llvm::dbgs() << ")\n";
  });

  Value result;
  auto type = cast<RefType>(signal.value.getType()).getNestedType();
  ImplicitLocOpBuilder builder(signal.value.getLoc(),
                               signal.value.getContext());
  builder.setInsertionPointAfterValue(signal.value);

  // Handle integers.
  if (auto intType = dyn_cast<IntegerType>(type)) {
    // If there are more than one slices, concatenate them. Integers are pretty
    // straightforward since there is no dedicated single-bit type. So
    // everything is just a concatenation.
    SmallVector<Value> operands;
    for (auto slice : slices)
      operands.push_back(slice.value);
    std::reverse(operands.begin(), operands.end()); // why, just why
    result = comb::ConcatOp::create(builder, operands);
    LLVM_DEBUG(llvm::dbgs() << "  - Created " << result << "\n");
  }

  // Handle structs.
  if (auto structType = dyn_cast<hw::StructType>(type)) {
    // Structs are trivial, since there are no struct slices. Everything is an
    // individual field that we can use directly to create the struct.
    SmallVector<Value> operands;
    for (auto slice : slices)
      operands.push_back(slice.value);
    result = hw::StructCreateOp::create(builder, structType, operands);
    LLVM_DEBUG(llvm::dbgs() << "  - Created " << result << "\n");
  }

  // Handle arrays.
  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) {
    // Our slices vector may consist of individual, scalar array elements or
    // entire slices of the array. In a first step, convert all scalar elements
    // into array slices.
    SmallVector<Value> scalars;
    SmallVector<Value> aggregates;
    auto flushScalars = [&] {
      if (scalars.empty())
        return;
      std::reverse(scalars.begin(), scalars.end()); // why, just why
      auto aggregate = hw::ArrayCreateOp::create(builder, scalars);
      aggregates.push_back(aggregate);
      scalars.clear();
      LLVM_DEBUG(llvm::dbgs() << "  - Created " << aggregate << "\n");
    };
    for (auto slice : slices) {
      if (slice.length == 0) {
        scalars.push_back(slice.value);
      } else {
        flushScalars();
        aggregates.push_back(slice.value);
      }
    }
    flushScalars();

    // If there are more than one aggregate slice of the array, concatenate
    // them into one single aggregate value.
    result = aggregates.back();
    if (aggregates.size() != 1) {
      std::reverse(aggregates.begin(), aggregates.end()); // why, just why
      result = hw::ArrayConcatOp::create(builder, aggregates);
      LLVM_DEBUG(llvm::dbgs() << "  - Created " << result << "\n");
    }
  }

  // Create the single drive with the aggregate result.
  assert(result);
  auto driveOp =
      DriveOp::create(builder, signal.value, result, driveDelay, driveEnable);
  signal.completeDrives.push_back(driveOp);
  LLVM_DEBUG(llvm::dbgs() << "  - Created " << driveOp << "\n");

  // Mark the old drives as to be deleted.
  for (auto slice : slices) {
    if (!slice.op)
      continue;
    LLVM_DEBUG(llvm::dbgs() << "  - Removed " << slice.op << "\n");
    pruner.eraseNow(slice.op);
  }
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct CombineDrivesPass
    : public llhd::impl::CombineDrivesPassBase<CombineDrivesPass> {
  void runOnOperation() override;
};
} // namespace

void CombineDrivesPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "Combining drives in "
                          << getOperation().getModuleNameAttr() << "\n");
  ModuleContext context(getOperation());

  // Take note of all projection operations.
  for (auto &op : *context.moduleOp.getBodyBlock())
    if (isa<SigExtractOp, SigArraySliceOp, SigArrayGetOp, SigStructExtractOp>(
            &op))
      context.traceProjection(op.getResult(0));

  // Aggregate drives to these projections.
  for (auto *signal : context.rootSignals)
    context.aggregateDrives(*signal);

  // Clean up any ops that have become obsolete.
  context.pruner.eraseNow();
}
