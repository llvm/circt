//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-combine-drives"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_COMBINEDRIVESPASS
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;
using hw::HWModuleOp;
using llvm::SpecificBumpPtrAllocator;

static unsigned getLength(Type type) {
  return TypeSwitch<Type, unsigned>(cast<hw::InOutType>(type).getElementType())
      .Case<IntegerType>([](auto type) { return type.getWidth(); })
      .Case<hw::ArrayType>([](auto type) { return type.getNumElements(); })
      .Case<hw::StructType>([](auto type) { return type.getElements().size(); })
      .Default([](auto) { return 0; });
}

namespace {
struct Signal {
  /// The SSA value representing the root signal. Mutually exclusive with
  /// `parent`.
  Value root;
  /// The parent aggregate signal that contains this signal. Mutually exclusive
  /// with `root`.
  Signal *parent = nullptr;
  /// Index of the field within the parent.
  unsigned index = 0;

  explicit Signal(Value root) : root(root) {}
  Signal(Signal *parent, unsigned index) : parent(parent), index(index) {}
};

struct SignalSlice {
  Signal *signal = nullptr;
  unsigned offset = 0;
  unsigned length = 0;
  /// Projections which pick elements based on a dynamic SSA value instead of a
  /// constant value will set `poison` to true to indicate the slice of the
  /// signal cannot be combined into a larger drive. (The dynamic value implies
  /// that individual elements of an aggregate are conditionally driven, which
  /// we do not handle in this pass.)
  bool poison = false;

  explicit operator bool() const { return signal != nullptr; }
};
} // namespace

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const Signal &signal) {
  if (signal.root) {
    signal.root.printAsOperand(os, OpPrintingFlags());
    return os;
  }
  return os << *signal.parent << "[" << signal.index << "]";
}

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os, SignalSlice slice) {
  return os << *slice.signal << "[" << slice.offset << ".."
            << (slice.offset + slice.length) << "]";
}

namespace {
struct ModuleContext {
  ModuleContext(HWModuleOp moduleOp) : moduleOp(moduleOp) {}

  // Utilities to trace the result of a projection op back to the root signal
  // being projected into.
  SignalSlice traceProjection(Value value);
  SignalSlice traceProjectionImpl(Value value);
  Signal *internSignal(Value root);
  Signal *internSignal(Signal *parent, unsigned index);

  /// The module within which we are combining drives.
  HWModuleOp moduleOp;
  /// The signal slice targeted by each projection op in the module.
  DenseMap<Value, SignalSlice> projections;

private:
  using SignalKey = std::pair<PointerUnion<Value, Signal *>, unsigned>;
  SpecificBumpPtrAllocator<Signal> signalAlloc;
  DenseMap<SignalKey, Signal *> internedSignals;
};
} // namespace

/// Trace the result of a projection op back to the root signal being projected
/// into. This returns the slice within the parent signal that the projection
/// targets.
SignalSlice ModuleContext::traceProjection(Value value) {
  // Check if we have already resolved this projection.
  if (auto projection = projections.lookup(value))
    return projection;

  // Otherwise trace the projection back to the root signal.
  auto projection = traceProjectionImpl(value);
  projections.insert({value, projection});
  LLVM_DEBUG(llvm::dbgs() << "- Traced " << value << " to " << projection
                          << "\n");
  return projection;
}

/// Uncached version of `traceProjection`.
SignalSlice ModuleContext::traceProjectionImpl(Value value) {
  if (auto op = value.getDefiningOp<SigExtractOp>()) {
    auto slice = traceProjection(op.getInput());
    IntegerAttr offsetAttr;
    if (!matchPattern(op.getLowBit(), m_Constant(&offsetAttr)))
      // TODO: What should we do if the index is not constant? Poison?
      assert(false && "this needs to do something");
    slice.offset += offsetAttr.getValue().getZExtValue();
    slice.length = getLength(value.getType());
    return slice;
  }

  if (auto op = value.getDefiningOp<SigArrayGetOp>()) {
    auto input = traceProjection(op.getInput());
    IntegerAttr indexAttr;
    if (!matchPattern(op.getIndex(), m_Constant(&indexAttr)))
      // TODO: What should we do if the index is not constant? Poison?
      assert(false && "this needs to do something");
    SignalSlice slice;
    slice.signal = internSignal(
        input.signal, input.offset + indexAttr.getValue().getZExtValue());
    slice.length = getLength(value.getType());
    return slice;
  }

  if (auto op = value.getDefiningOp<SigArraySliceOp>()) {
    auto slice = traceProjection(op.getInput());
    IntegerAttr offsetAttr;
    if (!matchPattern(op.getLowIndex(), m_Constant(&offsetAttr)))
      // TODO: What should we do if the index is not constant? Poison?
      assert(false && "this needs to do something");
    slice.offset += offsetAttr.getValue().getZExtValue();
    slice.length = getLength(value.getType());
    return slice;
  }

  if (auto op = value.getDefiningOp<SigStructExtractOp>()) {
    auto structType = cast<hw::StructType>(
        cast<hw::InOutType>(op.getInput().getType()).getElementType());
    auto input = traceProjection(op.getInput());
    assert(input.offset == 0);
    assert(input.length == structType.getElements().size());
    unsigned index = *structType.getFieldIndex(op.getFieldAttr());
    SignalSlice slice;
    slice.signal = internSignal(input.signal, index);
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
  if (!slot)
    slot = new (signalAlloc.Allocate()) Signal(root);
  return slot;
}

/// Return the `Signal` corresponding to the given parent signal and index
/// within the parent. Create one if it does not yet exist. This ensures that
/// aliasing projections all collapse to the same underlying signals.
Signal *ModuleContext::internSignal(Signal *parent, unsigned index) {
  auto &slot = internedSignals[{parent, index}];
  if (!slot)
    slot = new (signalAlloc.Allocate()) Signal(parent, index);
  return slot;
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
    if (isa<SigExtractOp, SigArrayGetOp, SigStructExtractOp>(&op))
      context.traceProjection(op.getResult(0));
}
