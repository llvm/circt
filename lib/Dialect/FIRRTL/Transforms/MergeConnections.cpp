//===- MergeConnections.cpp - Merge expanded connections --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass merges expanded connections into one connection.
// LowerTypes fully expands aggregate connections even when semantically
// not necessary to expand because it is required for ExpandWhen.
//
// More specifically this pass folds the following patterns:
//   %dest(0) <= v0
//   %dest(1) <= v1
//   ...
//   %dest(n) <= vn
// into
//   %dest <= {vn, .., v1, v0}
// Also if v0, v1, .., vn are subfield op like %a(0), %a(1), ..., a(n), then we
// merge entire connections into %dest <= %a.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-merge-connections"

using namespace circt;
using namespace firrtl;

// Return true if value is essentially constant.
static bool isConstantLike(Value value) {
  auto op = value.getDefiningOp();
  if (op && isa<ConstantOp, InvalidValueOp>(op))
    return true;
  if (auto bitcast = value.getDefiningOp<BitCastOp>())
    return isConstant(bitcast.input());

  // TODO: Add unrealized_conversion, asUInt, asSInt
  return false;
}

namespace {
//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

// A helper struct to merge connections.
struct MergeConnection {
  MergeConnection(FModuleOp moduleOp, bool enableAggressiveMerging)
      : moduleOp(moduleOp), enableAggressiveMerging(enableAggressiveMerging) {}

  // Return true if something is changed.
  bool run();
  bool changed = false;

  // Return true if the given connect op is merged.
  bool peelConnect(ConnectOp connect);

  // A map from a destination FieldRef to a pair of (i) the number of
  // connections seen so far and (ii) the vector to store subconnections.
  DenseMap<FieldRef, std::pair<unsigned, SmallVector<ConnectOp>>> connections;

  FModuleOp moduleOp;
  ImplicitLocOpBuilder *builder = nullptr;

  // If true, we merge connections even when source values will not be
  // simplified.
  bool enableAggressiveMerging = false;
};

bool MergeConnection::peelConnect(ConnectOp connect) {
  // Ignore connections between different types because it will produce a
  // partial connect. Also ignore non-passive connections or non-integer
  // connections.
  LLVM_DEBUG(llvm::dbgs() << "Visiting " << connect << "\n");
  auto destTy = connect.dest().getType().cast<FIRRTLType>();
  auto srcTy = connect.src().getType().cast<FIRRTLType>();
  if (destTy != srcTy || !destTy.isPassive() ||
      !firrtl::getBitWidth(destTy).hasValue())
    return false;

  auto destFieldRef = getFieldRefFromValue(connect.dest());
  auto destRoot = destFieldRef.getValue();

  // If dest is derived from mem op or has a ground type, we cannot merge them.
  // If the connect's destination is a root value, we cannot merge.
  if (destRoot.getDefiningOp<MemOp>() || destRoot == connect.dest())
    return false;

  Value parent;
  unsigned index;
  if (auto subfield = dyn_cast<SubfieldOp>(connect.dest().getDefiningOp()))
    parent = subfield.input(), index = subfield.fieldIndex();
  else if (auto subindex = dyn_cast<SubindexOp>(connect.dest().getDefiningOp()))
    parent = subindex.input(), index = subindex.index();
  else
    llvm_unreachable("unexpected destination");

  auto &countAndSubConnections = connections[getFieldRefFromValue(parent)];
  auto &count = countAndSubConnections.first;
  auto &subConnections = countAndSubConnections.second;

  // If it is the first time to visit the parent op, then allocate the vector
  // for subconnections.
  if (count == 0) {
    if (auto bundle = parent.getType().dyn_cast<BundleType>())
      subConnections.resize(bundle.getNumElements());
    if (auto vector = parent.getType().dyn_cast<FVectorType>())
      subConnections.resize(vector.getNumElements());
  }
  ++count;
  subConnections[index] = connect;

  // If we haven't visited all subconnections, stop at this point.
  if (count != subConnections.size())
    return false;

  changed = true;

  auto parentType = parent.getType();

  auto getMergedValue = [&](auto aggregateType) {
    SmallVector<Value> operands;

    // This flag tracks whether we can use the parent of source values as the
    // merged value.
    bool canUseSourceParent = true;
    bool areOperandsAllConstants = true;

    // The value which might be used as a merged value.
    Value sourceParent;

    auto checkSourceParent = [&](auto subelement, unsigned destIndex,
                                 unsigned sourceIndex) {
      // In the first iteration, register a parent value.
      if (destIndex == 0) {
        if (subelement.input().getType() == parentType)
          sourceParent = subelement.input();
        else {
          // If types are not same, it is not possible to use it.
          canUseSourceParent = false;
        }
      }

      // Check that input is the same as `sourceAggregate` and indexes match.
      canUseSourceParent &=
          subelement.input() == sourceParent && destIndex == sourceIndex;
    };

    for (auto idx : llvm::seq(0u, (unsigned)aggregateType.getNumElements())) {
      auto src = subConnections[idx].src();
      assert(src && "all subconnections are guranteed to exist");
      operands.push_back(src);

      areOperandsAllConstants &= isConstantLike(src);

      // From here, check whether the value is derived from the same aggregate
      // value.

      // If canUseSourceParent is already false, abort.
      if (!canUseSourceParent)
        continue;

      // If the value is an argument, it is not derived from an aggregate value.
      if (!src.getDefiningOp()) {
        canUseSourceParent = false;
        continue;
      }

      TypeSwitch<Operation *>(src.getDefiningOp())
          .template Case<SubfieldOp>([&](SubfieldOp subfield) {
            checkSourceParent(subfield, idx, subfield.fieldIndex());
          })
          .template Case<SubindexOp>([&](SubindexOp subindex) {
            checkSourceParent(subindex, idx, subindex.index());
          })
          .Default([&](auto) { canUseSourceParent = false; });
    }

    // If it is fine to use `sourceParent` as a merged value, we just
    // return it.
    if (canUseSourceParent) {
      LLVM_DEBUG(llvm::dbgs() << "Success to merge " << destFieldRef.getValue()
                              << " ,fieldID= " << destFieldRef.getFieldID()
                              << " to " << sourceParent << "\n";);
      // Erase connections except for subConnections[index] since it must be
      // erased at the top-level loop.
      for (auto idx : llvm::seq(0u, static_cast<unsigned>(operands.size())))
        if (idx != index)
          subConnections[idx].erase();
      return sourceParent;
    }

    // If operands are not all constants, we don't merge connections unless
    // "aggressive-merging" option is enabled.
    if (!enableAggressiveMerging && !areOperandsAllConstants)
      return Value();

    // Otherwise, we concat all values and cast them into the aggregate type.
    Value accumulate;
    for (auto e : llvm::enumerate(operands)) {
      // Erase connections except for subConnections[index] since it must be
      // erased at the top-level loop.
      if (e.index() != index)
        subConnections[e.index()].erase();
      auto value = e.value();
      auto bitwidth =
          firrtl::getBitWidth(value.getType().template cast<FIRRTLType>());
      assert(bitwidth &&
             "it should be checked at the beginning of `peelConnect`");
      value = builder->createOrFold<BitCastOp>(
          value.getLoc(), UIntType::get(value.getContext(), *bitwidth), value);

      if (parentType.isa<FVectorType>())
        accumulate = (accumulate ? builder->createOrFold<CatPrimOp>(
                                       accumulate.getLoc(), value, accumulate)
                                 : value);
      else {
        // Bundle subfields are filled from MSB to LSB.
        accumulate = (accumulate ? builder->createOrFold<CatPrimOp>(
                                       accumulate.getLoc(), accumulate, value)
                                 : value);
      }
    }
    return builder->createOrFold<BitCastOp>(accumulate.getLoc(), parentType,
                                            accumulate);
  };

  Value merged;
  if (auto bundle = parentType.dyn_cast_or_null<BundleType>())
    merged = getMergedValue(bundle);
  if (auto vector = parentType.dyn_cast_or_null<FVectorType>())
    merged = getMergedValue(vector);
  if (!merged)
    return false;

  builder->create<ConnectOp>(connect.getLoc(), parent, merged);
  return true;
}

bool MergeConnection::run() {
  ImplicitLocOpBuilder theBuilder(moduleOp.getLoc(), moduleOp.getContext());
  builder = &theBuilder;
  auto *body = moduleOp.getBody();
  // Merge connections by forward iterations.
  for (auto it = body->begin(), e = body->end(); it != e;) {
    auto connectOp = dyn_cast<ConnectOp>(*it);
    if (!connectOp) {
      it++;
      continue;
    }
    builder->setInsertionPointAfter(connectOp);
    builder->setLoc(connectOp.getLoc());
    bool removeOp = peelConnect(connectOp);
    ++it;
    if (removeOp)
      connectOp.erase();
  }

  // Clean up dead operations introduced by this pass.
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*body)))
    if (isa<SubfieldOp, SubindexOp, InvalidValueOp, ConstantOp, BitCastOp,
            CatPrimOp>(op))
      if (op.use_empty()) {
        changed = true;
        op.erase();
      }

  return changed;
}

struct MergeConnectionsPass
    : public MergeConnectionsBase<MergeConnectionsPass> {
  MergeConnectionsPass(bool enableAggressiveMergingFlag) {
    enableAggressiveMerging = enableAggressiveMergingFlag;
  }
  void runOnOperation() override;
};

} // namespace

void MergeConnectionsPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===----- Running MergeConnections "
                             "--------------------------------------===\n"
                          << "Module: '" << getOperation().getName() << "'\n";);

  MergeConnection mergeConnection(getOperation(), enableAggressiveMerging);
  bool changed = mergeConnection.run();

  if (!changed)
    return markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createMergeConnectionsPass(bool enableAggressiveMerging) {
  return std::make_unique<MergeConnectionsPass>(enableAggressiveMerging);
}
