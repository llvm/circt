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
//   %dest(0) <= subfield %a(0)
//   %dest(1) <= subfield %a(1)
//   ...
//   %dest(n) <= subfield %a(n)
// into
//   %dest <= %a
// This pass also merge connections if all of their source values are known to
// be constant.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <queue>

#define DEBUG_TYPE "firrtl-merge-connections"

using namespace circt;
using namespace firrtl;

namespace {
//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

// A helper struct to merge connections.
struct MergeConnection {
  MergeConnection(FModuleOp moduleOp) : moduleOp(moduleOp) {}

  // Return true if something is changed.
  bool run();
  bool changed = false;

  void peelConnect(ConnectOp connect);

  // A map from a destination FieldRef to connect op.
  DenseMap<FieldRef, ConnectOp> fieldRefToConnect;

  // Return a merged value which is connecting to `dest` if exists.
  Value getMergedConnectedValue(FieldRef dest);

  // Set of root values which can be merged.
  llvm::SetVector<Value> peeledRoot;

  // Vector which tracks dead connections.
  SmallVector<ConnectOp> shouldBeRemoved;

  FModuleOp moduleOp;
};

Value MergeConnection::getMergedConnectedValue(FieldRef fieldRef) {
  LLVM_DEBUG(llvm::dbgs() << "Merging ... " << fieldRef.getValue()
                          << ", fieldID=" << fieldRef.getFieldID() << "\n");

  // If there is a connection on this level, just return connected value and
  // erase the connection.
  auto it = fieldRefToConnect.find(fieldRef);
  if (it != fieldRefToConnect.end()) {
    auto src = it->second.src();
    shouldBeRemoved.push_back(it->second);
    return src;
  }

  auto type = getTypeFromFieldRef(fieldRef);
  auto builder = OpBuilder::atBlockEnd(moduleOp.getBody());

  auto getMergedValue = [&](auto aggregateType) {
    SmallVector<Value> operands;

    // This flag tracks whether we can use aggregate value as merged value.
    bool canUseSourceAggregate = true;
    Value sourceAggregate;

    auto checkSourceAggregate = [&](auto subelement, unsigned destIndex,
                                    unsigned sourceIndex) {
      // In the first iteration, register the source aggregate value.
      if (destIndex == 0) {
        if (subelement.input().getType() == type)
          sourceAggregate = subelement.input();
        else {
          // If the types are not same, it is not possible to use the source
          // aggregate.
          canUseSourceAggregate = false;
        }
      }

      // Check that input is the same as `sourceAggregate` and indexes match.
      canUseSourceAggregate &=
          subelement.input() == sourceAggregate && destIndex == sourceIndex;
    };

    for (auto idx : llvm::seq(0u, (unsigned)aggregateType.getNumElements())) {
      auto offset = aggregateType.getFieldID(idx);
      auto value = getMergedConnectedValue(fieldRef.getSubField(offset));
      if (!value)
        return Value();

      operands.push_back(value);

      if (!canUseSourceAggregate)
        continue;

      if (!value.getDefiningOp()) {
        canUseSourceAggregate = false;
        continue;
      }

      TypeSwitch<Operation *>(value.getDefiningOp())
          .template Case<SubfieldOp>([&](SubfieldOp subfield) {
            checkSourceAggregate(subfield, idx, subfield.fieldIndex());
          })
          .template Case<SubindexOp>([&](SubindexOp subindex) {
            checkSourceAggregate(subindex, idx, subindex.index());
          })
          .Default([&](auto) { canUseSourceAggregate = false; });
    }

    if (canUseSourceAggregate) {
      LLVM_DEBUG(llvm::dbgs() << "Success to merge " << fieldRef.getValue()
                              << " ,fieldID= " << fieldRef.getFieldID() << "to"
                              << sourceAggregate << "\n";);
      return sourceAggregate;
    }

    // Here, we concat all inputs and cast them into aggregate type.
    Value accumulate;
    for (auto value : operands) {
      value = builder.createOrFold<BitCastOp>(
          value.getLoc(),
          UIntType::get(value.getContext(),
                        *firrtl::getBitWidth(
                            value.getType().template cast<FIRRTLType>())),
          value);
      accumulate = (accumulate ? builder.createOrFold<CatPrimOp>(
                                     accumulate.getLoc(), accumulate, value)
                               : value);
    }
    return builder.createOrFold<BitCastOp>(accumulate.getLoc(), type,
                                           accumulate);
  };

  if (auto bundle = type.dyn_cast_or_null<BundleType>())
    return getMergedValue(bundle);
  if (auto vector = type.dyn_cast_or_null<FVectorType>())
    return getMergedValue(vector);

  return Value();
}

void MergeConnection::peelConnect(ConnectOp connect) {
  // Ignore connections between different types because it will produce a
  // partial connect.
  if (connect.src().getType() != connect.dest().getType())
    return;

  auto destFieldRef = getFieldRefFromValue(connect.dest());
  auto destRoot = destFieldRef.getValue();

  // If dest is derived from mem op or has a ground type, we cannot merge them.
  if (destRoot.getDefiningOp<MemOp>() ||
      destRoot.getType().cast<FIRRTLType>().isGround())
    return;

  fieldRefToConnect[destFieldRef] = connect;
  peeledRoot.insert(destRoot);
  return;
}

bool MergeConnection::run() {
  // Peel all connections.
  moduleOp.walk([&](ConnectOp connect) { peelConnect(connect); });

  // Iterate over roots.
  for (auto root : peeledRoot) {
    // If the root has a ground type, there is no connection to merge.
    if (root.getType().cast<FIRRTLType>().isGround())
      continue;

    unsigned size = shouldBeRemoved.size();
    auto value = getMergedConnectedValue({root, 0});
    if (!value) {
      // If we failed to get merged value, revert dead connections.
      shouldBeRemoved.resize(size);
      continue;
    }
    auto builder = OpBuilder::atBlockEnd(moduleOp.getBody());
    builder.create<ConnectOp>(root.getLoc(), root, value);
  }

  if (shouldBeRemoved.empty())
    return false;

  // Remove dead connections.
  for (auto connect : shouldBeRemoved)
    connect.erase();

  // Clean up.
  auto *body = moduleOp.getBody();
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*body)))
    if (isa<SubfieldOp, SubindexOp, InvalidValueOp, ConstantOp, BitCastOp>(op))
      if (op.use_empty()) {
        changed = true;
        op.erase();
      }

  return true;
}

struct MergeConnectionsPass
    : public MergeConnectionsBase<MergeConnectionsPass> {
  void runOnOperation() override;
};

} // namespace

void MergeConnectionsPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===----- Running MergeConnections "
                             "--------------------------------------===\n"
                          << "Module: '" << getOperation().getName() << "'\n";);

  MergeConnection mergeConnection(getOperation());
  bool changed = mergeConnection.run();

  if (!changed)
    return markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createMergeConnectionsPass() {
  return std::make_unique<MergeConnectionsPass>();
}
