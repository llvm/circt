//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InferResets pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/Debug.h"
#include "circt/Support/FieldRef.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "infer-resets"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_INFERRESETS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using llvm::SmallDenseSet;
using llvm::SmallSetVector;
using mlir::FailureOr;
using mlir::InferTypeOpInterface;

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Reset Network
//===----------------------------------------------------------------------===//

namespace {

/// A reset signal.
///
/// This essentially combines the exact `FieldRef` of the signal in question
/// with a type to be used for error reporting and inferring the reset kind.
struct ResetSignal {
  ResetSignal(FieldRef field, FIRRTLBaseType type) : field(field), type(type) {}
  bool operator<(const ResetSignal &other) const { return field < other.field; }
  bool operator==(const ResetSignal &other) const {
    return field == other.field;
  }
  bool operator!=(const ResetSignal &other) const { return !(*this == other); }

  FieldRef field;
  FIRRTLBaseType type;
};

/// A connection made to or from a reset network.
///
/// These drives are tracked for each reset network, and are used for error
/// reporting to the user.
struct ResetDrive {
  /// What's being driven.
  ResetSignal dst;
  /// What's driving.
  ResetSignal src;
  /// The location to use for diagnostics.
  Location loc;
};

/// A list of connections to a reset network.
using ResetDrives = SmallVector<ResetDrive, 1>;

/// All signals connected together into a reset network.
using ResetNetwork = llvm::iterator_range<
    llvm::EquivalenceClasses<ResetSignal>::member_iterator>;

/// Whether a reset is sync or async.
enum class ResetKind { Async, Sync };
} // namespace

namespace llvm {
template <>
struct DenseMapInfo<ResetSignal> {
  static unsigned getHashValue(const ResetSignal &x) {
    return circt::hash_value(x.field);
  }
  static bool isEqual(const ResetSignal &lhs, const ResetSignal &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

template <typename T>
static T &operator<<(T &os, const ResetKind &kind) {
  switch (kind) {
  case ResetKind::Async:
    return os << "async";
  case ResetKind::Sync:
    return os << "sync";
  }
  return os;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
/// Infer concrete reset types for abstract resets.
///
/// This pass replaces `reset` types in the IR with a concrete `asyncreset` or
/// `uint<1>` depending on how the reset is used.
///
/// 1. Build a global graph of the resets in the design by tracing reset signals
///    through instances.
/// 2. Infer the type of each reset network found in step 1.
/// 3. Walk the IR and update the type of wires and ports with the reset types
///    found in step 2.
/// 4. Require that no abstract resets remain on ports.
struct InferResetsPass
    : public circt::firrtl::impl::InferResetsBase<InferResetsPass> {
  void runOnOperation() override;
  void runOnOperationInner();

  using InferResetsBase::InferResetsBase;
  InferResetsPass(const InferResetsPass &other) : InferResetsBase(other) {}

  //===--------------------------------------------------------------------===//
  // Reset type inference

  void traceResets(CircuitOp circuit);
  void traceResets(FInstanceLike inst);
  void traceResets(Value dst, Value src, Location loc);
  void traceResets(Value value);
  void traceResets(Type dstType, Value dst, unsigned dstID, Type srcType,
                   Value src, unsigned srcID, Location loc);

  LogicalResult inferAndUpdateResets();
  FailureOr<ResetKind> inferReset(ResetNetwork net);
  LogicalResult updateReset(ResetNetwork net, ResetKind kind);
  bool updateReset(FieldRef field, FIRRTLBaseType resetType);

  LogicalResult verifyNoAbstractReset();

  //===--------------------------------------------------------------------===//
  // Utilities

  /// Get the reset network a signal belongs to.
  ResetNetwork getResetNetwork(ResetSignal signal) {
    return llvm::make_range(resetClasses.findLeader(signal),
                            resetClasses.member_end());
  }

  /// Get the drives of a reset network.
  ResetDrives &getResetDrives(ResetNetwork net) {
    return resetDrives[*net.begin()];
  }

  /// Guess the root node of a reset network, such that we have something for
  /// the user to make sense of.
  ResetSignal guessRoot(ResetNetwork net);
  ResetSignal guessRoot(ResetSignal signal) {
    return guessRoot(getResetNetwork(signal));
  }

  //===--------------------------------------------------------------------===//
  // Analysis data

  /// A map of all traced reset networks in the circuit.
  llvm::EquivalenceClasses<ResetSignal> resetClasses;

  /// A map of all connects to and from a reset.
  DenseMap<ResetSignal, ResetDrives> resetDrives;

  /// Cache of modules symbols
  InstanceGraph *instanceGraph = nullptr;
};
} // namespace

void InferResetsPass::runOnOperation() {
  runOnOperationInner();
  resetClasses = llvm::EquivalenceClasses<ResetSignal>();
  resetDrives.clear();
  markAnalysesPreserved<InstanceGraph>();
}

void InferResetsPass::runOnOperationInner() {
  instanceGraph = &getAnalysis<InstanceGraph>();

  // Trace the uninferred reset networks throughout the design.
  traceResets(getOperation());

  // Infer the type of the traced resets and update the IR.
  if (failed(inferAndUpdateResets()))
    return signalPassFailure();

  // Require that no Abstract Resets exist on ports in the design.
  if (failed(verifyNoAbstractReset()))
    return signalPassFailure();
}

ResetSignal InferResetsPass::guessRoot(ResetNetwork net) {
  ResetDrives &drives = getResetDrives(net);
  ResetSignal bestSignal = *net.begin();
  unsigned bestNumDrives = -1;

  for (auto signal : net) {
    // Don't consider `invalidvalue` for reporting as a root.
    if (isa_and_nonnull<InvalidValueOp>(
            signal.field.getValue().getDefiningOp()))
      continue;

    // Count the number of times this particular signal in the reset network is
    // assigned to.
    unsigned numDrives = 0;
    for (auto &drive : drives)
      if (drive.dst == signal)
        ++numDrives;

    // Keep track of the signal with the lowest number of assigns. These tend to
    // be the signals further up the reset tree. This will usually resolve to
    // the root of the reset tree far up in the design hierarchy.
    if (numDrives < bestNumDrives) {
      bestNumDrives = numDrives;
      bestSignal = signal;
    }
  }
  return bestSignal;
}

//===----------------------------------------------------------------------===//
// Custom Field IDs
//===----------------------------------------------------------------------===//

// The following functions implement custom field IDs specifically for the use
// in reset inference. They look much more like tracking fields on types than
// individual values. For example, vectors don't carry separate IDs for each of
// their elements. Instead they have one set of IDs for the entire vector, since
// the element type is uniform across all elements.

static unsigned getMaxFieldID(FIRRTLBaseType type) {
  return FIRRTLTypeSwitch<FIRRTLBaseType, unsigned>(type)
      .Case<BundleType>([](auto type) {
        unsigned id = 0;
        for (auto e : type.getElements())
          id += getMaxFieldID(e.type) + 1;
        return id;
      })
      .Case<FVectorType>(
          [](auto type) { return getMaxFieldID(type.getElementType()) + 1; })
      .Default([](auto) { return 0; });
}

static unsigned getFieldID(BundleType type, unsigned index) {
  assert(index < type.getNumElements());
  unsigned id = 1;
  for (unsigned i = 0; i < index; ++i)
    id += getMaxFieldID(type.getElementType(i)) + 1;
  return id;
}

static unsigned getFieldID(FVectorType type) { return 1; }

static unsigned getIndexForFieldID(BundleType type, unsigned fieldID) {
  assert(type.getNumElements() && "Bundle must have >0 fields");
  --fieldID;
  for (const auto &e : llvm::enumerate(type.getElements())) {
    auto numSubfields = getMaxFieldID(e.value().type) + 1;
    if (fieldID < numSubfields)
      return e.index();
    fieldID -= numSubfields;
  }
  assert(false && "field id outside bundle");
  return 0;
}

// If a field is pointing to a child of a zero-length vector, it is useless.
static bool isUselessVec(FIRRTLBaseType oldType, unsigned fieldID) {
  if (oldType.isGround()) {
    assert(fieldID == 0);
    return false;
  }

  // If this is a bundle type, recurse.
  if (auto bundleType = type_dyn_cast<BundleType>(oldType)) {
    unsigned index = getIndexForFieldID(bundleType, fieldID);
    return isUselessVec(bundleType.getElementType(index),
                        fieldID - getFieldID(bundleType, index));
  }

  // If this is a vector type, check if it is zero length.  Anything in a
  // zero-length vector is useless.
  if (auto vectorType = type_dyn_cast<FVectorType>(oldType)) {
    if (vectorType.getNumElements() == 0)
      return true;
    return isUselessVec(vectorType.getElementType(),
                        fieldID - getFieldID(vectorType));
  }

  return false;
}

// If a field is pointing to a child of a zero-length vector, it is useless.
static bool isUselessVec(FieldRef field) {
  return isUselessVec(
      getBaseType(type_cast<FIRRTLType>(field.getValue().getType())),
      field.getFieldID());
}

static bool getDeclName(Value value, SmallString<32> &string) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto module = cast<FModuleOp>(arg.getOwner()->getParentOp());
    string += module.getPortName(arg.getArgNumber());
    return true;
  }

  auto *op = value.getDefiningOp();
  return TypeSwitch<Operation *, bool>(op)
      .Case<InstanceOp, InstanceChoiceOp, MemOp>([&](auto op) {
        string += op.getName();
        string += ".";
        string += op.getPortName(cast<OpResult>(value).getResultNumber());
        return true;
      })
      .Case<WireOp, NodeOp, RegOp, RegResetOp>([&](auto op) {
        string += op.getName();
        return true;
      })
      .Default([](auto) { return false; });
}

static bool getFieldName(const FieldRef &fieldRef, SmallString<32> &string) {
  SmallString<64> name;
  auto value = fieldRef.getValue();
  if (!getDeclName(value, string))
    return false;

  auto type = value.getType();
  auto localID = fieldRef.getFieldID();
  while (localID) {
    if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      auto index = getIndexForFieldID(bundleType, localID);
      // Add the current field string, and recurse into a subfield.
      auto &element = bundleType.getElements()[index];
      if (!string.empty())
        string += ".";
      string += element.name.getValue();
      // Recurse in to the element type.
      type = element.type;
      localID = localID - getFieldID(bundleType, index);
    } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
      string += "[]";
      // Recurse in to the element type.
      type = vecType.getElementType();
      localID = localID - getFieldID(vecType);
    } else {
      // If we reach here, the field ref is pointing inside some aggregate type
      // that isn't a bundle or a vector. If the type is a ground type, then the
      // localID should be 0 at this point, and we should have broken from the
      // loop.
      llvm_unreachable("unsupported type");
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Reset Tracing
//===----------------------------------------------------------------------===//

/// Check whether a type contains a `ResetType`.
static bool typeContainsReset(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case<FIRRTLType>([](auto type) {
        return type.getRecursiveTypeProperties().hasUninferredReset;
      })
      .Default([](auto) { return false; });
}

/// Iterate over a circuit and follow all signals with `ResetType`, aggregating
/// them into reset nets. After this function returns, the `resetMap` is
/// populated with the reset networks in the circuit, alongside information on
/// drivers and their types that contribute to the reset.
void InferResetsPass::traceResets(CircuitOp circuit) {
  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    debugHeader("Tracing uninferred resets") << "\n\n";
  });

  SmallVector<std::pair<FModuleOp, SmallVector<Operation *>>> moduleToOps;

  for (auto module : circuit.getOps<FModuleOp>())
    moduleToOps.push_back({module, {}});

  hw::InnerRefNamespace irn{getAnalysis<SymbolTable>(),
                            getAnalysis<hw::InnerSymbolTableCollection>()};

  mlir::parallelForEach(circuit.getContext(), moduleToOps, [](auto &e) {
    e.first.walk([&](Operation *op) {
      // We are only interested in operations which are related to abstract
      // reset.
      if (llvm::any_of(
              op->getResultTypes(),
              [](mlir::Type type) { return typeContainsReset(type); }) ||
          llvm::any_of(op->getOperandTypes(), typeContainsReset))
        e.second.push_back(op);
    });
  });

  for (auto &[_, ops] : moduleToOps)
    for (auto *op : ops) {
      TypeSwitch<Operation *>(op)
          .Case<FConnectLike>([&](auto op) {
            traceResets(op.getDest(), op.getSrc(), op.getLoc());
          })
          .Case<FInstanceLike>([&](auto op) { traceResets(op); })
          .Case<RefSendOp>([&](auto op) {
            // Trace using base types.
            traceResets(op.getType().getType(), op.getResult(), 0,
                        op.getBase().getType().getPassiveType(), op.getBase(),
                        0, op.getLoc());
          })
          .Case<RefResolveOp>([&](auto op) {
            // Trace using base types.
            traceResets(op.getType(), op.getResult(), 0,
                        op.getRef().getType().getType(), op.getRef(), 0,
                        op.getLoc());
          })
          .Case<Forceable>([&](Forceable op) {
            if (auto node = dyn_cast<NodeOp>(op.getOperation()))
              traceResets(node.getResult(), node.getInput(), node.getLoc());
            // Trace reset into rwprobe.  Avoid invalid IR.
            if (op.isForceable())
              traceResets(op.getDataType(), op.getData(), 0, op.getDataType(),
                          op.getDataRef(), 0, op.getLoc());
          })
          .Case<RWProbeOp>([&](RWProbeOp op) {
            auto ist = irn.lookup(op.getTarget());
            assert(ist);
            auto ref = getFieldRefForTarget(ist);
            auto baseType = op.getType().getType();
            traceResets(baseType, op.getResult(), 0, baseType.getPassiveType(),
                        ref.getValue(), ref.getFieldID(), op.getLoc());
          })
          .Case<UninferredResetCastOp, ConstCastOp, RefCastOp,
                UnsafeDomainCastOp>([&](auto op) {
            traceResets(op.getResult(), op.getInput(), op.getLoc());
          })
          .Case<InvalidValueOp>([&](auto op) {
            // Uniquify `InvalidValueOp`s that are contributing to multiple
            // reset networks. These are tricky to handle because passes
            // like CSE will generally ensure that there is only a single
            // `InvalidValueOp` per type. However, a `reset` invalid value
            // may be connected to two reset networks that end up being
            // inferred as `asyncreset` and `uint<1>`. In that case, we need
            // a distinct `InvalidValueOp` for each reset network in order
            // to assign it the correct type.
            auto type = op.getType();
            if (!typeContainsReset(type) || op->hasOneUse() || op->use_empty())
              return;
            LLVM_DEBUG(llvm::dbgs() << "Uniquify " << op << "\n");
            ImplicitLocOpBuilder builder(op->getLoc(), op);
            for (auto &use :
                 llvm::make_early_inc_range(llvm::drop_begin(op->getUses()))) {
              // - `make_early_inc_range` since `getUses()` is invalidated
              // upon
              //   `use.set(...)`.
              // - `drop_begin` such that the first use can keep the
              // original op.
              auto newOp = InvalidValueOp::create(builder, type);
              use.set(newOp);
            }
          })

          .Case<SubfieldOp>([&](auto op) {
            // Associate the input bundle's resets with the output field's
            // resets.
            BundleType bundleType = op.getInput().getType();
            auto index = op.getFieldIndex();
            traceResets(op.getType(), op.getResult(), 0,
                        bundleType.getElements()[index].type, op.getInput(),
                        getFieldID(bundleType, index), op.getLoc());
          })

          .Case<SubindexOp, SubaccessOp>([&](auto op) {
            // Associate the input vector's resets with the output field's
            // resets.
            //
            // This collapses all elements in vectors into one shared
            // element which will ensure that reset inference provides a
            // uniform result for all elements.
            //
            // CAVEAT: This may infer reset networks that are too big, since
            // unrelated resets in the same vector end up looking as if they
            // were connected. However for the sake of type inference, this
            // is indistinguishable from them having to share the same type
            // (namely the vector element type).
            FVectorType vectorType = op.getInput().getType();
            traceResets(op.getType(), op.getResult(), 0,
                        vectorType.getElementType(), op.getInput(),
                        getFieldID(vectorType), op.getLoc());
          })

          .Case<RefSubOp>([&](RefSubOp op) {
            // Trace through ref.sub.
            auto aggType = op.getInput().getType().getType();
            uint64_t fieldID = TypeSwitch<FIRRTLBaseType, uint64_t>(aggType)
                                   .Case<FVectorType>([](auto type) {
                                     return getFieldID(type);
                                   })
                                   .Case<BundleType>([&](auto type) {
                                     return getFieldID(type, op.getIndex());
                                   });
            traceResets(op.getType(), op.getResult(), 0,
                        op.getResult().getType(), op.getInput(), fieldID,
                        op.getLoc());
          });
    }
}

/// Trace reset signals through an instance or instance choice. This essentially
/// associates the instance's port values with the target module's port values.
void InferResetsPass::traceResets(FInstanceLike inst) {
  LLVM_DEBUG(llvm::dbgs() << "Visiting instance " << inst.getInstanceName()
                          << "\n");
  auto moduleNames = inst.getReferencedModuleNamesAttr();
  for (auto moduleName : moduleNames.getAsRange<StringAttr>()) {
    auto *node = instanceGraph->lookup(moduleName);
    auto module = dyn_cast<FModuleOp>(*node->getModule());
    if (!module)
      return;

    // Establish a connection between the instance ports and module ports.
    for (const auto &it : llvm::enumerate(inst->getResults())) {
      Value dstPort = module.getArgument(it.index());
      Value srcPort = it.value();
      if (module.getPortDirection(it.index()) == Direction::Out)
        std::swap(dstPort, srcPort);
      traceResets(dstPort, srcPort, it.value().getLoc());
    }
  }
}

/// Analyze a connect of one (possibly aggregate) value to another.
/// Each drive involving a `ResetType` is recorded.
void InferResetsPass::traceResets(Value dst, Value src, Location loc) {
  // Analyze the actual connection.
  traceResets(dst.getType(), dst, 0, src.getType(), src, 0, loc);
}

/// Analyze a connect of one (possibly aggregate) value to another.
/// Each drive involving a `ResetType` is recorded.
void InferResetsPass::traceResets(Type dstType, Value dst, unsigned dstID,
                                  Type srcType, Value src, unsigned srcID,
                                  Location loc) {
  if (auto dstBundle = type_dyn_cast<BundleType>(dstType)) {
    auto srcBundle = type_cast<BundleType>(srcType);
    for (unsigned dstIdx = 0, e = dstBundle.getNumElements(); dstIdx < e;
         ++dstIdx) {
      auto dstField = dstBundle.getElements()[dstIdx].name;
      auto srcIdx = srcBundle.getElementIndex(dstField);
      if (!srcIdx)
        continue;
      auto &dstElt = dstBundle.getElements()[dstIdx];
      auto &srcElt = srcBundle.getElements()[*srcIdx];
      if (dstElt.isFlip) {
        traceResets(srcElt.type, src, srcID + getFieldID(srcBundle, *srcIdx),
                    dstElt.type, dst, dstID + getFieldID(dstBundle, dstIdx),
                    loc);
      } else {
        traceResets(dstElt.type, dst, dstID + getFieldID(dstBundle, dstIdx),
                    srcElt.type, src, srcID + getFieldID(srcBundle, *srcIdx),
                    loc);
      }
    }
    return;
  }

  if (auto dstVector = type_dyn_cast<FVectorType>(dstType)) {
    auto srcVector = type_cast<FVectorType>(srcType);
    auto srcElType = srcVector.getElementType();
    auto dstElType = dstVector.getElementType();
    // Collapse all elements into one shared element. See comment in traceResets
    // above for some context. Note that we are directly passing on the field ID
    // of the vector itself as a stand-in for its element type. This is not
    // really what `FieldRef` is designed to do, but tends to work since all the
    // places that need to reason about the resulting weird IDs are inside this
    // file. Normally you would pick a specific index from the vector, which
    // would also move the field ID forward by some amount. However, we can't
    // distinguish individual elements for the sake of type inference *and* we
    // have to support zero-length vectors for which the only available ID is
    // the vector itself. Therefore we always just pick the vector itself for
    // the field ID and make sure in `updateType` that we handle vectors
    // accordingly.
    traceResets(dstElType, dst, dstID + getFieldID(dstVector), srcElType, src,
                srcID + getFieldID(srcVector), loc);
    return;
  }

  // Handle connecting ref's.  Other uses trace using base type.
  if (auto dstRef = type_dyn_cast<RefType>(dstType)) {
    auto srcRef = type_cast<RefType>(srcType);
    return traceResets(dstRef.getType(), dst, dstID, srcRef.getType(), src,
                       srcID, loc);
  }

  // Handle reset connections.
  auto dstBase = type_dyn_cast<FIRRTLBaseType>(dstType);
  auto srcBase = type_dyn_cast<FIRRTLBaseType>(srcType);
  if (!dstBase || !srcBase)
    return;
  if (!type_isa<ResetType>(dstBase) && !type_isa<ResetType>(srcBase))
    return;

  FieldRef dstField(dst, dstID);
  FieldRef srcField(src, srcID);
  LLVM_DEBUG(llvm::dbgs() << "Visiting driver '" << dstField << "' = '"
                          << srcField << "' (" << dstType << " = " << srcType
                          << ")\n");

  // Determine the leaders for the dst and src reset networks before we make
  // the connection. This will allow us to later detect if dst got merged
  // into src, or src into dst.
  ResetSignal dstLeader =
      *resetClasses.findLeader(resetClasses.insert({dstField, dstBase}));
  ResetSignal srcLeader =
      *resetClasses.findLeader(resetClasses.insert({srcField, srcBase}));

  // Unify the two reset networks.
  ResetSignal unionLeader = *resetClasses.unionSets(dstLeader, srcLeader);
  assert(unionLeader == dstLeader || unionLeader == srcLeader);

  // If dst got merged into src, append dst's drives to src's, or vice
  // versa. Also, remove dst's or src's entry in resetDrives, because they
  // will never come up as a leader again.
  if (dstLeader != srcLeader) {
    auto &unionDrives = resetDrives[unionLeader]; // needed before finds
    auto mergedDrivesIt =
        resetDrives.find(unionLeader == dstLeader ? srcLeader : dstLeader);
    if (mergedDrivesIt != resetDrives.end()) {
      unionDrives.append(mergedDrivesIt->second);
      resetDrives.erase(mergedDrivesIt);
    }
  }

  // Keep note of this drive so we can point the user at the right location
  // in case something goes wrong.
  resetDrives[unionLeader].push_back(
      {{dstField, dstBase}, {srcField, srcBase}, loc});
}

//===----------------------------------------------------------------------===//
// Reset Inference
//===----------------------------------------------------------------------===//

LogicalResult InferResetsPass::inferAndUpdateResets() {
  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    debugHeader("Infer reset types") << "\n\n";
  });
  for (const auto &it : resetClasses) {
    if (!it->isLeader())
      continue;
    ResetNetwork net = resetClasses.members(*it);

    // Infer whether this should be a sync or async reset.
    auto kind = inferReset(net);
    if (failed(kind))
      return failure();

    // Update the types in the IR to match the inferred kind.
    if (failed(updateReset(net, *kind)))
      return failure();
  }
  return success();
}

FailureOr<ResetKind> InferResetsPass::inferReset(ResetNetwork net) {
  LLVM_DEBUG(llvm::dbgs() << "Inferring reset network with "
                          << std::distance(net.begin(), net.end())
                          << " nodes\n");

  // Go through the nodes and track the involved types.
  unsigned asyncDrives = 0;
  unsigned syncDrives = 0;
  unsigned invalidDrives = 0;
  for (ResetSignal signal : net) {
    // Keep track of whether this signal contributes a vote for async or sync.
    if (type_isa<AsyncResetType>(signal.type))
      ++asyncDrives;
    else if (type_isa<UIntType>(signal.type))
      ++syncDrives;
    else if (isUselessVec(signal.field) ||
             isa_and_nonnull<InvalidValueOp>(
                 signal.field.getValue().getDefiningOp()))
      ++invalidDrives;
  }
  LLVM_DEBUG(llvm::dbgs() << "- Found " << asyncDrives << " async, "
                          << syncDrives << " sync, " << invalidDrives
                          << " invalid drives\n");

  // Handle the case where we have no votes for either kind.
  if (asyncDrives == 0 && syncDrives == 0 && invalidDrives == 0) {
    ResetSignal root = guessRoot(net);
    auto diag = mlir::emitError(root.field.getValue().getLoc())
                << "reset network never driven with concrete type";
    for (ResetSignal signal : net)
      diag.attachNote(signal.field.getLoc()) << "here: ";
    return failure();
  }

  // Handle the case where we have votes for both kinds.
  if (asyncDrives > 0 && syncDrives > 0) {
    ResetSignal root = guessRoot(net);
    bool majorityAsync = asyncDrives >= syncDrives;
    auto diag = mlir::emitError(root.field.getValue().getLoc())
                << "reset network";
    SmallString<32> fieldName;
    if (getFieldName(root.field, fieldName))
      diag << " \"" << fieldName << "\"";
    diag << " simultaneously connected to async and sync resets";
    diag.attachNote(root.field.getValue().getLoc())
        << "majority of connections to this reset are "
        << (majorityAsync ? "async" : "sync");
    for (auto &drive : getResetDrives(net)) {
      if ((type_isa<AsyncResetType>(drive.dst.type) && !majorityAsync) ||
          (type_isa<AsyncResetType>(drive.src.type) && !majorityAsync) ||
          (type_isa<UIntType>(drive.dst.type) && majorityAsync) ||
          (type_isa<UIntType>(drive.src.type) && majorityAsync))
        diag.attachNote(drive.loc)
            << (type_isa<AsyncResetType>(drive.src.type) ? "async" : "sync")
            << " drive here:";
    }
    return failure();
  }

  // At this point we know that the type of the reset is unambiguous. If there
  // are any votes for async, we make the reset async. Otherwise we make it
  // sync.
  auto kind = (asyncDrives ? ResetKind::Async : ResetKind::Sync);
  LLVM_DEBUG(llvm::dbgs() << "- Inferred as " << kind << "\n");
  return kind;
}

//===----------------------------------------------------------------------===//
// Reset Updating
//===----------------------------------------------------------------------===//

LogicalResult InferResetsPass::updateReset(ResetNetwork net, ResetKind kind) {
  LLVM_DEBUG(llvm::dbgs() << "Updating reset network with "
                          << std::distance(net.begin(), net.end())
                          << " nodes to " << kind << "\n");

  // Determine the final type the reset should have.
  FIRRTLBaseType resetType;
  if (kind == ResetKind::Async)
    resetType = AsyncResetType::get(&getContext());
  else
    resetType = UIntType::get(&getContext(), 1);

  // Update all those values in the network that cannot be inferred from
  // operands. If we change the type of a module port (i.e. BlockArgument), add
  // the module to a module worklist since we need to update its function type.
  SmallSetVector<Operation *, 16> worklist;
  SmallDenseSet<Operation *> moduleWorklist;
  SmallDenseSet<std::pair<Operation *, Operation *>> extmoduleWorklist;
  for (auto signal : net) {
    Value value = signal.field.getValue();
    if (!isa<BlockArgument>(value) &&
        !isa_and_nonnull<WireOp, RegOp, RegResetOp, FInstanceLike,
                         InvalidValueOp, ConstCastOp, RefCastOp,
                         UninferredResetCastOp, RWProbeOp, AsResetPrimOp>(
            value.getDefiningOp()))
      continue;
    if (updateReset(signal.field, resetType)) {
      for (auto *user : value.getUsers())
        worklist.insert(user);
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        moduleWorklist.insert(blockArg.getOwner()->getParentOp());
        continue;
      }

      TypeSwitch<Operation *>(value.getDefiningOp())
          .Case<FInstanceLike>([&](FInstanceLike op) {
            for (auto moduleName : op.getReferencedModuleNamesAttr()) {
              auto *node = instanceGraph->lookup(cast<StringAttr>(moduleName));
              if (auto refModule = dyn_cast<FExtModuleOp>(*node->getModule()))
                extmoduleWorklist.insert({refModule, op.getOperation()});
            }
          })
          .Case<UninferredResetCastOp>([&](auto op) {
            op.replaceAllUsesWith(op.getInput());
            op.erase();
          })
          .Case<AsResetPrimOp>([&](auto op) {
            // Remove `asReset` casts for sync resets, or replace them with an
            // `asAsyncReset` cast for async resets.
            Value result = op.getInput();
            if (type_isa<AsyncResetType>(resetType)) {
              ImplicitLocOpBuilder builder(op.getLoc(), op);
              result = AsAsyncResetPrimOp::create(builder, op.getInput());
            }
            op.replaceAllUsesWith(result);
            op.erase();
          });
    }
  }

  // Process the worklist of operations that have their type changed, pushing
  // types down the SSA dataflow graph. This is important because we change the
  // reset types in aggregates, and then need all the subindex, subfield, and
  // subaccess operations to be updated as appropriate.
  while (!worklist.empty()) {
    auto *wop = worklist.pop_back_val();
    SmallVector<Type, 2> types;
    if (auto op = dyn_cast<InferTypeOpInterface>(wop)) {
      // Determine the new result types.
      SmallVector<Type, 2> types;
      if (failed(op.inferReturnTypes(op->getContext(), op->getLoc(),
                                     op->getOperands(), op->getAttrDictionary(),
                                     op->getPropertiesStorage(),
                                     op->getRegions(), types)))
        return failure();

      // Update the results and add the changed ones to the
      // worklist.
      for (auto it : llvm::zip(op->getResults(), types)) {
        auto newType = std::get<1>(it);
        if (std::get<0>(it).getType() == newType)
          continue;
        std::get<0>(it).setType(newType);
        for (auto *user : std::get<0>(it).getUsers())
          worklist.insert(user);
      }
      LLVM_DEBUG(llvm::dbgs() << "- Inferred " << *op << "\n");
    } else if (auto uop = dyn_cast<UninferredResetCastOp>(wop)) {
      for (auto *user : uop.getResult().getUsers())
        worklist.insert(user);
      uop.replaceAllUsesWith(uop.getInput());
      LLVM_DEBUG(llvm::dbgs() << "- Inferred " << uop << "\n");
      uop.erase();
    }
  }

  // Update module types based on the type of the block arguments.
  for (auto *op : moduleWorklist) {
    auto module = dyn_cast<FModuleOp>(op);
    if (!module)
      continue;

    SmallVector<Attribute> argTypes;
    argTypes.reserve(module.getNumPorts());
    for (auto arg : module.getArguments())
      argTypes.push_back(TypeAttr::get(arg.getType()));

    module.setPortTypesAttr(ArrayAttr::get(op->getContext(), argTypes));
    LLVM_DEBUG(llvm::dbgs()
               << "- Updated type of module '" << module.getName() << "'\n");
  }

  // Update extmodule types based on their instantiation.
  for (auto [mod, instOp] : extmoduleWorklist) {
    auto module = cast<FExtModuleOp>(mod);

    SmallVector<Attribute> types;
    for (auto type : instOp->getResultTypes())
      types.push_back(TypeAttr::get(type));

    module.setPortTypesAttr(ArrayAttr::get(module->getContext(), types));
    LLVM_DEBUG(llvm::dbgs()
               << "- Updated type of extmodule '" << module.getName() << "'\n");
  }

  return success();
}

/// Update the type of a single field within a type.
static FIRRTLBaseType updateType(FIRRTLBaseType oldType, unsigned fieldID,
                                 FIRRTLBaseType fieldType) {
  // If this is a ground type, simply replace it, preserving constness.
  if (oldType.isGround()) {
    assert(fieldID == 0);
    return fieldType.getConstType(oldType.isConst());
  }

  // If this is a bundle type, update the corresponding field.
  if (auto bundleType = type_dyn_cast<BundleType>(oldType)) {
    unsigned index = getIndexForFieldID(bundleType, fieldID);
    SmallVector<BundleType::BundleElement> fields(bundleType.begin(),
                                                  bundleType.end());
    fields[index].type = updateType(
        fields[index].type, fieldID - getFieldID(bundleType, index), fieldType);
    return BundleType::get(oldType.getContext(), fields, bundleType.isConst());
  }

  // If this is a vector type, update the element type.
  if (auto vectorType = type_dyn_cast<FVectorType>(oldType)) {
    auto newType = updateType(vectorType.getElementType(),
                              fieldID - getFieldID(vectorType), fieldType);
    return FVectorType::get(newType, vectorType.getNumElements(),
                            vectorType.isConst());
  }

  llvm_unreachable("unknown aggregate type");
  return oldType;
}

/// Update the reset type of a specific field.
bool InferResetsPass::updateReset(FieldRef field, FIRRTLBaseType resetType) {
  // Compute the updated type.
  auto oldType = type_cast<FIRRTLType>(field.getValue().getType());
  FIRRTLType newType = mapBaseType(oldType, [&](auto base) {
    return updateType(base, field.getFieldID(), resetType);
  });

  // Update the type if necessary.
  if (oldType == newType)
    return false;
  LLVM_DEBUG(llvm::dbgs() << "- Updating '" << field << "' from " << oldType
                          << " to " << newType << "\n");
  field.getValue().setType(newType);
  return true;
}

LogicalResult InferResetsPass::verifyNoAbstractReset() {
  bool hasAbstractResetPorts = false;
  for (FModuleLike module :
       getOperation().getBodyBlock()->getOps<FModuleLike>()) {
    for (PortInfo port : module.getPorts()) {
      if (getBaseOfType<ResetType>(port.type)) {
        auto diag = emitError(port.loc)
                    << "a port \"" << port.getName()
                    << "\" with abstract reset type was unable to be "
                       "inferred by InferResets (is this a top-level port?)";
        diag.attachNote(module->getLoc())
            << "the module with this uninferred reset port was defined here";
        hasAbstractResetPorts = true;
      }
    }
  }

  if (hasAbstractResetPorts)
    return failure();
  return success();
}
