//===- ElaborationPass.cpp - RTG ElaborationPass implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass elaborates the random parts of the RTG dialect.
// It performs randomization top-down, i.e., random constructs in a sequence
// that is invoked multiple times can yield different randomization results
// for each invokation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/IR/RTGVisitors.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/Support/Debug.h"
#include <queue>
#include <random>

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_ELABORATIONPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::rtg;
using llvm::MapVector;

#define DEBUG_TYPE "rtg-elaboration"

//===----------------------------------------------------------------------===//
// Uniform Distribution Helper
//
// Simplified version of
// https://github.com/llvm/llvm-project/blob/main/libcxx/include/__random/uniform_int_distribution.h
// We use our custom version here to get the same results when compiled with
// different compiler versions and standard libraries.
//===----------------------------------------------------------------------===//

static uint32_t computeMask(size_t w) {
  size_t n = w / 32 + (w % 32 != 0);
  size_t w0 = w / n;
  return w0 > 0 ? uint32_t(~0) >> (32 - w0) : 0;
}

/// Get a number uniformly at random in the in specified range.
static uint32_t getUniformlyInRange(std::mt19937 &rng, uint32_t a, uint32_t b) {
  const uint32_t diff = b - a + 1;
  if (diff == 1)
    return a;

  const uint32_t digits = std::numeric_limits<uint32_t>::digits;
  if (diff == 0)
    return rng();

  uint32_t width = digits - llvm::countl_zero(diff) - 1;
  if ((diff & (std::numeric_limits<uint32_t>::max() >> (digits - width))) != 0)
    ++width;

  uint32_t mask = computeMask(diff);
  uint32_t u;
  do {
    u = rng() & mask;
  } while (u >= diff);

  return u + a;
}

//===----------------------------------------------------------------------===//
// Elaborator Value
//===----------------------------------------------------------------------===//

namespace {
struct BagStorage;
struct SequenceStorage;
struct RandomizedSequenceStorage;
struct SetStorage;

/// Represents a unique virtual register.
struct VirtualRegister {
  VirtualRegister(uint64_t id, ArrayAttr allowedRegs)
      : id(id), allowedRegs(allowedRegs) {}

  bool operator==(const VirtualRegister &other) const {
    assert(
        id != other.id ||
        allowedRegs == other.allowedRegs &&
            "instances with the same ID must have the same allowed registers");
    return id == other.id;
  }

  // The ID of this virtual register.
  uint64_t id;

  // The list of fixed registers allowed to be selected for this virtual
  // register.
  ArrayAttr allowedRegs;
};

struct LabelValue {
  LabelValue(StringAttr name, uint64_t id = 0) : name(name), id(id) {}

  bool operator==(const LabelValue &other) const {
    return name == other.name && id == other.id;
  }

  /// The label name. For unique labels, this is just the prefix.
  StringAttr name;

  /// Standard label declarations always have id=0
  uint64_t id;
};

/// The abstract base class for elaborated values.
using ElaboratorValue =
    std::variant<TypedAttr, BagStorage *, bool, size_t, SequenceStorage *,
                 RandomizedSequenceStorage *, SetStorage *, VirtualRegister,
                 LabelValue>;

// NOLINTNEXTLINE(readability-identifier-naming)
llvm::hash_code hash_value(const VirtualRegister &val) {
  return llvm::hash_value(val.id);
}

// NOLINTNEXTLINE(readability-identifier-naming)
llvm::hash_code hash_value(const LabelValue &val) {
  return llvm::hash_combine(val.id, val.name);
}

// NOLINTNEXTLINE(readability-identifier-naming)
llvm::hash_code hash_value(const ElaboratorValue &val) {
  return std::visit(
      [&val](const auto &alternative) {
        // Include index in hash to make sure same value as different
        // alternatives don't collide.
        return llvm::hash_combine(val.index(), alternative);
      },
      val);
}

} // namespace

namespace llvm {

template <>
struct DenseMapInfo<bool> {
  static inline unsigned getEmptyKey() { return false; }
  static inline unsigned getTombstoneKey() { return true; }
  static unsigned getHashValue(const bool &val) { return val * 37U; }

  static bool isEqual(const bool &lhs, const bool &rhs) { return lhs == rhs; }
};

template <>
struct DenseMapInfo<VirtualRegister> {
  static inline VirtualRegister getEmptyKey() {
    return VirtualRegister(0, ArrayAttr());
  }
  static inline VirtualRegister getTombstoneKey() {
    return VirtualRegister(~0, ArrayAttr());
  }
  static unsigned getHashValue(const VirtualRegister &val) {
    return llvm::hash_combine(val.id, val.allowedRegs);
  }

  static bool isEqual(const VirtualRegister &lhs, const VirtualRegister &rhs) {
    return lhs == rhs;
  }
};

template <>
struct DenseMapInfo<LabelValue> {
  static inline LabelValue getEmptyKey() { return LabelValue(StringAttr(), 0); }
  static inline LabelValue getTombstoneKey() {
    return LabelValue(StringAttr(), ~0);
  }
  static unsigned getHashValue(const LabelValue &val) {
    return llvm::hash_combine(val.name, val.id);
  }

  static bool isEqual(const LabelValue &lhs, const LabelValue &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

//===----------------------------------------------------------------------===//
// Elaborator Value Storages and Internalization
//===----------------------------------------------------------------------===//

namespace {

/// Lightweight object to be used as the key for internalization sets. It caches
/// the hashcode of the internalized object and a pointer to it. This allows a
/// delayed allocation and construction of the actual object and thus only has
/// to happen if the object is not already in the set.
template <typename StorageTy>
struct HashedStorage {
  HashedStorage(unsigned hashcode = 0, StorageTy *storage = nullptr)
      : hashcode(hashcode), storage(storage) {}

  unsigned hashcode;
  StorageTy *storage;
};

/// A DenseMapInfo implementation to support 'insert_as' for the internalization
/// sets. When comparing two 'HashedStorage's we can just compare the already
/// internalized storage pointers, otherwise we have to call the costly
/// 'isEqual' method.
template <typename StorageTy>
struct StorageKeyInfo {
  static inline HashedStorage<StorageTy> getEmptyKey() {
    return HashedStorage<StorageTy>(0,
                                    DenseMapInfo<StorageTy *>::getEmptyKey());
  }
  static inline HashedStorage<StorageTy> getTombstoneKey() {
    return HashedStorage<StorageTy>(
        0, DenseMapInfo<StorageTy *>::getTombstoneKey());
  }

  static inline unsigned getHashValue(const HashedStorage<StorageTy> &key) {
    return key.hashcode;
  }
  static inline unsigned getHashValue(const StorageTy &key) {
    return key.hashcode;
  }

  static inline bool isEqual(const HashedStorage<StorageTy> &lhs,
                             const HashedStorage<StorageTy> &rhs) {
    return lhs.storage == rhs.storage;
  }
  static inline bool isEqual(const StorageTy &lhs,
                             const HashedStorage<StorageTy> &rhs) {
    if (isEqual(rhs, getEmptyKey()) || isEqual(rhs, getTombstoneKey()))
      return false;

    return lhs.isEqual(rhs.storage);
  }
};

/// Storage object for an '!rtg.set<T>'.
struct SetStorage {
  SetStorage(SetVector<ElaboratorValue> &&set, Type type)
      : hashcode(llvm::hash_combine(
            type, llvm::hash_combine_range(set.begin(), set.end()))),
        set(std::move(set)), type(type) {}

  bool isEqual(const SetStorage *other) const {
    return hashcode == other->hashcode && set == other->set &&
           type == other->type;
  }

  // The cached hashcode to avoid repeated computations.
  const unsigned hashcode;

  // Stores the elaborated values contained in the set.
  const SetVector<ElaboratorValue> set;

  // Store the set type such that we can materialize this evaluated value
  // also in the case where the set is empty.
  const Type type;
};

/// Storage object for an '!rtg.bag<T>'.
struct BagStorage {
  BagStorage(MapVector<ElaboratorValue, uint64_t> &&bag, Type type)
      : hashcode(llvm::hash_combine(
            type, llvm::hash_combine_range(bag.begin(), bag.end()))),
        bag(std::move(bag)), type(type) {}

  bool isEqual(const BagStorage *other) const {
    return hashcode == other->hashcode && llvm::equal(bag, other->bag) &&
           type == other->type;
  }

  // The cached hashcode to avoid repeated computations.
  const unsigned hashcode;

  // Stores the elaborated values contained in the bag with their number of
  // occurences.
  const MapVector<ElaboratorValue, uint64_t> bag;

  // Store the bag type such that we can materialize this evaluated value
  // also in the case where the bag is empty.
  const Type type;
};

/// Storage object for an '!rtg.sequence'.
struct SequenceStorage {
  SequenceStorage(StringAttr familyName, SmallVector<ElaboratorValue> &&args)
      : hashcode(llvm::hash_combine(
            familyName, llvm::hash_combine_range(args.begin(), args.end()))),
        familyName(familyName), args(std::move(args)) {}

  bool isEqual(const SequenceStorage *other) const {
    return hashcode == other->hashcode && familyName == other->familyName &&
           args == other->args;
  }

  // The cached hashcode to avoid repeated computations.
  const unsigned hashcode;

  // The name of the sequence family this sequence is derived from.
  const StringAttr familyName;

  // The elaborator values used during substitution of the sequence family.
  const SmallVector<ElaboratorValue> args;
};

/// Storage object for an '!rtg.randomized_sequence'.
struct RandomizedSequenceStorage {
  RandomizedSequenceStorage(StringRef name,
                            ContextResourceAttrInterface context,
                            StringAttr test, SequenceStorage *sequence)
      : hashcode(llvm::hash_combine(name, context, test, sequence)), name(name),
        context(context), test(test), sequence(sequence) {}

  bool isEqual(const RandomizedSequenceStorage *other) const {
    return hashcode == other->hashcode && name == other->name &&
           context == other->context && test == other->test &&
           sequence == other->sequence;
  }

  // The cached hashcode to avoid repeated computations.
  const unsigned hashcode;

  // The name of this fully substituted and elaborated sequence.
  const StringRef name;

  // The context under which this sequence is placed.
  const ContextResourceAttrInterface context;

  // The test in which this sequence is placed.
  const StringAttr test;

  const SequenceStorage *sequence;
};

/// An 'Internalizer' object internalizes storages and takes ownership of them.
/// When the initializer object is destroyed, all owned storages are also
/// deallocated and thus must not be accessed anymore.
class Internalizer {
public:
  /// Internalize a storage of type `StorageTy` constructed with arguments
  /// `args`. The pointers returned by this method can be used to compare
  /// objects when, e.g., computing set differences, uniquing the elements in a
  /// set, etc. Otherwise, we'd need to do a deep value comparison in those
  /// situations.
  template <typename StorageTy, typename... Args>
  StorageTy *internalize(Args &&...args) {
    StorageTy storage(std::forward<Args>(args)...);

    auto existing = getInternSet<StorageTy>().insert_as(
        HashedStorage<StorageTy>(storage.hashcode), storage);
    StorageTy *&storagePtr = existing.first->storage;
    if (existing.second)
      storagePtr =
          new (allocator.Allocate<StorageTy>()) StorageTy(std::move(storage));

    return storagePtr;
  }

private:
  template <typename StorageTy>
  DenseSet<HashedStorage<StorageTy>, StorageKeyInfo<StorageTy>> &
  getInternSet() {
    if constexpr (std::is_same_v<StorageTy, SetStorage>)
      return internedSets;
    else if constexpr (std::is_same_v<StorageTy, BagStorage>)
      return internedBags;
    else if constexpr (std::is_same_v<StorageTy, SequenceStorage>)
      return internedSequences;
    else if constexpr (std::is_same_v<StorageTy, RandomizedSequenceStorage>)
      return internedRandomizedSequences;
    else
      static_assert(!sizeof(StorageTy),
                    "no intern set available for this storage type.");
  }

  // This allocator allocates on the heap. It automatically deallocates all
  // objects it allocated once the allocator itself is destroyed.
  llvm::BumpPtrAllocator allocator;

  // The sets holding the internalized objects. We use one set per storage type
  // such that we can have a simpler equality checking function (no need to
  // compare some sort of TypeIDs).
  DenseSet<HashedStorage<SetStorage>, StorageKeyInfo<SetStorage>> internedSets;
  DenseSet<HashedStorage<BagStorage>, StorageKeyInfo<BagStorage>> internedBags;
  DenseSet<HashedStorage<SequenceStorage>, StorageKeyInfo<SequenceStorage>>
      internedSequences;
  DenseSet<HashedStorage<RandomizedSequenceStorage>,
           StorageKeyInfo<RandomizedSequenceStorage>>
      internedRandomizedSequences;
};

} // namespace

#ifndef NDEBUG

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ElaboratorValue &value);

static void print(TypedAttr val, llvm::raw_ostream &os) {
  os << "<attr " << val << ">";
}

static void print(BagStorage *val, llvm::raw_ostream &os) {
  os << "<bag {";
  llvm::interleaveComma(val->bag, os,
                        [&](const std::pair<ElaboratorValue, uint64_t> &el) {
                          os << el.first << " -> " << el.second;
                        });
  os << "} at " << val << ">";
}

static void print(bool val, llvm::raw_ostream &os) {
  os << "<bool " << (val ? "true" : "false") << ">";
}

static void print(size_t val, llvm::raw_ostream &os) {
  os << "<index " << val << ">";
}

static void print(SequenceStorage *val, llvm::raw_ostream &os) {
  os << "<sequence @" << val->familyName.getValue() << "(";
  llvm::interleaveComma(val->args, os,
                        [&](const ElaboratorValue &val) { os << val; });
  os << ") at " << val << ">";
}

static void print(RandomizedSequenceStorage *val, llvm::raw_ostream &os) {
  os << "<randomized-sequence @" << val->name << " derived from @"
     << val->sequence->familyName.getValue() << " under context "
     << val->context << " in test " << val->test << "(";
  llvm::interleaveComma(val->sequence->args, os,
                        [&](const ElaboratorValue &val) { os << val; });
  os << ") at " << val << ">";
}

static void print(SetStorage *val, llvm::raw_ostream &os) {
  os << "<set {";
  llvm::interleaveComma(val->set, os,
                        [&](const ElaboratorValue &val) { os << val; });
  os << "} at " << val << ">";
}

static void print(const VirtualRegister &val, llvm::raw_ostream &os) {
  os << "<virtual-register " << val.id << " " << val.allowedRegs << ">";
}

static void print(const LabelValue &val, llvm::raw_ostream &os) {
  os << "<label " << val.id << " " << val.name << ">";
}

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ElaboratorValue &value) {
  std::visit([&](auto val) { print(val, os); }, value);

  return os;
}

#endif

//===----------------------------------------------------------------------===//
// Elaborator Value Materialization
//===----------------------------------------------------------------------===//

namespace {

/// Construct an SSA value from a given elaborated value.
class Materializer {
public:
  Materializer(OpBuilder builder) : builder(builder) {}

  /// Materialize IR representing the provided `ElaboratorValue` and return the
  /// `Value` or a null value on failure.
  Value materialize(ElaboratorValue val, Location loc,
                    std::queue<RandomizedSequenceStorage *> &elabRequests,
                    function_ref<InFlightDiagnostic()> emitError) {
    auto iter = materializedValues.find(val);
    if (iter != materializedValues.end())
      return iter->second;

    LLVM_DEBUG(llvm::dbgs() << "Materializing " << val << "\n\n");

    return std::visit(
        [&](auto val) { return visit(val, loc, elabRequests, emitError); },
        val);
  }

  /// If `op` is not in the same region as the materializer insertion point, a
  /// clone is created at the materializer's insertion point by also
  /// materializing the `ElaboratorValue`s for each operand just before it.
  /// Otherwise, all operations after the materializer's insertion point are
  /// deleted until `op` is reached. An error is returned if the operation is
  /// before the insertion point.
  LogicalResult
  materialize(Operation *op, DenseMap<Value, ElaboratorValue> &state,
              std::queue<RandomizedSequenceStorage *> &elabRequests) {
    if (op->getNumRegions() > 0)
      return op->emitOpError("ops with nested regions must be elaborated away");

    // We don't support opaque values. If there is an SSA value that has a
    // use-site it needs an equivalent ElaborationValue representation.
    // NOTE: We could support cases where there is initially a use-site but that
    // op is guaranteed to be deleted during elaboration. Or the use-sites are
    // replaced with freshly materialized values from the ElaborationValue. But
    // then, why can't we delete the value defining op?
    for (auto res : op->getResults())
      if (!res.use_empty())
        return op->emitOpError(
            "ops with results that have uses are not supported");

    if (op->getParentRegion() == builder.getBlock()->getParent()) {
      // We are doing in-place materialization, so mark all ops deleted until we
      // reach the one to be materialized and modify it in-place.
      deleteOpsUntil([&](auto iter) { return &*iter == op; });

      if (builder.getInsertionPoint() == builder.getBlock()->end())
        return op->emitError("operation did not occur after the current "
                             "materializer insertion point");

      LLVM_DEBUG(llvm::dbgs() << "Modifying in-place: " << *op << "\n\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Materializing a clone of " << *op << "\n\n");
      op = builder.clone(*op);
      builder.setInsertionPoint(op);
    }

    for (auto &operand : op->getOpOperands()) {
      auto emitError = [&]() {
        auto diag = op->emitError();
        diag.attachNote(op->getLoc())
            << "while materializing value for operand#"
            << operand.getOperandNumber();
        return diag;
      };

      Value val = materialize(state.at(operand.get()), op->getLoc(),
                              elabRequests, emitError);
      if (!val)
        return failure();

      operand.set(val);
    }

    builder.setInsertionPointAfter(op);
    return success();
  }

  /// Should be called once the `Region` is successfully materialized. No calls
  /// to `materialize` should happen after this anymore.
  void finalize() {
    deleteOpsUntil([](auto iter) { return false; });

    for (auto *op : llvm::reverse(toDelete))
      op->erase();
  }

  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&...args) {
    return builder.create<OpTy>(location, std::forward<Args>(args)...);
  }

private:
  void deleteOpsUntil(function_ref<bool(Block::iterator)> stop) {
    auto ip = builder.getInsertionPoint();
    while (ip != builder.getBlock()->end() && !stop(ip)) {
      LLVM_DEBUG(llvm::dbgs() << "Marking to be deleted: " << *ip << "\n\n");
      toDelete.push_back(&*ip);

      builder.setInsertionPointAfter(&*ip);
      ip = builder.getInsertionPoint();
    }
  }

  Value visit(TypedAttr val, Location loc,
              std::queue<RandomizedSequenceStorage *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    // For index attributes (and arithmetic operations on them) we use the
    // index dialect.
    if (auto intAttr = dyn_cast<IntegerAttr>(val);
        intAttr && isa<IndexType>(val.getType())) {
      Value res = builder.create<index::ConstantOp>(loc, intAttr);
      materializedValues[val] = res;
      return res;
    }

    // For any other attribute, we just call the materializer of the dialect
    // defining that attribute.
    auto *op =
        val.getDialect().materializeConstant(builder, val, val.getType(), loc);
    if (!op) {
      emitError() << "materializer of dialect '"
                  << val.getDialect().getNamespace()
                  << "' unable to materialize value for attribute '" << val
                  << "'";
      return Value();
    }

    Value res = op->getResult(0);
    materializedValues[val] = res;
    return res;
  }

  Value visit(size_t val, Location loc,
              std::queue<RandomizedSequenceStorage *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    Value res = builder.create<index::ConstantOp>(loc, val);
    materializedValues[val] = res;
    return res;
  }

  Value visit(bool val, Location loc,
              std::queue<RandomizedSequenceStorage *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    Value res = builder.create<index::BoolConstantOp>(loc, val);
    materializedValues[val] = res;
    return res;
  }

  Value visit(SetStorage *val, Location loc,
              std::queue<RandomizedSequenceStorage *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> elements;
    elements.reserve(val->set.size());
    for (auto el : val->set) {
      auto materialized = materialize(el, loc, elabRequests, emitError);
      if (!materialized)
        return Value();

      elements.push_back(materialized);
    }

    auto res = builder.create<SetCreateOp>(loc, val->type, elements);
    materializedValues[val] = res;
    return res;
  }

  Value visit(BagStorage *val, Location loc,
              std::queue<RandomizedSequenceStorage *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> values, weights;
    values.reserve(val->bag.size());
    weights.reserve(val->bag.size());
    for (auto [val, weight] : val->bag) {
      auto materializedVal = materialize(val, loc, elabRequests, emitError);
      auto materializedWeight =
          materialize(weight, loc, elabRequests, emitError);
      if (!materializedVal || !materializedWeight)
        return Value();

      values.push_back(materializedVal);
      weights.push_back(materializedWeight);
    }

    auto res = builder.create<BagCreateOp>(loc, val->type, values, weights);
    materializedValues[val] = res;
    return res;
  }

  Value visit(SequenceStorage *val, Location loc,
              std::queue<RandomizedSequenceStorage *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    emitError() << "materializing a non-randomized sequence not supported yet";
    return Value();
  }

  Value visit(RandomizedSequenceStorage *val, Location loc,
              std::queue<RandomizedSequenceStorage *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    elabRequests.push(val);
    Value seq = builder.create<GetSequenceOp>(
        loc, SequenceType::get(builder.getContext(), {}), val->name);
    return builder.create<RandomizeSequenceOp>(loc, seq);
  }

  Value visit(const VirtualRegister &val, Location loc,
              std::queue<RandomizedSequenceStorage *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    auto res = builder.create<VirtualRegisterOp>(loc, val.allowedRegs);
    materializedValues[val] = res;
    return res;
  }

  Value visit(const LabelValue &val, Location loc,
              std::queue<RandomizedSequenceStorage *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    if (val.id == 0) {
      auto res = builder.create<LabelDeclOp>(loc, val.name, ValueRange());
      materializedValues[val] = res;
      return res;
    }

    auto res = builder.create<LabelUniqueDeclOp>(loc, val.name, ValueRange());
    materializedValues[val] = res;
    return res;
  }

private:
  /// Cache values we have already materialized to reuse them later. We start
  /// with an insertion point at the start of the block and cache the (updated)
  /// insertion point such that future materializations can also reuse previous
  /// materializations without running into dominance issues (or requiring
  /// additional checks to avoid them).
  DenseMap<ElaboratorValue, Value> materializedValues;

  /// Cache the builder to continue insertions at their current insertion point
  /// for the reason stated above.
  OpBuilder builder;

  SmallVector<Operation *> toDelete;
};

//===----------------------------------------------------------------------===//
// Elaboration Visitor
//===----------------------------------------------------------------------===//

/// Used to signal to the elaboration driver whether the operation should be
/// removed.
enum class DeletionKind { Keep, Delete };

/// Elaborator state that should be shared by all elaborator instances.
struct ElaboratorSharedState {
  ElaboratorSharedState(SymbolTable &table, unsigned seed)
      : table(table), rng(seed) {}

  SymbolTable &table;
  std::mt19937 rng;
  Namespace names;
  Namespace labelNames;
  Internalizer internalizer;

  /// The worklist used to keep track of the test and sequence operations to
  /// make sure they are processed top-down (BFS traversal).
  std::queue<RandomizedSequenceStorage *> worklist;

  uint64_t virtualRegisterID = 0;
  uint64_t uniqueLabelID = 1;
};

/// A collection of state per RTG test.
struct TestState {
  /// The name of the test.
  StringAttr name;

  /// The context switches registered for this test.
  MapVector<
      std::pair<ContextResourceAttrInterface, ContextResourceAttrInterface>,
      SequenceStorage *>
      contextSwitches;
};

/// Interprets the IR to perform and lower the represented randomizations.
class Elaborator : public RTGOpVisitor<Elaborator, FailureOr<DeletionKind>> {
public:
  using RTGBase = RTGOpVisitor<Elaborator, FailureOr<DeletionKind>>;
  using RTGBase::visitOp;

  Elaborator(ElaboratorSharedState &sharedState, TestState &testState,
             Materializer &materializer,
             ContextResourceAttrInterface currentContext = {})
      : sharedState(sharedState), testState(testState),
        materializer(materializer), currentContext(currentContext) {}

  template <typename ValueTy>
  inline ValueTy get(Value val) const {
    return std::get<ValueTy>(state.at(val));
  }

  FailureOr<DeletionKind> visitConstantLike(Operation *op) {
    assert(op->hasTrait<OpTrait::ConstantLike>() &&
           "op is expected to be constant-like");

    SmallVector<OpFoldResult, 1> result;
    auto foldResult = op->fold(result);
    (void)foldResult; // Make sure there is a user when assertions are off.
    assert(succeeded(foldResult) &&
           "constant folder of a constant-like must always succeed");
    auto attr = dyn_cast<TypedAttr>(result[0].dyn_cast<Attribute>());
    if (!attr)
      return op->emitError(
          "only typed attributes supported for constant-like operations");

    auto intAttr = dyn_cast<IntegerAttr>(attr);
    if (intAttr && isa<IndexType>(attr.getType()))
      state[op->getResult(0)] = size_t(intAttr.getInt());
    else if (intAttr && intAttr.getType().isSignlessInteger(1))
      state[op->getResult(0)] = bool(intAttr.getInt());
    else
      state[op->getResult(0)] = attr;

    return DeletionKind::Delete;
  }

  /// Print a nice error message for operations we don't support yet.
  FailureOr<DeletionKind> visitUnhandledOp(Operation *op) {
    return op->emitOpError("elaboration not supported");
  }

  FailureOr<DeletionKind> visitExternalOp(Operation *op) {
    if (op->hasTrait<OpTrait::ConstantLike>())
      return visitConstantLike(op);

    // TODO: we only have this to be able to write tests for this pass without
    // having to add support for more operations for now, so it should be
    // removed once it is not necessary anymore for writing tests
    if (op->use_empty())
      return DeletionKind::Keep;

    return visitUnhandledOp(op);
  }

  FailureOr<DeletionKind> visitOp(GetSequenceOp op) {
    SmallVector<ElaboratorValue> replacements;
    state[op.getResult()] =
        sharedState.internalizer.internalize<SequenceStorage>(
            op.getSequenceAttr(), std::move(replacements));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SubstituteSequenceOp op) {
    auto *seq = get<SequenceStorage *>(op.getSequence());

    SmallVector<ElaboratorValue> replacements(seq->args);
    for (auto replacement : op.getReplacements())
      replacements.push_back(state.at(replacement));

    state[op.getResult()] =
        sharedState.internalizer.internalize<SequenceStorage>(
            seq->familyName, std::move(replacements));

    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(RandomizeSequenceOp op) {
    auto *seq = get<SequenceStorage *>(op.getSequence());

    auto name = sharedState.names.newName(seq->familyName.getValue());
    state[op.getResult()] =
        sharedState.internalizer.internalize<RandomizedSequenceStorage>(
            name, currentContext, testState.name, seq);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(EmbedSequenceOp op) {
    auto *seq = get<RandomizedSequenceStorage *>(op.getSequence());
    if (seq->context != currentContext) {
      auto err = op->emitError("attempting to place sequence ")
                 << seq->name << " derived from "
                 << seq->sequence->familyName.getValue() << " under context "
                 << currentContext
                 << ", but it was previously randomized for context ";
      if (seq->context)
        err << seq->context;
      else
        err << "'default'";
      return err;
    }

    return DeletionKind::Keep;
  }

  FailureOr<DeletionKind> visitOp(SetCreateOp op) {
    SetVector<ElaboratorValue> set;
    for (auto val : op.getElements())
      set.insert(state.at(val));

    state[op.getSet()] = sharedState.internalizer.internalize<SetStorage>(
        std::move(set), op.getSet().getType());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SetSelectRandomOp op) {
    auto set = get<SetStorage *>(op.getSet())->set;

    size_t selected;
    if (auto intAttr =
            op->getAttrOfType<IntegerAttr>("rtg.elaboration_custom_seed")) {
      std::mt19937 customRng(intAttr.getInt());
      selected = getUniformlyInRange(customRng, 0, set.size() - 1);
    } else {
      selected = getUniformlyInRange(sharedState.rng, 0, set.size() - 1);
    }

    state[op.getResult()] = set[selected];
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SetDifferenceOp op) {
    auto original = get<SetStorage *>(op.getOriginal())->set;
    auto diff = get<SetStorage *>(op.getDiff())->set;

    SetVector<ElaboratorValue> result(original);
    result.set_subtract(diff);

    state[op.getResult()] = sharedState.internalizer.internalize<SetStorage>(
        std::move(result), op.getResult().getType());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SetUnionOp op) {
    SetVector<ElaboratorValue> result;
    for (auto set : op.getSets())
      result.set_union(get<SetStorage *>(set)->set);

    state[op.getResult()] = sharedState.internalizer.internalize<SetStorage>(
        std::move(result), op.getType());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SetSizeOp op) {
    auto size = get<SetStorage *>(op.getSet())->set.size();
    state[op.getResult()] = size;
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagCreateOp op) {
    MapVector<ElaboratorValue, uint64_t> bag;
    for (auto [val, multiple] :
         llvm::zip(op.getElements(), op.getMultiples())) {
      // If the multiple is not stored as an AttributeValue, the elaboration
      // must have already failed earlier (since we don't have
      // unevaluated/opaque values).
      bag[state.at(val)] += get<size_t>(multiple);
    }

    state[op.getBag()] = sharedState.internalizer.internalize<BagStorage>(
        std::move(bag), op.getType());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagSelectRandomOp op) {
    auto bag = get<BagStorage *>(op.getBag())->bag;

    SmallVector<std::pair<ElaboratorValue, uint32_t>> prefixSum;
    prefixSum.reserve(bag.size());
    uint32_t accumulator = 0;
    for (auto [val, weight] : bag) {
      accumulator += weight;
      prefixSum.push_back({val, accumulator});
    }

    auto customRng = sharedState.rng;
    if (auto intAttr =
            op->getAttrOfType<IntegerAttr>("rtg.elaboration_custom_seed")) {
      customRng = std::mt19937(intAttr.getInt());
    }

    auto idx = getUniformlyInRange(customRng, 0, accumulator - 1);
    auto *iter = llvm::upper_bound(
        prefixSum, idx,
        [](uint32_t a, const std::pair<ElaboratorValue, uint32_t> &b) {
          return a < b.second;
        });

    state[op.getResult()] = iter->first;
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagDifferenceOp op) {
    auto original = get<BagStorage *>(op.getOriginal())->bag;
    auto diff = get<BagStorage *>(op.getDiff())->bag;

    MapVector<ElaboratorValue, uint64_t> result;
    for (const auto &el : original) {
      if (!diff.contains(el.first)) {
        result.insert(el);
        continue;
      }

      if (op.getInf())
        continue;

      auto toDiff = diff.lookup(el.first);
      if (el.second <= toDiff)
        continue;

      result.insert({el.first, el.second - toDiff});
    }

    state[op.getResult()] = sharedState.internalizer.internalize<BagStorage>(
        std::move(result), op.getType());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagUnionOp op) {
    MapVector<ElaboratorValue, uint64_t> result;
    for (auto bag : op.getBags()) {
      auto val = get<BagStorage *>(bag)->bag;
      for (auto [el, multiple] : val)
        result[el] += multiple;
    }

    state[op.getResult()] = sharedState.internalizer.internalize<BagStorage>(
        std::move(result), op.getType());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagUniqueSizeOp op) {
    auto size = get<BagStorage *>(op.getBag())->bag.size();
    state[op.getResult()] = size;
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(FixedRegisterOp op) {
    return visitConstantLike(op);
  }

  FailureOr<DeletionKind> visitOp(VirtualRegisterOp op) {
    state[op.getResult()] = VirtualRegister(sharedState.virtualRegisterID++,
                                            op.getAllowedRegsAttr());
    return DeletionKind::Delete;
  }

  StringAttr substituteFormatString(StringAttr formatString,
                                    ValueRange substitutes) const {
    if (substitutes.empty() || formatString.empty())
      return formatString;

    auto original = formatString.getValue().str();
    for (auto [i, subst] : llvm::enumerate(substitutes)) {
      size_t startPos = 0;
      std::string from = "{{" + std::to_string(i) + "}}";
      while ((startPos = original.find(from, startPos)) != std::string::npos) {
        auto substString = std::to_string(get<size_t>(subst));
        original.replace(startPos, from.length(), substString);
      }
    }

    return StringAttr::get(formatString.getContext(), original);
  }

  FailureOr<DeletionKind> visitOp(LabelDeclOp op) {
    auto substituted =
        substituteFormatString(op.getFormatStringAttr(), op.getArgs());
    sharedState.labelNames.add(substituted.getValue());
    state[op.getLabel()] = LabelValue(substituted);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(LabelUniqueDeclOp op) {
    state[op.getLabel()] = LabelValue(
        substituteFormatString(op.getFormatStringAttr(), op.getArgs()),
        sharedState.uniqueLabelID++);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(LabelOp op) { return DeletionKind::Keep; }

  FailureOr<DeletionKind> visitOp(RandomNumberInRangeOp op) {
    size_t lower = get<size_t>(op.getLowerBound());
    size_t upper = get<size_t>(op.getUpperBound()) - 1;
    if (lower > upper)
      return op->emitError("cannot select a number from an empty range");

    if (auto intAttr =
            op->getAttrOfType<IntegerAttr>("rtg.elaboration_custom_seed")) {
      std::mt19937 customRng(intAttr.getInt());
      state[op.getResult()] =
          size_t(getUniformlyInRange(customRng, lower, upper));
    } else {
      state[op.getResult()] =
          size_t(getUniformlyInRange(sharedState.rng, lower, upper));
    }

    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(OnContextOp op) {
    ContextResourceAttrInterface from = currentContext,
                                 to = cast<ContextResourceAttrInterface>(
                                     get<TypedAttr>(op.getContext()));
    if (!currentContext)
      from = DefaultContextAttr::get(op->getContext(), to.getType());

    auto emitError = [&]() {
      auto diag = op.emitError();
      diag.attachNote(op.getLoc())
          << "while materializing value for context switching for " << op;
      return diag;
    };

    if (from == to) {
      Value seqVal = materializer.materialize(
          get<SequenceStorage *>(op.getSequence()), op.getLoc(),
          sharedState.worklist, emitError);
      Value randSeqVal =
          materializer.create<RandomizeSequenceOp>(op.getLoc(), seqVal);
      materializer.create<EmbedSequenceOp>(op.getLoc(), randSeqVal);
      return DeletionKind::Delete;
    }

    // Switch to the desired context.
    auto *iter = testState.contextSwitches.find({from, to});
    // NOTE: we could think about supporting context switching via intermediate
    // context, i.e., treat it as a transitive relation.
    if (iter == testState.contextSwitches.end())
      return op->emitError("no context transition registered to switch from ")
             << from << " to " << to;

    auto familyName = iter->second->familyName;
    SmallVector<ElaboratorValue> args{from, to,
                                      get<SequenceStorage *>(op.getSequence())};
    auto *seq = sharedState.internalizer.internalize<SequenceStorage>(
        familyName, std::move(args));
    auto *randSeq =
        sharedState.internalizer.internalize<RandomizedSequenceStorage>(
            sharedState.names.newName(familyName.getValue()), to,
            testState.name, seq);
    Value seqVal = materializer.materialize(randSeq, op.getLoc(),
                                            sharedState.worklist, emitError);
    materializer.create<EmbedSequenceOp>(op.getLoc(), seqVal);

    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(ContextSwitchOp op) {
    testState.contextSwitches[{op.getFromAttr(), op.getToAttr()}] =
        get<SequenceStorage *>(op.getSequence());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(scf::IfOp op) {
    bool cond = get<bool>(op.getCondition());
    auto &toElaborate = cond ? op.getThenRegion() : op.getElseRegion();
    if (toElaborate.empty())
      return DeletionKind::Delete;

    // Just reuse this elaborator for the nested region because we need access
    // to the elaborated values outside the nested region (since it is not
    // isolated from above) and we want to materialize the region inline, thus
    // don't need a new materializer instance.
    if (failed(elaborate(toElaborate)))
      return failure();

    // Map the results of the 'scf.if' to the yielded values.
    for (auto [res, out] :
         llvm::zip(op.getResults(),
                   toElaborate.front().getTerminator()->getOperands()))
      state[res] = state.at(out);

    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(scf::ForOp op) {
    if (!(std::holds_alternative<size_t>(state.at(op.getLowerBound())) &&
          std::holds_alternative<size_t>(state.at(op.getStep())) &&
          std::holds_alternative<size_t>(state.at(op.getUpperBound()))))
      return op->emitOpError("can only elaborate index type iterator");

    auto lowerBound = get<size_t>(op.getLowerBound());
    auto step = get<size_t>(op.getStep());
    auto upperBound = get<size_t>(op.getUpperBound());

    // Prepare for first iteration by assigning the nested regions block
    // arguments. We can just reuse this elaborator because we need access to
    // values elaborated in the parent region anyway and materialize everything
    // inline (i.e., don't need a new materializer).
    state[op.getInductionVar()] = lowerBound;
    for (auto [iterArg, initArg] :
         llvm::zip(op.getRegionIterArgs(), op.getInitArgs()))
      state[iterArg] = state.at(initArg);

    // This loop performs the actual 'scf.for' loop iterations.
    for (size_t i = lowerBound; i < upperBound; i += step) {
      if (failed(elaborate(op.getBodyRegion())))
        return failure();

      // Prepare for the next iteration by updating the mapping of the nested
      // regions block arguments
      state[op.getInductionVar()] = i + step;
      for (auto [iterArg, prevIterArg] :
           llvm::zip(op.getRegionIterArgs(),
                     op.getBody()->getTerminator()->getOperands()))
        state[iterArg] = state.at(prevIterArg);
    }

    // Transfer the previously yielded values to the for loop result values.
    for (auto [res, iterArg] :
         llvm::zip(op->getResults(), op.getRegionIterArgs()))
      state[res] = state.at(iterArg);

    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(scf::YieldOp op) {
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(index::AddOp op) {
    size_t lhs = get<size_t>(op.getLhs());
    size_t rhs = get<size_t>(op.getRhs());
    state[op.getResult()] = lhs + rhs;
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(index::CmpOp op) {
    size_t lhs = get<size_t>(op.getLhs());
    size_t rhs = get<size_t>(op.getRhs());
    bool result;
    switch (op.getPred()) {
    case index::IndexCmpPredicate::EQ:
      result = lhs == rhs;
      break;
    case index::IndexCmpPredicate::NE:
      result = lhs != rhs;
      break;
    case index::IndexCmpPredicate::ULT:
      result = lhs < rhs;
      break;
    case index::IndexCmpPredicate::ULE:
      result = lhs <= rhs;
      break;
    case index::IndexCmpPredicate::UGT:
      result = lhs > rhs;
      break;
    case index::IndexCmpPredicate::UGE:
      result = lhs >= rhs;
      break;
    default:
      return op->emitOpError("elaboration not supported");
    }
    state[op.getResult()] = result;
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> dispatchOpVisitor(Operation *op) {
    return TypeSwitch<Operation *, FailureOr<DeletionKind>>(op)
        .Case<
            // Index ops
            index::AddOp, index::CmpOp,
            // SCF ops
            scf::IfOp, scf::ForOp, scf::YieldOp>(
            [&](auto op) { return visitOp(op); })
        .Default([&](Operation *op) { return RTGBase::dispatchOpVisitor(op); });
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  LogicalResult elaborate(Region &region,
                          ArrayRef<ElaboratorValue> regionArguments = {}) {
    if (region.getBlocks().size() > 1)
      return region.getParentOp()->emitOpError(
          "regions with more than one block are not supported");

    for (auto [arg, elabArg] :
         llvm::zip(region.getArguments(), regionArguments))
      state[arg] = elabArg;

    Block *block = &region.front();
    for (auto &op : *block) {
      auto result = dispatchOpVisitor(&op);
      if (failed(result))
        return failure();

      if (*result == DeletionKind::Keep)
        if (failed(materializer.materialize(&op, state, sharedState.worklist)))
          return failure();

      LLVM_DEBUG({
        llvm::dbgs() << "Elaborated " << op << " to\n[";

        llvm::interleaveComma(op.getResults(), llvm::dbgs(), [&](auto res) {
          if (state.contains(res))
            llvm::dbgs() << state.at(res);
          else
            llvm::dbgs() << "unknown";
        });

        llvm::dbgs() << "]\n\n";
      });
    }

    return success();
  }

private:
  // State to be shared between all elaborator instances.
  ElaboratorSharedState &sharedState;

  // State to a specific RTG test and the sequences placed within it.
  TestState &testState;

  // Allows us to materialize ElaboratorValues to the IR operations necessary to
  // obtain an SSA value representing that elaborated value.
  Materializer &materializer;

  // A map from SSA values to a pointer of an interned elaborator value.
  DenseMap<Value, ElaboratorValue> state;

  // The current context we are elaborating under.
  ContextResourceAttrInterface currentContext;
};
} // namespace

//===----------------------------------------------------------------------===//
// Elaborator Pass
//===----------------------------------------------------------------------===//

namespace {
struct ElaborationPass
    : public rtg::impl::ElaborationPassBase<ElaborationPass> {
  using Base::Base;

  void runOnOperation() override;
  void cloneTargetsIntoTests(SymbolTable &table);
  LogicalResult elaborateModule(ModuleOp moduleOp, SymbolTable &table);
  LogicalResult inlineSequences(TestOp testOp, SymbolTable &table);
};
} // namespace

void ElaborationPass::runOnOperation() {
  auto moduleOp = getOperation();
  SymbolTable table(moduleOp);

  cloneTargetsIntoTests(table);

  if (failed(elaborateModule(moduleOp, table)))
    return signalPassFailure();
}

void ElaborationPass::cloneTargetsIntoTests(SymbolTable &table) {
  auto moduleOp = getOperation();
  for (auto target : llvm::make_early_inc_range(moduleOp.getOps<TargetOp>())) {
    for (auto test : moduleOp.getOps<TestOp>()) {
      // If the test requires nothing from a target, we can always run it.
      if (test.getTarget().getEntries().empty())
        continue;

      // If the target requirements do not match, skip this test
      // TODO: allow target refinements, just not coarsening
      if (target.getTarget() != test.getTarget())
        continue;

      IRRewriter rewriter(test);
      // Create a new test for the matched target
      auto newTest = cast<TestOp>(test->clone());
      newTest.setSymName(test.getSymName().str() + "_" +
                         target.getSymName().str());
      table.insert(newTest, rewriter.getInsertionPoint());

      // Copy the target body into the newly created test
      IRMapping mapping;
      rewriter.setInsertionPointToStart(newTest.getBody());
      for (auto &op : target.getBody()->without_terminator())
        rewriter.clone(op, mapping);

      for (auto [returnVal, result] :
           llvm::zip(target.getBody()->getTerminator()->getOperands(),
                     newTest.getBody()->getArguments()))
        result.replaceAllUsesWith(mapping.lookup(returnVal));

      newTest.getBody()->eraseArguments(0,
                                        newTest.getBody()->getNumArguments());
      newTest.setTarget(DictType::get(&getContext(), {}));
    }

    target->erase();
  }

  // Erase all remaining non-matched tests.
  for (auto test : llvm::make_early_inc_range(moduleOp.getOps<TestOp>()))
    if (!test.getTarget().getEntries().empty())
      test->erase();
}

LogicalResult ElaborationPass::elaborateModule(ModuleOp moduleOp,
                                               SymbolTable &table) {
  ElaboratorSharedState state(table, seed);

  // Update the name cache
  state.names.add(moduleOp);

  // Initialize the worklist with the test ops since they cannot be placed by
  // other ops.
  DenseMap<StringAttr, TestState> testStates;
  for (auto testOp : moduleOp.getOps<TestOp>()) {
    LLVM_DEBUG(llvm::dbgs()
               << "\n=== Elaborating test @" << testOp.getSymName() << "\n\n");
    Materializer materializer(OpBuilder::atBlockBegin(testOp.getBody()));
    testStates[testOp.getSymNameAttr()].name = testOp.getSymNameAttr();
    Elaborator elaborator(state, testStates[testOp.getSymNameAttr()],
                          materializer);
    if (failed(elaborator.elaborate(testOp.getBodyRegion())))
      return failure();

    materializer.finalize();
  }

  // Do top-down BFS traversal such that elaborating a sequence further down
  // does not fix the outcome for multiple placements.
  while (!state.worklist.empty()) {
    auto *curr = state.worklist.front();
    state.worklist.pop();

    if (table.lookup<SequenceOp>(curr->name))
      continue;

    auto familyOp = table.lookup<SequenceOp>(curr->sequence->familyName);
    // TODO: don't clone if this is the only remaining reference to this
    // sequence
    OpBuilder builder(familyOp);
    auto seqOp = builder.cloneWithoutRegions(familyOp);
    seqOp.getBodyRegion().emplaceBlock();
    seqOp.setSymName(curr->name);
    table.insert(seqOp);
    assert(seqOp.getSymName() == curr->name && "should not have been renamed");

    LLVM_DEBUG(llvm::dbgs()
               << "\n=== Elaborating sequence family @" << familyOp.getSymName()
               << " into @" << seqOp.getSymName() << " under context "
               << curr->context << "\n\n");

    Materializer materializer(OpBuilder::atBlockBegin(seqOp.getBody()));
    Elaborator elaborator(state, testStates[curr->test], materializer,
                          curr->context);
    if (failed(elaborator.elaborate(familyOp.getBodyRegion(),
                                    curr->sequence->args)))
      return failure();

    materializer.finalize();
  }

  for (auto testOp : moduleOp.getOps<TestOp>()) {
    // Inline all sequences and remove the operations that place the sequences.
    if (failed(inlineSequences(testOp, table)))
      return failure();

    // Convert 'rtg.label_unique_decl' to 'rtg.label_decl' by choosing a unique
    // name based on the set of names we collected during elaboration.
    for (auto labelOp :
         llvm::make_early_inc_range(testOp.getOps<LabelUniqueDeclOp>())) {
      IRRewriter rewriter(labelOp);
      auto newName = state.labelNames.newName(labelOp.getFormatString());
      rewriter.replaceOpWithNewOp<LabelDeclOp>(labelOp, newName, ValueRange());
    }
  }

  // Remove all sequences since they are not accessible from the outside and
  // are not needed anymore since we fully inlined them.
  for (auto seqOp : llvm::make_early_inc_range(moduleOp.getOps<SequenceOp>()))
    seqOp->erase();

  return success();
}

LogicalResult ElaborationPass::inlineSequences(TestOp testOp,
                                               SymbolTable &table) {
  OpBuilder builder(testOp);
  for (auto iter = testOp.getBody()->begin();
       iter != testOp.getBody()->end();) {
    auto embedOp = dyn_cast<EmbedSequenceOp>(&*iter);
    if (!embedOp) {
      ++iter;
      continue;
    }

    auto randSeqOp = embedOp.getSequence().getDefiningOp<RandomizeSequenceOp>();
    if (!randSeqOp)
      return embedOp->emitError("sequence operand not directly defined by "
                                "'rtg.randomize_sequence' op");
    auto getSeqOp = randSeqOp.getSequence().getDefiningOp<GetSequenceOp>();
    if (!getSeqOp)
      return randSeqOp->emitError(
          "sequence operand not directly defined by 'rtg.get_sequence' op");

    auto seqOp = table.lookup<SequenceOp>(getSeqOp.getSequenceAttr());

    builder.setInsertionPointAfter(embedOp);
    IRMapping mapping;
    for (auto &op : *seqOp.getBody())
      builder.clone(op, mapping);

    (iter++)->erase();

    if (randSeqOp->use_empty())
      randSeqOp->erase();

    if (getSeqOp->use_empty())
      getSeqOp->erase();
  }

  return success();
}
