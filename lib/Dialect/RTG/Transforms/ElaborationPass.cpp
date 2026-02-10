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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/Support/Debug.h"
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
struct ArrayStorage;
struct BagStorage;
struct SequenceStorage;
struct RandomizedSequenceStorage;
struct InterleavedSequenceStorage;
struct SetStorage;
struct VirtualRegisterStorage;
struct UniqueLabelStorage;
struct TupleStorage;
struct MemoryStorage;
struct MemoryBlockStorage;
struct SymbolicComputationWithIdentityStorage;
struct SymbolicComputationWithIdentityValue;
struct SymbolicComputationStorage;

/// The abstract base class for elaborated values.
using ElaboratorValue =
    std::variant<TypedAttr, BagStorage *, bool, size_t, SequenceStorage *,
                 RandomizedSequenceStorage *, InterleavedSequenceStorage *,
                 SetStorage *, VirtualRegisterStorage *, UniqueLabelStorage *,
                 ArrayStorage *, TupleStorage *, MemoryStorage *,
                 MemoryBlockStorage *, SymbolicComputationWithIdentityStorage *,
                 SymbolicComputationWithIdentityValue *,
                 SymbolicComputationStorage *>;

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

// Values with structural equivalence intended to be internalized.
//===----------------------------------------------------------------------===//

/// Storage object for an '!rtg.set<T>'.
struct SetStorage {
  static unsigned computeHash(const SetVector<ElaboratorValue> &set,
                              Type type) {
    llvm::hash_code setHash = 0;
    for (auto el : set) {
      // Just XOR all hashes because it's a commutative operation and
      // `llvm::hash_combine_range` is not commutative.
      // We don't want the order in which elements were added to influence the
      // hash and thus the equivalence of sets.
      setHash = setHash ^ llvm::hash_combine(el);
    }
    return llvm::hash_combine(type, setHash);
  }

  SetStorage(SetVector<ElaboratorValue> &&set, Type type)
      : hashcode(computeHash(set, type)), set(std::move(set)), type(type) {}

  bool isEqual(const SetStorage *other) const {
    // Note: we are not using the `==` operator of `SetVector` because it
    // takes the order in which elements were added into account (since it's a
    // vector after all). We just use it as a convenient way to keep track of a
    // deterministic order for re-materialization.
    bool allContained = true;
    for (auto el : set)
      allContained &= other->set.contains(el);

    return hashcode == other->hashcode && set.size() == other->set.size() &&
           allContained && type == other->type;
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

/// Storage object for interleaved '!rtg.randomized_sequence'es.
struct InterleavedSequenceStorage {
  InterleavedSequenceStorage(SmallVector<ElaboratorValue> &&sequences,
                             uint32_t batchSize)
      : sequences(std::move(sequences)), batchSize(batchSize),
        hashcode(llvm::hash_combine(
            llvm::hash_combine_range(sequences.begin(), sequences.end()),
            batchSize)) {}

  explicit InterleavedSequenceStorage(RandomizedSequenceStorage *sequence)
      : sequences(SmallVector<ElaboratorValue>(1, sequence)), batchSize(1),
        hashcode(llvm::hash_combine(
            llvm::hash_combine_range(sequences.begin(), sequences.end()),
            batchSize)) {}

  bool isEqual(const InterleavedSequenceStorage *other) const {
    return hashcode == other->hashcode && sequences == other->sequences &&
           batchSize == other->batchSize;
  }

  const SmallVector<ElaboratorValue> sequences;

  const uint32_t batchSize;

  // The cached hashcode to avoid repeated computations.
  const unsigned hashcode;
};

/// Storage object for '!rtg.array`-typed values.
struct ArrayStorage {
  ArrayStorage(Type type, SmallVector<ElaboratorValue> &&array)
      : hashcode(llvm::hash_combine(
            type, llvm::hash_combine_range(array.begin(), array.end()))),
        type(type), array(array) {}

  bool isEqual(const ArrayStorage *other) const {
    return hashcode == other->hashcode && type == other->type &&
           array == other->array;
  }

  // The cached hashcode to avoid repeated computations.
  const unsigned hashcode;

  /// The type of the array. This is necessary because an array of size 0
  /// cannot be reconstructed without knowing the original element type.
  const Type type;

  /// The label name. For unique labels, this is just the prefix.
  const SmallVector<ElaboratorValue> array;
};

/// Storage object for 'tuple`-typed values.
struct TupleStorage {
  TupleStorage(SmallVector<ElaboratorValue> &&values)
      : hashcode(llvm::hash_combine_range(values.begin(), values.end())),
        values(std::move(values)) {}

  bool isEqual(const TupleStorage *other) const {
    return hashcode == other->hashcode && values == other->values;
  }

  // The cached hashcode to avoid repeated computations.
  const unsigned hashcode;

  const SmallVector<ElaboratorValue> values;
};

struct SymbolicComputationStorage {
  SymbolicComputationStorage(const DenseMap<Value, ElaboratorValue> &state,
                             Operation *op)
      : name(op->getName()), resultTypes(op->getResultTypes()),
        operands(llvm::map_range(op->getOperands(),
                                 [&](Value v) { return state.lookup(v); })),
        attributes(op->getAttrDictionary()),
        properties(op->getPropertiesAsAttribute()),
        hashcode(llvm::hash_combine(name, llvm::hash_combine_range(resultTypes),
                                    llvm::hash_combine_range(operands),
                                    attributes, op->hashProperties())) {}

  bool isEqual(const SymbolicComputationStorage *other) const {
    return hashcode == other->hashcode && name == other->name &&
           resultTypes == other->resultTypes && operands == other->operands &&
           attributes == other->attributes && properties == other->properties;
  }

  const OperationName name;
  const SmallVector<Type> resultTypes;
  const SmallVector<ElaboratorValue> operands;
  const DictionaryAttr attributes;
  const Attribute properties;
  const unsigned hashcode;
};

// Values with identity not intended to be internalized.
//===----------------------------------------------------------------------===//

/// Base class for storages that represent values with identity, i.e., two
/// values are not considered equivalent if they are structurally the same, but
/// each definition of such a value is unique. E.g., unique labels or virtual
/// registers. These cannot be materialized anew in each nested sequence, but
/// must be passed as arguments.
struct IdentityValue {

  IdentityValue(Type type, Location loc) : type(type), loc(loc) {}

#ifndef NDEBUG

  /// In debug mode, track whether this value was already materialized to
  /// assert if it's illegally materialized multiple times.
  ///
  /// Instead of deleting operations defining these values and materializing
  /// them again, we could retain the operations. However, we still need
  /// specific storages to represent these values in some cases, e.g., to get
  /// the size of a memory allocation. Also, elaboration of nested control-flow
  /// regions (e.g. `scf.for`) relies on materialization of such values lazily
  /// instead of cloning the operations eagerly.
  bool alreadyMaterialized = false;

#endif

  const Type type;
  const Location loc;
};

/// Represents a unique virtual register.
struct VirtualRegisterStorage : IdentityValue {
  VirtualRegisterStorage(VirtualRegisterConfigAttr allowedRegs, Type type,
                         Location loc)
      : IdentityValue(type, loc), allowedRegs(allowedRegs) {}

  // NOTE: we don't need an 'isEqual' function and 'hashcode' here because
  // VirtualRegisters are never internalized.

  // The list of fixed registers allowed to be selected for this virtual
  // register.
  const VirtualRegisterConfigAttr allowedRegs;
};

struct UniqueLabelStorage : IdentityValue {
  UniqueLabelStorage(const ElaboratorValue &name, Location loc)
      : IdentityValue(LabelType::get(loc->getContext()), loc), name(name) {}

  // NOTE: we don't need an 'isEqual' function and 'hashcode' here because
  // VirtualRegisters are never internalized.

  /// The label name. For unique labels, this is just the prefix.
  const ElaboratorValue name;
};

/// Storage object for '!rtg.isa.memoryblock`-typed values.
struct MemoryBlockStorage : IdentityValue {
  MemoryBlockStorage(const APInt &baseAddress, const APInt &endAddress,
                     Type type, Location loc)
      : IdentityValue(type, loc), baseAddress(baseAddress),
        endAddress(endAddress) {}

  // The base address of the memory. The width of the APInt also represents the
  // address width of the memory. This is an APInt to support memories of
  // >64-bit machines.
  const APInt baseAddress;

  // The last address of the memory.
  const APInt endAddress;
};

/// Storage object for '!rtg.isa.memory`-typed values.
struct MemoryStorage : IdentityValue {
  MemoryStorage(MemoryBlockStorage *memoryBlock, size_t size, size_t alignment,
                Location loc)
      : IdentityValue(MemoryType::get(memoryBlock->type.getContext(),
                                      memoryBlock->baseAddress.getBitWidth()),
                      loc),
        memoryBlock(memoryBlock), size(size), alignment(alignment) {}

  MemoryBlockStorage *memoryBlock;
  const size_t size;
  const size_t alignment;
};

/// Storage object for an '!rtg.randomized_sequence'.
struct RandomizedSequenceStorage : IdentityValue {
  RandomizedSequenceStorage(ContextResourceAttrInterface context,
                            SequenceStorage *sequence, Location loc)
      : IdentityValue(
            RandomizedSequenceType::get(sequence->familyName.getContext()),
            loc),
        context(context), sequence(sequence) {}

  // The context under which this sequence is placed.
  const ContextResourceAttrInterface context;

  const SequenceStorage *sequence;
};

/// Operation must have at least 1 result.
struct SymbolicComputationWithIdentityStorage : IdentityValue {
  SymbolicComputationWithIdentityStorage(
      const DenseMap<Value, ElaboratorValue> &state, Operation *op)
      : IdentityValue(op->getResult(0).getType(), op->getLoc()),
        name(op->getName()), resultTypes(op->getResultTypes()),
        operands(llvm::map_range(op->getOperands(),
                                 [&](Value v) { return state.lookup(v); })),
        attributes(op->getAttrDictionary()),
        properties(op->getPropertiesAsAttribute()) {}

  const OperationName name;
  const SmallVector<Type> resultTypes;
  const SmallVector<ElaboratorValue> operands;
  const DictionaryAttr attributes;
  const Attribute properties;
};

struct SymbolicComputationWithIdentityValue : IdentityValue {
  SymbolicComputationWithIdentityValue(
      Type type, const SymbolicComputationWithIdentityStorage *storage,
      unsigned idx)
      : IdentityValue(type, storage->loc), storage(storage), idx(idx) {
    assert(
        idx != 0 &&
        "Use SymbolicComputationWithIdentityStorage for result with index 0.");
  }

  const SymbolicComputationWithIdentityStorage *storage;
  const unsigned idx;
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
    static_assert(!std::is_base_of_v<IdentityValue, StorageTy> &&
                  "values with identity must not be internalized");

    StorageTy storage(std::forward<Args>(args)...);

    auto existing = getInternSet<StorageTy>().insert_as(
        HashedStorage<StorageTy>(storage.hashcode), storage);
    StorageTy *&storagePtr = existing.first->storage;
    if (existing.second)
      storagePtr =
          new (allocator.Allocate<StorageTy>()) StorageTy(std::move(storage));

    return storagePtr;
  }

  template <typename StorageTy, typename... Args>
  StorageTy *create(Args &&...args) {
    static_assert(std::is_base_of_v<IdentityValue, StorageTy> &&
                  "values with structural equivalence must be internalized");

    return new (allocator.Allocate<StorageTy>())
        StorageTy(std::forward<Args>(args)...);
  }

private:
  template <typename StorageTy>
  DenseSet<HashedStorage<StorageTy>, StorageKeyInfo<StorageTy>> &
  getInternSet() {
    if constexpr (std::is_same_v<StorageTy, ArrayStorage>)
      return internedArrays;
    else if constexpr (std::is_same_v<StorageTy, SetStorage>)
      return internedSets;
    else if constexpr (std::is_same_v<StorageTy, BagStorage>)
      return internedBags;
    else if constexpr (std::is_same_v<StorageTy, SequenceStorage>)
      return internedSequences;
    else if constexpr (std::is_same_v<StorageTy, RandomizedSequenceStorage>)
      return internedRandomizedSequences;
    else if constexpr (std::is_same_v<StorageTy, InterleavedSequenceStorage>)
      return internedInterleavedSequences;
    else if constexpr (std::is_same_v<StorageTy, TupleStorage>)
      return internedTuples;
    else if constexpr (std::is_same_v<StorageTy, SymbolicComputationStorage>)
      return internedSymbolicComputationWithIdentityValues;
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
  DenseSet<HashedStorage<ArrayStorage>, StorageKeyInfo<ArrayStorage>>
      internedArrays;
  DenseSet<HashedStorage<SetStorage>, StorageKeyInfo<SetStorage>> internedSets;
  DenseSet<HashedStorage<BagStorage>, StorageKeyInfo<BagStorage>> internedBags;
  DenseSet<HashedStorage<SequenceStorage>, StorageKeyInfo<SequenceStorage>>
      internedSequences;
  DenseSet<HashedStorage<RandomizedSequenceStorage>,
           StorageKeyInfo<RandomizedSequenceStorage>>
      internedRandomizedSequences;
  DenseSet<HashedStorage<InterleavedSequenceStorage>,
           StorageKeyInfo<InterleavedSequenceStorage>>
      internedInterleavedSequences;
  DenseSet<HashedStorage<TupleStorage>, StorageKeyInfo<TupleStorage>>
      internedTuples;
  DenseSet<HashedStorage<SymbolicComputationStorage>,
           StorageKeyInfo<SymbolicComputationStorage>>
      internedSymbolicComputationWithIdentityValues;
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
  os << "<randomized-sequence derived from @"
     << val->sequence->familyName.getValue() << " under context "
     << val->context << "(";
  llvm::interleaveComma(val->sequence->args, os,
                        [&](const ElaboratorValue &val) { os << val; });
  os << ") at " << val << ">";
}

static void print(InterleavedSequenceStorage *val, llvm::raw_ostream &os) {
  os << "<interleaved-sequence [";
  llvm::interleaveComma(val->sequences, os,
                        [&](const ElaboratorValue &val) { os << val; });
  os << "] batch-size " << val->batchSize << " at " << val << ">";
}

static void print(ArrayStorage *val, llvm::raw_ostream &os) {
  os << "<array [";
  llvm::interleaveComma(val->array, os,
                        [&](const ElaboratorValue &val) { os << val; });
  os << "] at " << val << ">";
}

static void print(SetStorage *val, llvm::raw_ostream &os) {
  os << "<set {";
  llvm::interleaveComma(val->set, os,
                        [&](const ElaboratorValue &val) { os << val; });
  os << "} at " << val << ">";
}

static void print(const VirtualRegisterStorage *val, llvm::raw_ostream &os) {
  os << "<virtual-register " << val << " " << val->allowedRegs << ">";
}

static void print(const UniqueLabelStorage *val, llvm::raw_ostream &os) {
  os << "<unique-label " << val << " " << val->name << ">";
}

static void print(const TupleStorage *val, llvm::raw_ostream &os) {
  os << "<tuple (";
  llvm::interleaveComma(val->values, os,
                        [&](const ElaboratorValue &val) { os << val; });
  os << ")>";
}

static void print(const MemoryStorage *val, llvm::raw_ostream &os) {
  os << "<memory {" << ElaboratorValue(val->memoryBlock)
     << ", size=" << val->size << ", alignment=" << val->alignment << "}>";
}

static void print(const MemoryBlockStorage *val, llvm::raw_ostream &os) {
  os << "<memory-block {"
     << ", address-width=" << val->baseAddress.getBitWidth()
     << ", base-address=" << val->baseAddress
     << ", end-address=" << val->endAddress << "}>";
}

static void print(const SymbolicComputationWithIdentityValue *val,
                  llvm::raw_ostream &os) {
  os << "<symbolic-computation-with-identity-value (" << val->storage << ") at "
     << val->idx << ">";
}

static void print(const SymbolicComputationWithIdentityStorage *val,
                  llvm::raw_ostream &os) {
  os << "<symbolic-computation-with-identity " << val->name << "(";
  llvm::interleaveComma(val->operands, os,
                        [&](const ElaboratorValue &val) { os << val; });
  os << ") -> " << val->resultTypes << " with attributes " << val->attributes
     << " and properties " << val->properties;
  os << ">";
}

static void print(const SymbolicComputationStorage *val,
                  llvm::raw_ostream &os) {
  os << "<symbolic-computation " << val->name << "(";
  llvm::interleaveComma(val->operands, os,
                        [&](const ElaboratorValue &val) { os << val; });
  os << ") -> " << val->resultTypes << " with attributes " << val->attributes
     << " and properties " << val->properties;
  os << ">";
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

/// State that should be shared by all elaborator and materializer instances.
struct SharedState {
  SharedState(SymbolTable &table, unsigned seed) : table(table), rng(seed) {}

  SymbolTable &table;
  std::mt19937 rng;
  Namespace names;
  Internalizer internalizer;
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

/// Construct an SSA value from a given elaborated value.
class Materializer {
public:
  Materializer(OpBuilder builder, TestState &testState,
               SharedState &sharedState,
               SmallVector<ElaboratorValue> &blockArgs)
      : builder(builder), testState(testState), sharedState(sharedState),
        blockArgs(blockArgs) {}

  /// Materialize IR representing the provided `ElaboratorValue` and return the
  /// `Value` or a null value on failure.
  Value materialize(ElaboratorValue val, Location loc,
                    function_ref<InFlightDiagnostic()> emitError) {
    auto iter = materializedValues.find(val);
    if (iter != materializedValues.end())
      return iter->second;

    LLVM_DEBUG(llvm::dbgs() << "Materializing " << val);

    // In debug mode, track whether values with identity were already
    // materialized before and assert in such a situation.
    Value res = std::visit(
        [&](auto value) {
          if constexpr (std::is_base_of_v<IdentityValue,
                                          std::remove_pointer_t<
                                              std::decay_t<decltype(value)>>>) {
            if (identityValueRoot.contains(value)) {
#ifndef NDEBUG
              bool &materialized =
                  static_cast<IdentityValue *>(value)->alreadyMaterialized;
              assert(!materialized && "must not already be materialized");
              materialized = true;
#endif

              return visit(value, loc, emitError);
            }

            Value arg = builder.getBlock()->addArgument(value->type, loc);
            blockArgs.push_back(val);
            blockArgTypes.push_back(arg.getType());
            materializedValues[val] = arg;
            return arg;
          }

          return visit(value, loc, emitError);
        },
        val);

    LLVM_DEBUG(llvm::dbgs() << " to\n" << res << "\n\n");

    return res;
  }

  /// If `op` is not in the same region as the materializer insertion point, a
  /// clone is created at the materializer's insertion point by also
  /// materializing the `ElaboratorValue`s for each operand just before it.
  /// Otherwise, all operations after the materializer's insertion point are
  /// deleted until `op` is reached. An error is returned if the operation is
  /// before the insertion point.
  LogicalResult materialize(Operation *op,
                            DenseMap<Value, ElaboratorValue> &state) {
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

      auto elabVal = state.at(operand.get());
      Value val = materialize(elabVal, op->getLoc(), emitError);
      if (!val)
        return failure();

      state[val] = elabVal;
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

  /// Tell this materializer that it is responsible for materializing the given
  /// identity value at the earliest position it is needed, and should't
  /// request the value via block argument.
  void registerIdentityValue(IdentityValue *val) {
    identityValueRoot.insert(val);
  }

  ArrayRef<Type> getBlockArgTypes() const { return blockArgTypes; }

  void map(ElaboratorValue eval, Value val) { materializedValues[eval] = val; }

  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&...args) {
    return OpTy::create(builder, location, std::forward<Args>(args)...);
  }

private:
  SequenceOp elaborateSequence(const RandomizedSequenceStorage *seq,
                               SmallVector<ElaboratorValue> &elabArgs);

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
              function_ref<InFlightDiagnostic()> emitError) {
    Value res = ConstantOp::create(builder, loc, val);
    materializedValues[val] = res;
    return res;
  }

  Value visit(size_t val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    Value res = index::ConstantOp::create(builder, loc, val);
    materializedValues[val] = res;
    return res;
  }

  Value visit(bool val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    Value res = index::BoolConstantOp::create(builder, loc, val);
    materializedValues[val] = res;
    return res;
  }

  Value visit(ArrayStorage *val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> elements;
    elements.reserve(val->array.size());
    for (auto el : val->array) {
      auto materialized = materialize(el, loc, emitError);
      if (!materialized)
        return Value();

      elements.push_back(materialized);
    }

    Value res = ArrayCreateOp::create(builder, loc, val->type, elements);
    materializedValues[val] = res;
    return res;
  }

  Value visit(SetStorage *val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> elements;
    elements.reserve(val->set.size());
    for (auto el : val->set) {
      auto materialized = materialize(el, loc, emitError);
      if (!materialized)
        return Value();

      elements.push_back(materialized);
    }

    auto res = SetCreateOp::create(builder, loc, val->type, elements);
    materializedValues[val] = res;
    return res;
  }

  Value visit(BagStorage *val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> values, weights;
    values.reserve(val->bag.size());
    weights.reserve(val->bag.size());
    for (auto [val, weight] : val->bag) {
      auto materializedVal = materialize(val, loc, emitError);
      auto materializedWeight = materialize(weight, loc, emitError);
      if (!materializedVal || !materializedWeight)
        return Value();

      values.push_back(materializedVal);
      weights.push_back(materializedWeight);
    }

    auto res = BagCreateOp::create(builder, loc, val->type, values, weights);
    materializedValues[val] = res;
    return res;
  }

  Value visit(MemoryBlockStorage *val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    auto intType = builder.getIntegerType(val->baseAddress.getBitWidth());
    Value res = MemoryBlockDeclareOp::create(
        builder, val->loc, val->type,
        IntegerAttr::get(intType, val->baseAddress),
        IntegerAttr::get(intType, val->endAddress));
    materializedValues[val] = res;
    return res;
  }

  Value visit(MemoryStorage *val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    auto memBlock = materialize(val->memoryBlock, val->loc, emitError);
    auto memSize = materialize(val->size, val->loc, emitError);
    auto memAlign = materialize(val->alignment, val->loc, emitError);
    if (!(memBlock && memSize && memAlign))
      return {};

    Value res =
        MemoryAllocOp::create(builder, val->loc, memBlock, memSize, memAlign);
    materializedValues[val] = res;
    return res;
  }

  Value visit(SequenceStorage *val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    emitError() << "materializing a non-randomized sequence not supported yet";
    return Value();
  }

  Value visit(RandomizedSequenceStorage *val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    // To know which values we have to pass by argument (and not just pass all
    // that migth be used eagerly), we have to elaborate the sequence family if
    // not already done so.
    // We need to get back the sequence to reference, and the list of elaborated
    // values to pass as arguments.
    SmallVector<ElaboratorValue> elabArgs;
    // NOTE: we wouldn't need to elaborate the sequence if it doesn't contain
    // randomness to be elaborated.
    SequenceOp seqOp = elaborateSequence(val, elabArgs);
    if (!seqOp)
      return {};

    // Materialize all the values we need to pass as arguments and collect their
    // types.
    SmallVector<Value> args;
    SmallVector<Type> argTypes;
    for (auto arg : elabArgs) {
      Value materialized = materialize(arg, val->loc, emitError);
      if (!materialized)
        return {};

      args.push_back(materialized);
      argTypes.push_back(materialized.getType());
    }

    Value res = GetSequenceOp::create(
        builder, val->loc, SequenceType::get(builder.getContext(), argTypes),
        seqOp.getSymName());

    // Only materialize a substitute_sequence op when we have arguments to
    // substitute since this op does not support 0 arguments.
    if (!args.empty())
      res = SubstituteSequenceOp::create(builder, val->loc, res, args);

    res = RandomizeSequenceOp::create(builder, val->loc, res);

    materializedValues[val] = res;
    return res;
  }

  Value visit(InterleavedSequenceStorage *val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> sequences;
    for (auto seqVal : val->sequences) {
      Value materialized = materialize(seqVal, loc, emitError);
      if (!materialized)
        return {};

      sequences.push_back(materialized);
    }

    if (sequences.size() == 1)
      return sequences[0];

    Value res =
        InterleaveSequencesOp::create(builder, loc, sequences, val->batchSize);
    materializedValues[val] = res;
    return res;
  }

  Value visit(VirtualRegisterStorage *val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    Value res = VirtualRegisterOp::create(builder, val->loc, val->allowedRegs);
    materializedValues[val] = res;
    return res;
  }

  Value visit(UniqueLabelStorage *val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    auto materialized = materialize(val->name, val->loc, emitError);
    if (!materialized)
      return {};
    Value res = LabelUniqueDeclOp::create(builder, val->loc, materialized);
    materializedValues[val] = res;
    return res;
  }

  Value visit(TupleStorage *val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> materialized;
    materialized.reserve(val->values.size());
    for (auto v : val->values)
      materialized.push_back(materialize(v, loc, emitError));
    Value res = TupleCreateOp::create(builder, loc, materialized);
    materializedValues[val] = res;
    return res;
  }

  Value visit(SymbolicComputationWithIdentityValue *val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    auto *noConstStorage =
        const_cast<SymbolicComputationWithIdentityStorage *>(val->storage);
    auto res0 = materialize(noConstStorage, loc, emitError);
    if (!res0)
      return {};

    auto *op = res0.getDefiningOp();
    auto res = op->getResults()[val->idx];
    materializedValues[val] = res;
    return res;
  }

  Value visit(SymbolicComputationWithIdentityStorage *val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> operands;
    for (auto operand : val->operands) {
      auto materialized = materialize(operand, val->loc, emitError);
      if (!materialized)
        return {};

      operands.push_back(materialized);
    }

    OperationState state(val->loc, val->name);
    state.addTypes(val->resultTypes);
    state.attributes = val->attributes;
    state.propertiesAttr = val->properties;
    state.addOperands(operands);
    auto *op = builder.create(state);

    materializedValues[val] = op->getResult(0);
    return op->getResult(0);
  }

  Value visit(SymbolicComputationStorage *val, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> operands;
    for (auto operand : val->operands) {
      auto materialized = materialize(operand, loc, emitError);
      if (!materialized)
        return {};

      operands.push_back(materialized);
    }

    OperationState state(loc, val->name);
    state.addTypes(val->resultTypes);
    state.attributes = val->attributes;
    state.propertiesAttr = val->properties;
    state.addOperands(operands);
    auto *op = builder.create(state);

    for (auto res : op->getResults())
      materializedValues[val] = res;

    return op->getResult(0);
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

  TestState &testState;
  SharedState &sharedState;

  /// Keep track of the block arguments we had to add to this materializer's
  /// block for identity values and also remember which elaborator values are
  /// expected to be passed as arguments from outside.
  SmallVector<ElaboratorValue> &blockArgs;
  SmallVector<Type> blockArgTypes;

  /// Identity values in this set are materialized by this materializer,
  /// otherwise they are added as block arguments and the block that wants to
  /// embed this sequence is expected to provide a value for it.
  DenseSet<IdentityValue *> identityValueRoot;
};

//===----------------------------------------------------------------------===//
// Elaboration Visitor
//===----------------------------------------------------------------------===//

/// Used to signal to the elaboration driver whether the operation should be
/// removed.
enum class DeletionKind { Keep, Delete };

/// Interprets the IR to perform and lower the represented randomizations.
class Elaborator : public RTGOpVisitor<Elaborator, FailureOr<DeletionKind>> {
public:
  using RTGBase = RTGOpVisitor<Elaborator, FailureOr<DeletionKind>>;
  using RTGBase::visitOp;

  Elaborator(SharedState &sharedState, TestState &testState,
             Materializer &materializer,
             ContextResourceAttrInterface currentContext = {})
      : sharedState(sharedState), testState(testState),
        materializer(materializer), currentContext(currentContext) {}

  template <typename ValueTy>
  inline ValueTy get(Value val) const {
    return std::get<ValueTy>(state.at(val));
  }

  /// Print a nice error message for operations we don't support yet.
  FailureOr<DeletionKind> visitUnhandledOp(Operation *op) {
    return visitOpGeneric(op);
  }

  FailureOr<DeletionKind> visitExternalOp(Operation *op) {
    return visitOpGeneric(op);
  }

  FailureOr<DeletionKind> visitOp(GetSequenceOp op) {
    SmallVector<ElaboratorValue> replacements;
    state[op.getResult()] =
        sharedState.internalizer.internalize<SequenceStorage>(
            op.getSequenceAttr().getAttr(), std::move(replacements));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SubstituteSequenceOp op) {
    if (isSymbolic(state.at(op.getSequence())))
      return visitOpGeneric(op);

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
    auto *randomizedSeq =
        sharedState.internalizer.create<RandomizedSequenceStorage>(
            currentContext, seq, op.getLoc());
    materializer.registerIdentityValue(randomizedSeq);
    state[op.getResult()] =
        sharedState.internalizer.internalize<InterleavedSequenceStorage>(
            randomizedSeq);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(InterleaveSequencesOp op) {
    SmallVector<ElaboratorValue> sequences;
    for (auto seq : op.getSequences())
      sequences.push_back(state.at(seq));

    state[op.getResult()] =
        sharedState.internalizer.internalize<InterleavedSequenceStorage>(
            std::move(sequences), op.getBatchSize());
    return DeletionKind::Delete;
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  LogicalResult isValidContext(ElaboratorValue value, Operation *op) const {
    if (std::holds_alternative<RandomizedSequenceStorage *>(value)) {
      auto *seq = std::get<RandomizedSequenceStorage *>(value);
      if (seq->context != currentContext) {
        auto err = op->emitError("attempting to place sequence derived from ")
                   << seq->sequence->familyName.getValue() << " under context "
                   << currentContext
                   << ", but it was previously randomized for context ";
        if (seq->context)
          err << seq->context;
        else
          err << "'default'";
        return err;
      }
      return success();
    }

    auto *interVal = std::get<InterleavedSequenceStorage *>(value);
    for (auto val : interVal->sequences)
      if (failed(isValidContext(val, op)))
        return failure();
    return success();
  }

  FailureOr<DeletionKind> visitOp(EmbedSequenceOp op) {
    auto *seqVal = get<InterleavedSequenceStorage *>(op.getSequence());
    if (failed(isValidContext(seqVal, op)))
      return failure();

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

    if (set.empty())
      return op->emitError("cannot select from an empty set");

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

  // {a0,a1} x {b0,b1} x {c0,c1} -> {(a0), (a1)} -> {(a0,b0), (a0,b1), (a1,b0),
  // (a1,b1)} -> {(a0,b0,c0), (a0,b0,c1), (a0,b1,c0), (a0,b1,c1), (a1,b0,c0),
  // (a1,b0,c1), (a1,b1,c0), (a1,b1,c1)}
  FailureOr<DeletionKind> visitOp(SetCartesianProductOp op) {
    SetVector<ElaboratorValue> result;
    SmallVector<SmallVector<ElaboratorValue>> tuples;
    tuples.push_back({});

    for (auto input : op.getInputs()) {
      auto &set = get<SetStorage *>(input)->set;
      if (set.empty()) {
        SetVector<ElaboratorValue> empty;
        state[op.getResult()] =
            sharedState.internalizer.internalize<SetStorage>(std::move(empty),
                                                             op.getType());
        return DeletionKind::Delete;
      }

      for (unsigned i = 0, e = tuples.size(); i < e; ++i) {
        for (auto setEl : set.getArrayRef().drop_back()) {
          tuples.push_back(tuples[i]);
          tuples.back().push_back(setEl);
        }
        tuples[i].push_back(set.back());
      }
    }

    for (auto &tup : tuples)
      result.insert(
          sharedState.internalizer.internalize<TupleStorage>(std::move(tup)));

    state[op.getResult()] = sharedState.internalizer.internalize<SetStorage>(
        std::move(result), op.getType());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SetConvertToBagOp op) {
    auto set = get<SetStorage *>(op.getInput())->set;
    MapVector<ElaboratorValue, uint64_t> bag;
    for (auto val : set)
      bag.insert({val, 1});
    state[op.getResult()] = sharedState.internalizer.internalize<BagStorage>(
        std::move(bag), op.getType());
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

    if (bag.empty())
      return op->emitError("cannot select from an empty bag");

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

  FailureOr<DeletionKind> visitOp(BagConvertToSetOp op) {
    auto bag = get<BagStorage *>(op.getInput())->bag;
    SetVector<ElaboratorValue> set;
    for (auto [k, v] : bag)
      set.insert(k);
    state[op.getResult()] = sharedState.internalizer.internalize<SetStorage>(
        std::move(set), op.getType());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(VirtualRegisterOp op) {
    auto *val = sharedState.internalizer.create<VirtualRegisterStorage>(
        op.getAllowedRegsAttr(), op.getType(), op.getLoc());
    state[op.getResult()] = val;
    materializer.registerIdentityValue(val);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(ArrayCreateOp op) {
    SmallVector<ElaboratorValue> array;
    array.reserve(op.getElements().size());
    for (auto val : op.getElements())
      array.emplace_back(state.at(val));

    state[op.getResult()] = sharedState.internalizer.internalize<ArrayStorage>(
        op.getResult().getType(), std::move(array));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(ArrayExtractOp op) {
    auto array = get<ArrayStorage *>(op.getArray())->array;
    size_t idx = get<size_t>(op.getIndex());

    if (array.size() <= idx)
      return op->emitError("invalid to access index ")
             << idx << " of an array with " << array.size() << " elements";

    state[op.getResult()] = array[idx];
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(ArrayInjectOp op) {
    auto arrayOpaque = state.at(op.getArray());
    auto idxOpaque = state.at(op.getIndex());
    if (isSymbolic(arrayOpaque) || isSymbolic(idxOpaque))
      return visitOpGeneric(op);

    auto array = std::get<ArrayStorage *>(arrayOpaque)->array;
    size_t idx = std::get<size_t>(idxOpaque);

    if (array.size() <= idx)
      return op->emitError("invalid to access index ")
             << idx << " of an array with " << array.size() << " elements";

    array[idx] = state.at(op.getValue());
    state[op.getResult()] = sharedState.internalizer.internalize<ArrayStorage>(
        op.getResult().getType(), std::move(array));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(ArraySizeOp op) {
    auto array = get<ArrayStorage *>(op.getArray())->array;
    state[op.getResult()] = array.size();
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(LabelUniqueDeclOp op) {
    auto *val = sharedState.internalizer.create<UniqueLabelStorage>(
        state.at(op.getNamePrefix()), op.getLoc());
    state[op.getLabel()] = val;
    materializer.registerIdentityValue(val);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(RandomNumberInRangeOp op) {
    size_t lower = get<size_t>(op.getLowerBound());
    size_t upper = get<size_t>(op.getUpperBound());
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

  FailureOr<DeletionKind> visitOp(IntToImmediateOp op) {
    size_t input = get<size_t>(op.getInput());
    auto width = op.getType().getWidth();
    auto emitError = [&]() { return op->emitError(); };
    if (input > APInt::getAllOnes(width).getZExtValue())
      return emitError() << "cannot represent " << input << " with " << width
                         << " bits";

    state[op.getResult()] =
        ImmediateAttr::get(op.getContext(), APInt(width, input));
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
          get<SequenceStorage *>(op.getSequence()), op.getLoc(), emitError);
      if (!seqVal)
        return failure();

      Value randSeqVal =
          materializer.create<RandomizeSequenceOp>(op.getLoc(), seqVal);
      materializer.create<EmbedSequenceOp>(op.getLoc(), randSeqVal);
      return DeletionKind::Delete;
    }

    // Switch to the desired context.
    // First, check if a context switch is registered that has the concrete
    // context as source and target.
    auto *iter = testState.contextSwitches.find({from, to});

    // Try with 'any' context as target and the concrete context as source.
    if (iter == testState.contextSwitches.end())
      iter = testState.contextSwitches.find(
          {from, AnyContextAttr::get(op->getContext(), to.getType())});

    // Try with 'any' context as source and the concrete context as target.
    if (iter == testState.contextSwitches.end())
      iter = testState.contextSwitches.find(
          {AnyContextAttr::get(op->getContext(), from.getType()), to});

    // Try with 'any' context for both the source and the target.
    if (iter == testState.contextSwitches.end())
      iter = testState.contextSwitches.find(
          {AnyContextAttr::get(op->getContext(), from.getType()),
           AnyContextAttr::get(op->getContext(), to.getType())});

    // Otherwise, fail with an error because we couldn't find a user
    // specification on how to switch between the requested contexts.
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
    auto *randSeq = sharedState.internalizer.create<RandomizedSequenceStorage>(
        to, seq, op.getLoc());
    materializer.registerIdentityValue(randSeq);
    Value seqVal = materializer.materialize(randSeq, op.getLoc(), emitError);
    if (!seqVal)
      return failure();

    materializer.create<EmbedSequenceOp>(op.getLoc(), seqVal);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(ContextSwitchOp op) {
    testState.contextSwitches[{op.getFromAttr(), op.getToAttr()}] =
        get<SequenceStorage *>(op.getSequence());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(MemoryBlockDeclareOp op) {
    auto *val = sharedState.internalizer.create<MemoryBlockStorage>(
        op.getBaseAddress(), op.getEndAddress(), op.getType(), op.getLoc());
    state[op.getResult()] = val;
    materializer.registerIdentityValue(val);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(MemoryAllocOp op) {
    size_t size = get<size_t>(op.getSize());
    size_t alignment = get<size_t>(op.getAlignment());
    auto *memBlock = get<MemoryBlockStorage *>(op.getMemoryBlock());
    auto *val = sharedState.internalizer.create<MemoryStorage>(
        memBlock, size, alignment, op.getLoc());
    state[op.getResult()] = val;
    materializer.registerIdentityValue(val);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(MemorySizeOp op) {
    auto *memory = get<MemoryStorage *>(op.getMemory());
    state[op.getResult()] = memory->size;
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(TupleCreateOp op) {
    SmallVector<ElaboratorValue> values;
    values.reserve(op.getElements().size());
    for (auto el : op.getElements())
      values.push_back(state.at(el));

    state[op.getResult()] =
        sharedState.internalizer.internalize<TupleStorage>(std::move(values));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(TupleExtractOp op) {
    auto *tuple = get<TupleStorage *>(op.getTuple());
    state[op.getResult()] = tuple->values[op.getIndex().getZExtValue()];
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
    SmallVector<ElaboratorValue> yieldedVals;
    if (failed(elaborate(toElaborate, {}, yieldedVals)))
      return failure();

    // Map the results of the 'scf.if' to the yielded values.
    for (auto [res, out] : llvm::zip(op.getResults(), yieldedVals))
      state[res] = out;

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
    SmallVector<ElaboratorValue> yieldedVals;
    for (size_t i = lowerBound; i < upperBound; i += step) {
      yieldedVals.clear();
      if (failed(elaborate(op.getBodyRegion(), {}, yieldedVals)))
        return failure();

      // Prepare for the next iteration by updating the mapping of the nested
      // regions block arguments
      state[op.getInductionVar()] = i + step;
      for (auto [iterArg, prevIterArg] :
           llvm::zip(op.getRegionIterArgs(), yieldedVals))
        state[iterArg] = prevIterArg;
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

  FailureOr<DeletionKind> visitOp(arith::AddIOp op) {
    if (!isa<IndexType>(op.getType()))
      return visitOpGeneric(op);

    size_t lhs = get<size_t>(op.getLhs());
    size_t rhs = get<size_t>(op.getRhs());
    state[op.getResult()] = lhs + rhs;
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(arith::AndIOp op) {
    if (!op.getType().isSignlessInteger(1))
      return visitOpGeneric(op);

    bool lhs = get<bool>(op.getLhs());
    bool rhs = get<bool>(op.getRhs());
    state[op.getResult()] = lhs && rhs;
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(arith::XOrIOp op) {
    if (!op.getType().isSignlessInteger(1))
      return visitOpGeneric(op);

    bool lhs = get<bool>(op.getLhs());
    bool rhs = get<bool>(op.getRhs());
    state[op.getResult()] = lhs != rhs;
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(arith::OrIOp op) {
    if (!op.getType().isSignlessInteger(1))
      return visitOpGeneric(op);

    bool lhs = get<bool>(op.getLhs());
    bool rhs = get<bool>(op.getRhs());
    state[op.getResult()] = lhs || rhs;
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(arith::SelectOp op) {
    auto condOpaque = state.at(op.getCondition());
    if (isSymbolic(condOpaque))
      return visitOpGeneric(op);

    bool cond = std::get<bool>(condOpaque);
    auto trueVal = state.at(op.getTrueValue());
    auto falseVal = state.at(op.getFalseValue());
    state[op.getResult()] = cond ? trueVal : falseVal;
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

  bool isSymbolic(ElaboratorValue val) {
    return std::holds_alternative<SymbolicComputationWithIdentityValue *>(
               val) ||
           std::holds_alternative<SymbolicComputationWithIdentityStorage *>(
               val) ||
           std::holds_alternative<SymbolicComputationStorage *>(val);
  }

  bool isSymbolic(Operation *op) {
    return llvm::any_of(op->getOperands(), [&](auto operand) {
      auto val = state.at(operand);
      return isSymbolic(val);
    });
  }

  /// If all operands are constants, try to fold the operation and register its
  /// result values with the folded results in the elaborator state.
  /// Returns 'false' if operation has to be handled symbolically.
  bool attemptConcreteCase(Operation *op) {
    if (op->getNumResults() == 0)
      return false;

    SmallVector<Attribute> operands;
    for (auto operand : op->getOperands()) {
      auto evalValue = state[operand];
      if (std::holds_alternative<TypedAttr>(evalValue))
        operands.push_back(std::get<TypedAttr>(evalValue));
      else if (std::holds_alternative<size_t>(evalValue))
        operands.push_back(IntegerAttr::get(IndexType::get(op->getContext()),
                                            std::get<size_t>(evalValue)));
      else if (std::holds_alternative<bool>(evalValue))
        operands.push_back(
            BoolAttr::get(op->getContext(), std::get<bool>(evalValue)));
      else
        operands.push_back(Attribute());
    }

    SmallVector<OpFoldResult> results;
    if (failed(op->fold(operands, results)))
      return false;

    if (results.size() != op->getNumResults())
      return false;

    for (auto [res, val] : llvm::zip(results, op->getResults())) {
      auto attr = llvm::dyn_cast_or_null<TypedAttr>(res.dyn_cast<Attribute>());
      if (!attr)
        return false;

      if (attr.getType() != val.getType())
        return false;

      auto intAttr = dyn_cast<IntegerAttr>(attr);
      if (intAttr && isa<IndexType>(attr.getType()))
        state[op->getResult(0)] = size_t(intAttr.getInt());
      else if (intAttr && intAttr.getType().isSignlessInteger(1))
        state[op->getResult(0)] = bool(intAttr.getInt());
      else
        state[op->getResult(0)] = attr;
    }

    return true;
  }

  FailureOr<DeletionKind> visitOpGeneric(Operation *op) {
    if (op->getNumResults() == 0)
      return DeletionKind::Keep;

    if (attemptConcreteCase(op))
      return DeletionKind::Delete;

    if (mlir::isMemoryEffectFree(op)) {
      if (op->getNumResults() != 1)
        return op->emitOpError(
            "symbolic elaboration of memory-effect-free operations with "
            "multiple results not supported");

      state[op->getResult(0)] =
          sharedState.internalizer.internalize<SymbolicComputationStorage>(
              state, op);
      return DeletionKind::Delete;
    }

    // We assume that reordering operations with only allocate effects is
    // allowed.
    // FIXME: this is not how the MLIR MemoryEffects interface intends it.
    // We should create our own interface/trait for that use-case.
    // Or modify the elaboration pass to keep track of the ordering of such
    // instructions and materialize all operations that are not already
    // materialized but have to happen before the current alloc operation to be
    // materialized.
    bool onlyAlloc = mlir::hasSingleEffect<mlir::MemoryEffects::Allocate>(op);
    onlyAlloc |= isa<ValidateOp>(op);

    auto *validationVal =
        sharedState.internalizer.create<SymbolicComputationWithIdentityStorage>(
            state, op);
    materializer.registerIdentityValue(validationVal);
    state[op->getResult(0)] = validationVal;

    for (auto [i, res] : llvm::enumerate(op->getResults())) {
      if (i == 0)
        continue;
      auto *val =
          sharedState.internalizer.create<SymbolicComputationWithIdentityValue>(
              res.getType(), validationVal, i);
      state[res] = val;
      materializer.registerIdentityValue(val);
    }
    return onlyAlloc ? DeletionKind::Delete : DeletionKind::Keep;
  }

  bool supportsSymbolicValuesNonGenerically(Operation *op) {
    return isa<SubstituteSequenceOp, ArrayCreateOp, ArrayInjectOp,
               TupleCreateOp, arith::SelectOp>(op);
  }

  FailureOr<DeletionKind> dispatchOpVisitor(Operation *op) {
    if (isSymbolic(op) && !supportsSymbolicValuesNonGenerically(op))
      return visitOpGeneric(op);

    return TypeSwitch<Operation *, FailureOr<DeletionKind>>(op)
        .Case<
            // Arith ops
            arith::AddIOp, arith::XOrIOp, arith::AndIOp, arith::OrIOp,
            arith::SelectOp,
            // Index ops
            index::AddOp, index::CmpOp,
            // SCF ops
            scf::IfOp, scf::ForOp, scf::YieldOp>(
            [&](auto op) { return visitOp(op); })
        .Default([&](Operation *op) { return RTGBase::dispatchOpVisitor(op); });
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  LogicalResult elaborate(Region &region,
                          ArrayRef<ElaboratorValue> regionArguments,
                          SmallVector<ElaboratorValue> &terminatorOperands) {
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
        if (failed(materializer.materialize(&op, state)))
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

    if (region.front().mightHaveTerminator())
      for (auto val : region.front().getTerminator()->getOperands())
        terminatorOperands.push_back(state.at(val));

    return success();
  }

private:
  // State to be shared between all elaborator instances.
  SharedState &sharedState;

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

SequenceOp
Materializer::elaborateSequence(const RandomizedSequenceStorage *seq,
                                SmallVector<ElaboratorValue> &elabArgs) {
  auto familyOp =
      sharedState.table.lookup<SequenceOp>(seq->sequence->familyName);
  // TODO: don't clone if this is the only remaining reference to this
  // sequence
  OpBuilder builder(familyOp);
  auto seqOp = builder.cloneWithoutRegions(familyOp);
  auto name = sharedState.names.newName(seq->sequence->familyName.getValue());
  seqOp.setSymName(name);
  seqOp.getBodyRegion().emplaceBlock();
  sharedState.table.insert(seqOp);
  assert(seqOp.getSymName() == name && "should not have been renamed");

  LLVM_DEBUG(llvm::dbgs() << "\n=== Elaborating sequence family @"
                          << familyOp.getSymName() << " into @"
                          << seqOp.getSymName() << " under context "
                          << seq->context << "\n\n");

  Materializer materializer(OpBuilder::atBlockBegin(seqOp.getBody()), testState,
                            sharedState, elabArgs);
  Elaborator elaborator(sharedState, testState, materializer, seq->context);
  SmallVector<ElaboratorValue> yieldedVals;
  if (failed(elaborator.elaborate(familyOp.getBodyRegion(), seq->sequence->args,
                                  yieldedVals)))
    return {};

  seqOp.setSequenceType(
      SequenceType::get(builder.getContext(), materializer.getBlockArgTypes()));
  materializer.finalize();

  return seqOp;
}

//===----------------------------------------------------------------------===//
// Elaborator Pass
//===----------------------------------------------------------------------===//

namespace {
struct ElaborationPass
    : public rtg::impl::ElaborationPassBase<ElaborationPass> {
  using Base::Base;

  void runOnOperation() override;
  void matchTestsAgainstTargets(SymbolTable &table);
  LogicalResult elaborateModule(ModuleOp moduleOp, SymbolTable &table);
};
} // namespace

void ElaborationPass::runOnOperation() {
  auto moduleOp = getOperation();
  SymbolTable table(moduleOp);

  matchTestsAgainstTargets(table);

  if (failed(elaborateModule(moduleOp, table)))
    return signalPassFailure();
}

void ElaborationPass::matchTestsAgainstTargets(SymbolTable &table) {
  auto moduleOp = getOperation();

  for (auto test : llvm::make_early_inc_range(moduleOp.getOps<TestOp>())) {
    if (test.getTargetAttr())
      continue;

    bool matched = false;

    for (auto target : moduleOp.getOps<TargetOp>()) {
      // Check if the target type is a subtype of the test's target type
      // This means that for each entry in the test's target type, there must be
      // a corresponding entry with the same name and type in the target's type
      bool isSubtype = true;
      auto testEntries = test.getTargetType().getEntries();
      auto targetEntries = target.getTarget().getEntries();

      // Check if target is a subtype of test requirements
      // Since entries are sorted by name, we can do this in a single pass
      size_t targetIdx = 0;
      for (auto testEntry : testEntries) {
        // Find the matching entry in target entries.
        while (targetIdx < targetEntries.size() &&
               targetEntries[targetIdx].name.getValue() <
                   testEntry.name.getValue())
          targetIdx++;

        // Check if we found a matching entry with the same name and type
        if (targetIdx >= targetEntries.size() ||
            targetEntries[targetIdx].name != testEntry.name ||
            targetEntries[targetIdx].type != testEntry.type) {
          isSubtype = false;
          break;
        }
      }

      if (!isSubtype)
        continue;

      IRRewriter rewriter(test);
      // Create a new test for the matched target
      auto newTest = cast<TestOp>(test->clone());
      newTest.setSymName(test.getSymName().str() + "_" +
                         target.getSymName().str());

      // Set the target symbol specifying that this test is only suitable for
      // that target.
      newTest.setTargetAttr(target.getSymNameAttr());

      table.insert(newTest, rewriter.getInsertionPoint());
      matched = true;
    }

    if (matched || deleteUnmatchedTests)
      test->erase();
  }
}

static bool onlyLegalToMaterializeInTarget(Type type) {
  return isa<MemoryBlockType, ContextResourceTypeInterface>(type);
}

LogicalResult ElaborationPass::elaborateModule(ModuleOp moduleOp,
                                               SymbolTable &table) {
  SharedState state(table, seed);

  // Update the name cache
  state.names.add(moduleOp);

  struct TargetElabResult {
    DictType targetType;
    SmallVector<ElaboratorValue> yields;
    TestState testState;
  };

  // Map to store elaborated targets
  DenseMap<StringAttr, TargetElabResult> targetMap;
  for (auto targetOp : moduleOp.getOps<TargetOp>()) {
    LLVM_DEBUG(llvm::dbgs() << "=== Elaborating target @"
                            << targetOp.getSymName() << "\n\n");

    auto &result = targetMap[targetOp.getSymNameAttr()];
    result.targetType = targetOp.getTarget();

    SmallVector<ElaboratorValue> blockArgs;
    Materializer targetMaterializer(OpBuilder::atBlockBegin(targetOp.getBody()),
                                    result.testState, state, blockArgs);
    Elaborator targetElaborator(state, result.testState, targetMaterializer);

    // Elaborate the target
    if (failed(targetElaborator.elaborate(targetOp.getBodyRegion(), {},
                                          result.yields)))
      return failure();
  }

  // Initialize the worklist with the test ops since they cannot be placed by
  // other ops.
  for (auto testOp : moduleOp.getOps<TestOp>()) {
    // Skip tests without a target attribute - these couldn't be matched
    // against any target but can be useful to keep around for reporting
    // purposes.
    if (!testOp.getTargetAttr())
      continue;

    LLVM_DEBUG(llvm::dbgs()
               << "\n=== Elaborating test @" << testOp.getTemplateName()
               << " for target @" << *testOp.getTarget() << "\n\n");

    // Get the target for this test
    auto targetResult = targetMap[testOp.getTargetAttr()];
    TestState testState = targetResult.testState;
    testState.name = testOp.getSymNameAttr();

    SmallVector<ElaboratorValue> filteredYields;
    unsigned i = 0;
    for (auto [entry, yield] :
         llvm::zip(targetResult.targetType.getEntries(), targetResult.yields)) {
      if (i >= testOp.getTargetType().getEntries().size())
        break;

      if (entry.name == testOp.getTargetType().getEntries()[i].name) {
        filteredYields.push_back(yield);
        ++i;
      }
    }

    // Now elaborate the test with the same state, passing the target yield
    // values as arguments
    SmallVector<ElaboratorValue> blockArgs;
    Materializer materializer(OpBuilder::atBlockBegin(testOp.getBody()),
                              testState, state, blockArgs);

    for (auto [arg, val] :
         llvm::zip(testOp.getBody()->getArguments(), filteredYields))
      if (onlyLegalToMaterializeInTarget(arg.getType()))
        materializer.map(val, arg);

    Elaborator elaborator(state, testState, materializer);
    SmallVector<ElaboratorValue> ignore;
    if (failed(elaborator.elaborate(testOp.getBodyRegion(), filteredYields,
                                    ignore)))
      return failure();

    materializer.finalize();
  }

  return success();
}
