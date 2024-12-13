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

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/IR/RTGVisitors.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/FoldingSet.h"
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
// Elaborator Values
//===----------------------------------------------------------------------===//

namespace {

/// The abstract base class for elaborated values.
class ElaboratorValue {
public:
  enum class ValueKind {
    Attribute = 0U,
    Set,
    Bag,
    Sequence,
    Index,
    Bool,
    None
  };

  union StorageTy {
    StorageTy() : ptr(nullptr) {}
    StorageTy(const void *ptr) : ptr(ptr) {}
    StorageTy(size_t index) : index(index) {}
    StorageTy(bool boolean) : boolean(boolean) {}

    const void *ptr;
    size_t index;
    bool boolean;
  };

  ElaboratorValue(ValueKind kind = ValueKind::None,
                  StorageTy storage = StorageTy())
      : kind(kind), storage(storage) {}

  // This constructor is needed for LLVM RTTI
  ElaboratorValue(StorageTy storage) : ElaboratorValue() {}

  llvm::hash_code getHashValue() const {
    switch (kind) {
    case ValueKind::Attribute:
    case ValueKind::Set:
    case ValueKind::Bag:
    case ValueKind::Sequence:
      return llvm::hash_combine(kind, storage.ptr);
    case ValueKind::Index:
      return llvm::hash_combine(kind, storage.index);
    case ValueKind::Bool:
      return llvm::hash_combine(kind, storage.boolean);
    case ValueKind::None:
      return llvm::hash_value(kind);
    }
    llvm::llvm_unreachable_internal("all cases handled above");
  }

  bool operator==(const ElaboratorValue &other) const {
    if (kind != other.kind)
      return false;

    switch (kind) {
    case ValueKind::Attribute:
    case ValueKind::Set:
    case ValueKind::Bag:
    case ValueKind::Sequence:
      return storage.ptr == other.storage.ptr;
    case ValueKind::Index:
      return storage.index == other.storage.index;
    case ValueKind::Bool:
      return storage.boolean == other.storage.boolean;
    case ValueKind::None:
      return true;
    }
    llvm::llvm_unreachable_internal("all cases handled above");
  }

  operator bool() const { return kind != ValueKind::None; }

  ValueKind getKind() const { return kind; }
  StorageTy getStorage() const { return storage; }

private:
  ValueKind kind;
  StorageTy storage;
};

} // namespace

namespace llvm {
/// Add support for llvm style casts. We provide a cast between To and From if
/// From is mlir::Attribute or derives from it.
template <typename To, typename From>
struct CastInfo<To, From,
                std::enable_if_t<std::is_same_v<ElaboratorValue,
                                                std::remove_const_t<From>> ||
                                 std::is_base_of_v<ElaboratorValue, From>>>
    : DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {
  /// Arguments are taken as mlir::Attribute here and not as `From`, because
  /// when casting from an intermediate type of the hierarchy to one of its
  /// children, the val.getTypeID() inside T::classof will use the static
  /// getTypeID of the parent instead of the non-static Type::getTypeID that
  /// returns the dynamic ID. This means that T::classof would end up comparing
  /// the static TypeID of the children to the static TypeID of its parent,
  /// making it impossible to downcast from the parent to the child.
  static inline bool isPossible(ElaboratorValue ty) {
    /// Return a constant true instead of a dynamic true when casting to self or
    /// up the hierarchy.
    if constexpr (std::is_base_of_v<To, From>) {
      return true;
    } else {
      return To::classof(ty);
    }
  }
  static inline To doCast(ElaboratorValue value) {
    return To(value.getStorage());
  }
  static To castFailed() { return To(); }
};

template <>
struct DenseMapInfo<ElaboratorValue> {
  static inline ElaboratorValue getEmptyKey() { return ElaboratorValue(); }
  static inline ElaboratorValue getTombstoneKey() {
    return ElaboratorValue(ElaboratorValue::ValueKind::None,
                           reinterpret_cast<const void *>(~0ULL));
  }
  static unsigned getHashValue(const ElaboratorValue &value) {
    return value.getHashValue();
  }
  static bool isEqual(const ElaboratorValue &lhs, const ElaboratorValue &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

namespace {

struct SetStorage : public llvm::FoldingSetNode {
  SetStorage(SetVector<ElaboratorValue> &&set, Type type)
      : set(std::move(set)), type(type) {}

  // NOLINTNEXTLINE(readability-identifier-naming)
  static void Profile(llvm::FoldingSetNodeID &ID,
                      const SetVector<ElaboratorValue> &set, Type type) {
    for (auto el : set) {
      ID.AddPointer(el.getStorage().ptr);
      ID.AddInteger(static_cast<unsigned>(el.getKind()));
    }
    ID.AddPointer(type.getAsOpaquePointer());
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  void Profile(llvm::FoldingSetNodeID &ID) const { Profile(ID, set, type); }

  // Stores the elaborated values of the set.
  SetVector<ElaboratorValue> set;

  // Store the set type such that we can materialize this evaluated value
  // also in the case where the set is empty.
  Type type;
};

struct BagStorage : public llvm::FoldingSetNode {
  BagStorage(MapVector<ElaboratorValue, uint64_t> &&bag, Type type)
      : bag(std::move(bag)), type(type) {}

  // NOLINTNEXTLINE(readability-identifier-naming)
  static void Profile(llvm::FoldingSetNodeID &ID,
                      const MapVector<ElaboratorValue, uint64_t> &bag,
                      Type type) {
    for (auto el : bag) {
      ID.AddPointer(el.first.getStorage().ptr);
      ID.AddInteger(static_cast<unsigned>(el.first.getKind()));
      ID.AddInteger(el.second);
    }
    ID.AddPointer(type.getAsOpaquePointer());
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  void Profile(llvm::FoldingSetNodeID &ID) const { Profile(ID, bag, type); }

  // Stores the elaborated values of the bag.
  MapVector<ElaboratorValue, uint64_t> bag;

  // Store the bag type such that we can materialize this evaluated value
  // also in the case where the bag is empty.
  Type type;
};

struct SequenceStorage : public llvm::FoldingSetNode {
  SequenceStorage(StringRef name, StringAttr familyName,
                  SmallVector<ElaboratorValue> &&args)
      : name(name), familyName(familyName), args(std::move(args)) {}

  // NOLINTNEXTLINE(readability-identifier-naming)
  static void Profile(llvm::FoldingSetNodeID &ID, StringRef name,
                      StringAttr familyName, ArrayRef<ElaboratorValue> args) {
    ID.AddString(name);
    ID.AddPointer(familyName.getAsOpaquePointer());
    for (auto el : args) {
      ID.AddPointer(el.getStorage().ptr);
      ID.AddInteger(static_cast<unsigned>(el.getKind()));
    }
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  void Profile(llvm::FoldingSetNodeID &ID) const {
    Profile(ID, name, familyName, args);
  }

  StringRef name;
  StringAttr familyName;
  SmallVector<ElaboratorValue> args;
};

class Internalizer {
public:
  template <typename StorageTy, typename... Args>
  StorageTy *internalize(Args &&...args) {
    llvm::FoldingSetNodeID profile;
    StorageTy::Profile(profile, args...);
    void *insertPos = nullptr;
    if (auto *storage =
            getInternSet<StorageTy>().FindNodeOrInsertPos(profile, insertPos))
      return static_cast<StorageTy *>(storage);
    auto *storagePtr = new (allocator.Allocate<StorageTy>())
        StorageTy(std::forward<Args>(args)...);
    getInternSet<StorageTy>().InsertNode(storagePtr, insertPos);
    return storagePtr;
  }

  template <typename StorageTy>
  llvm::FoldingSet<StorageTy> &getInternSet() {
    assert(false && "no generic internalization set");
  }

  template <>
  llvm::FoldingSet<SetStorage> &getInternSet() {
    return internedSets;
  }

  template <>
  llvm::FoldingSet<BagStorage> &getInternSet() {
    return internedBags;
  }

  template <>
  llvm::FoldingSet<SequenceStorage> &getInternSet() {
    return internedSequences;
  }

  //  BagStorage *internalize(BagStorage &&storage) {
  //   llvm::FoldingSetNodeID profile;
  //   storage.Profile(profile);
  //   void *insertPos = nullptr;
  //   if (auto *bag = internedBags.FindNodeOrInsertPos(profile, insertPos))
  //     return bag;
  //   auto *storagePtr = new BagStorage(std::move(storage));
  //   internedBags.InsertNode(storagePtr, insertPos);
  //   return storagePtr;
  // }

private:
  llvm::BumpPtrAllocator allocator;
  // A map used to intern elaborator values. We do this such that we can
  // compare pointers when, e.g., computing set differences, uniquing the
  // elements in a set, etc. Otherwise, we'd need to do a deep value comparison
  // in those situations.
  // Use a pointer as the key with custom MapInfo because of object slicing when
  // inserting an object of a derived class of ElaboratorValue.
  // The custom MapInfo makes sure that we do a value comparison instead of
  // comparing the pointers.
  llvm::FoldingSet<SetStorage> internedSets;
  llvm::FoldingSet<BagStorage> internedBags;
  llvm::FoldingSet<SequenceStorage> internedSequences;
};

/// Holds any typed attribute. Wrapping around an MLIR `Attribute` allows us to
/// use this elaborator value class for any values that have a corresponding
/// MLIR attribute rather than one per kind of attribute. We only support typed
/// attributes because for materialization we need to provide the type to the
/// dialect's materializer.
struct AttributeValue : public ElaboratorValue {
  AttributeValue() = default;
  AttributeValue(StorageTy storage)
      : ElaboratorValue(ValueKind::Attribute, storage) {}
  AttributeValue(TypedAttr attr)
      : ElaboratorValue(ValueKind::Attribute, attr.getAsOpaquePointer()) {
    assert(attr && "null attributes not allowed");
  }

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue &val) {
    return val.getKind() == ValueKind::Attribute;
  }

  TypedAttr getAttr() const {
    return cast<TypedAttr>(Attribute::getFromOpaquePointer(getStorage().ptr));
  }
};

/// Holds an evaluated value of a `IndexType`'d value.
struct IndexValue : public ElaboratorValue {
  IndexValue() = default;
  IndexValue(StorageTy storage) : ElaboratorValue(ValueKind::Index, storage) {}
  IndexValue(size_t index) : ElaboratorValue(ValueKind::Index, index) {}

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue &val) {
    return val.getKind() == ValueKind::Index;
  }

  size_t getIndex() const { return getStorage().index; }
};

/// Holds an evaluated value of an `i1` type'd value.
struct BoolValue : public ElaboratorValue {
  BoolValue() = default;
  BoolValue(StorageTy storage) : ElaboratorValue(ValueKind::Bool, storage) {}
  BoolValue(bool value) : ElaboratorValue(ValueKind::Bool, value) {}

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue &val) {
    return val.getKind() == ValueKind::Bool;
  }

  bool getBool() const { return getStorage().boolean; }
};

/// Holds an evaluated value of a `SetType`'d value.
struct SetValue : public ElaboratorValue {
  SetValue() = default;
  SetValue(StorageTy storage) : ElaboratorValue(ValueKind::Set, storage) {}
  SetValue(Internalizer &internalizer, SetVector<ElaboratorValue> &&set,
           Type type)
      : ElaboratorValue(ValueKind::Set, internalizer.internalize<SetStorage>(
                                            std::move(set), type)) {}

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue &val) {
    return val.getKind() == ValueKind::Set;
  }

  const SetVector<ElaboratorValue> &getSet() const {
    return static_cast<const SetStorage *>(getStorage().ptr)->set;
  }

  Type getType() const {
    return static_cast<const SetStorage *>(getStorage().ptr)->type;
  }
};

/// Holds an evaluated value of a `BagType`'d value.
struct BagValue : public ElaboratorValue {
  BagValue() = default;
  BagValue(StorageTy storage) : ElaboratorValue(ValueKind::Bag, storage) {}
  BagValue(Internalizer &internalizer,
           MapVector<ElaboratorValue, uint64_t> &&bag, Type type)
      : ElaboratorValue(ValueKind::Bag, internalizer.internalize<BagStorage>(
                                            std::move(bag), type)) {}

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue &val) {
    return val.getKind() == ValueKind::Bag;
  }

  const MapVector<ElaboratorValue, uint64_t> &getBag() const {
    return static_cast<const BagStorage *>(getStorage().ptr)->bag;
  }

  Type getType() const {
    return static_cast<const BagStorage *>(getStorage().ptr)->type;
  }
};

/// Holds an evaluated value of a `SequenceType`'d value.
struct SequenceValue : public ElaboratorValue {
  SequenceValue() = default;
  SequenceValue(StorageTy storage)
      : ElaboratorValue(ValueKind::Sequence, storage) {}
  SequenceValue(Internalizer &internalizer, StringRef name,
                StringAttr familyName, SmallVector<ElaboratorValue> &&args)
      : ElaboratorValue(ValueKind::Sequence,
                        internalizer.internalize<SequenceStorage>(
                            name, familyName, std::move(args))) {}

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue &val) {
    return val.getKind() == ValueKind::Sequence;
  }

  StringRef getName() const {
    return static_cast<const SequenceStorage *>(getStorage().ptr)->name;
  }
  StringAttr getFamilyName() const {
    return static_cast<const SequenceStorage *>(getStorage().ptr)->familyName;
  }
  ArrayRef<ElaboratorValue> getArgs() const {
    return static_cast<const SequenceStorage *>(getStorage().ptr)->args;
  }
};
} // namespace

#ifndef NDEBUG
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ElaboratorValue &value) {
  TypeSwitch<ElaboratorValue>(value)
      .Case<AttributeValue>(
          [&](auto val) { os << "<attr " << val.getAttr() << ">"; })
      .Case<IndexValue>(
          [&](auto val) { os << "<index " << val.getIndex() << ">"; })
      .Case<BoolValue>(
          [&](auto val) { os << "<bool " << val.getBool() << ">"; })
      .Case<SetValue>([&](auto val) {
        os << "<set {";
        llvm::interleaveComma(val.getSet(), os);
        os << "} at " << val.getStorage().ptr << ">";
      })
      .Case<BagValue>([&](auto val) {
        os << "<bag {";
        llvm::interleaveComma(
            val.getBag(), os,
            [&](const std::pair<ElaboratorValue, uint64_t> &el) {
              os << el.first << " -> " << el.second;
            });
        os << "} at " << val.getStorage().ptr << ">";
      })
      .Case<SequenceValue>([&](auto val) {
        os << "<sequence @" << val.getName() << " derived from @"
           << val.getFamilyName().getValue() << "(";
        llvm::interleaveComma(val.getArgs(), os,
                              [&](const ElaboratorValue &val) { os << val; });
        os << ") at " << val.getStorage().ptr << ">";
      })
      .Default([](auto val) {
        assert(false && "all cases must be covered above");
        return Value();
      });
  return os;
}
#endif

//===----------------------------------------------------------------------===//
// Hash Map Helpers
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Main Elaborator Implementation
//===----------------------------------------------------------------------===//

namespace {

/// Construct an SSA value from a given elaborated value.
class Materializer {
public:
  Value materialize(ElaboratorValue val, Location loc,
                    std::queue<SequenceValue> &elabRequests,
                    function_ref<InFlightDiagnostic()> emitError) {
    assert(block && "must call reset before calling this function");

    auto iter = materializedValues.find(val);
    if (iter != materializedValues.end())
      return iter->second;

    LLVM_DEBUG(llvm::dbgs() << "Materializing " << val << "\n\n");

    OpBuilder builder(block, insertionPoint);
    return TypeSwitch<ElaboratorValue, Value>(val)
        .Case<AttributeValue, IndexValue, BoolValue, SetValue, BagValue,
              SequenceValue>([&](auto val) {
          return visit(val, builder, loc, elabRequests, emitError);
        })
        .Default([](auto val) {
          assert(false && "all cases must be covered above");
          return Value();
        });
  }

  Materializer &reset(Block *block) {
    materializedValues.clear();
    this->block = block;
    insertionPoint = block->begin();
    return *this;
  }

private:
  Value visit(const AttributeValue &val, OpBuilder &builder, Location loc,
              std::queue<SequenceValue> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    auto attr = val.getAttr();

    // For index attributes (and arithmetic operations on them) we use the
    // index dialect.
    if (auto intAttr = dyn_cast<IntegerAttr>(attr);
        intAttr && isa<IndexType>(attr.getType())) {
      Value res = builder.create<index::ConstantOp>(loc, intAttr);
      materializedValues[val] = res;
      return res;
    }

    // For any other attribute, we just call the materializer of the dialect
    // defining that attribute.
    auto *op = attr.getDialect().materializeConstant(builder, attr,
                                                     attr.getType(), loc);
    if (!op) {
      emitError() << "materializer of dialect '"
                  << attr.getDialect().getNamespace()
                  << "' unable to materialize value for attribute '" << attr
                  << "'";
      return Value();
    }

    Value res = op->getResult(0);
    materializedValues[val] = res;
    return res;
  }

  Value visit(const IndexValue &val, OpBuilder &builder, Location loc,
              std::queue<SequenceValue> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    Value res = builder.create<index::ConstantOp>(loc, val.getIndex());
    materializedValues[val] = res;
    return res;
  }

  Value visit(const BoolValue &val, OpBuilder &builder, Location loc,
              std::queue<SequenceValue> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    Value res = builder.create<index::BoolConstantOp>(loc, val.getBool());
    materializedValues[val] = res;
    return res;
  }

  Value visit(const SetValue &val, OpBuilder &builder, Location loc,
              std::queue<SequenceValue> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> elements;
    elements.reserve(val.getSet().size());
    for (auto el : val.getSet()) {
      auto materialized = materialize(el, loc, elabRequests, emitError);
      if (!materialized)
        return Value();

      elements.push_back(materialized);
    }

    auto res = builder.create<SetCreateOp>(loc, val.getType(), elements);
    materializedValues[val] = res;
    return res;
  }

  Value visit(const BagValue &val, OpBuilder &builder, Location loc,
              std::queue<SequenceValue> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> values, weights;
    values.reserve(val.getBag().size());
    weights.reserve(val.getBag().size());
    for (auto [val, weight] : val.getBag()) {
      auto materializedVal = materialize(val, loc, elabRequests, emitError);
      auto materializedWeight =
          materialize(IndexValue(weight), loc, elabRequests, emitError);
      if (!materializedVal || !materializedWeight)
        return Value();

      values.push_back(materializedVal);
      weights.push_back(materializedWeight);
    }

    auto res = builder.create<BagCreateOp>(loc, val.getType(), values, weights);
    materializedValues[val] = res;
    return res;
  }

  Value visit(const SequenceValue &val, OpBuilder &builder, Location loc,
              std::queue<SequenceValue> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    elabRequests.push(val);
    return builder.create<SequenceClosureOp>(loc, val.getName(), ValueRange());
  }

private:
  /// Cache values we have already materialized to reuse them later. We start
  /// with an insertion point at the start of the block and cache the (updated)
  /// insertion point such that future materializations can also reuse previous
  /// materializations without running into dominance issues (or requiring
  /// additional checks to avoid them).
  DenseMap<ElaboratorValue, Value> materializedValues;

  /// Cache the builders to continue insertions at their current insertion point
  /// for the reason stated above.
  Block *block;
  Block::iterator insertionPoint;
};

/// Used to signal to the elaboration driver whether the operation should be
/// removed.
enum class DeletionKind { Keep, Delete };

/// A collection of things to be passed to the visitor functions
struct VisitorInfo {
  /// The visitor function can set this to a block that it wants elaborated.
  /// Once it is fully elaborated, the visitor function is called again. This
  /// can occur repeatedly until this field is set to 'nullptr'.
  Block *toElaborate = nullptr;
};

/// Interprets the IR to perform and lower the represented randomizations.
class Elaborator
    : public RTGOpVisitor<Elaborator, FailureOr<DeletionKind>, VisitorInfo &> {
public:
  using RTGBase =
      RTGOpVisitor<Elaborator, FailureOr<DeletionKind>, VisitorInfo &>;
  using RTGBase::visitOp;
  using RTGBase::visitRegisterOp;

  Elaborator(SymbolTable &table, std::mt19937 &rng) : rng(rng), table(table) {}

  inline void store(Value val, const ElaboratorValue &eval) {
    state[val] = eval;
  }
  template <typename ValueTy>
  inline ValueTy get(Value val) {
    return dyn_cast<ValueTy>(state.at(val));
  }
  inline ElaboratorValue get(Value val) { return state.at(val); }

  /// Print a nice error message for operations we don't support yet.
  FailureOr<DeletionKind> visitUnhandledOp(Operation *op, VisitorInfo &info) {
    return op->emitOpError("elaboration not supported");
  }

  FailureOr<DeletionKind> visitExternalOp(Operation *op, VisitorInfo &info) {
    // TODO: we only have this to be able to write tests for this pass without
    // having to add support for more operations for now, so it should be
    // removed once it is not necessary anymore for writing tests
    if (op->use_empty())
      return DeletionKind::Keep;

    return visitUnhandledOp(op, info);
  }

  FailureOr<DeletionKind> visitOp(SequenceClosureOp op, VisitorInfo &info) {
    SmallVector<ElaboratorValue> args;
    for (auto arg : op.getArgs())
      args.push_back(get(arg));

    auto familyName = op.getSequenceAttr();
    auto name = names.newName(familyName.getValue());
    store(op.getResult(),
          SequenceValue(internalizer, name, familyName, std::move(args)));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(InvokeSequenceOp op, VisitorInfo &info) {
    return DeletionKind::Keep;
  }

  FailureOr<DeletionKind> visitOp(SetCreateOp op, VisitorInfo &info) {
    SetVector<ElaboratorValue> set;
    for (auto val : op.getElements())
      set.insert(get(val));

    store(op.getSet(),
          SetValue(internalizer, std::move(set), op.getSet().getType()));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SetSelectRandomOp op, VisitorInfo &info) {
    auto set = cast<SetValue>(get(op.getSet()));

    size_t selected;
    if (auto intAttr =
            op->getAttrOfType<IntegerAttr>("rtg.elaboration_custom_seed")) {
      std::mt19937 customRng(intAttr.getInt());
      selected = getUniformlyInRange(customRng, 0, set.getSet().size() - 1);
    } else {
      selected = getUniformlyInRange(rng, 0, set.getSet().size() - 1);
    }

    store(op.getResult(), set.getSet()[selected]);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SetDifferenceOp op, VisitorInfo &info) {
    auto original = get<SetValue>(op.getOriginal()).getSet();
    auto diff = get<SetValue>(op.getDiff()).getSet();

    SetVector<ElaboratorValue> result(original);
    result.set_subtract(diff);

    store(op.getResult(),
          SetValue(internalizer, std::move(result), op.getResult().getType()));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SetUnionOp op, VisitorInfo &info) {
    SetVector<ElaboratorValue> result;
    for (auto set : op.getSets())
      result.set_union(get<SetValue>(set).getSet());

    store(op.getResult(),
          SetValue(internalizer, std::move(result), op.getType()));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SetSizeOp op, VisitorInfo &info) {
    auto size = get<SetValue>(op.getSet()).getSet().size();
    store(op.getResult(), IndexValue(size));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagCreateOp op, VisitorInfo &info) {
    MapVector<ElaboratorValue, uint64_t> bag;
    for (auto [val, multiple] :
         llvm::zip(op.getElements(), op.getMultiples())) {
      auto interpValue = get(val);
      // If the multiple is not stored as an AttributeValue, the elaboration
      // must have already failed earlier (since we don't have
      // unevaluated/opaque values).
      auto interpMultiple = get<IndexValue>(multiple);
      bag[interpValue] += interpMultiple.getIndex();
    }

    store(op.getBag(), BagValue(internalizer, std::move(bag), op.getType()));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagSelectRandomOp op, VisitorInfo &info) {
    auto bag = get<BagValue>(op.getBag());

    SmallVector<std::pair<ElaboratorValue, uint32_t>> prefixSum;
    prefixSum.reserve(bag.getBag().size());
    uint32_t accumulator = 0;
    for (auto [val, weight] : bag.getBag()) {
      accumulator += weight;
      prefixSum.push_back({val, accumulator});
    }

    auto customRng = rng;
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

    store(op.getResult(), iter->first);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagDifferenceOp op, VisitorInfo &info) {
    auto original = get<BagValue>(op.getOriginal());
    auto diff = get<BagValue>(op.getDiff());

    MapVector<ElaboratorValue, uint64_t> result;
    for (const auto &el : original.getBag()) {
      if (!diff.getBag().contains(el.first)) {
        result.insert(el);
        continue;
      }

      if (op.getInf())
        continue;

      auto toDiff = diff.getBag().lookup(el.first);
      if (el.second <= toDiff)
        continue;

      result.insert({el.first, el.second - toDiff});
    }

    store(op.getResult(),
          BagValue(internalizer, std::move(result), op.getType()));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagUnionOp op, VisitorInfo &info) {
    MapVector<ElaboratorValue, uint64_t> result;
    for (auto bag : op.getBags()) {
      auto val = get<BagValue>(bag);
      for (auto [el, multiple] : val.getBag())
        result[el] += multiple;
    }

    store(op.getResult(),
          BagValue(internalizer, std::move(result), op.getType()));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagUniqueSizeOp op, VisitorInfo &info) {
    auto size = get<BagValue>(op.getBag()).getBag().size();
    store(op.getResult(), IndexValue(size));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(scf::IfOp op, VisitorInfo &info) {
    bool cond = get<BoolValue>(op.getCondition()).getBool();
    if (!info.toElaborate) {
      info.toElaborate = cond ? op.thenBlock() : op.elseBlock();
      return info.toElaborate ? DeletionKind::Keep : DeletionKind::Delete;
    }

    for (auto [res, out] : llvm::zip(
             op.getResults(), info.toElaborate->getTerminator()->getOperands()))
      store(res, get(out));

    info.toElaborate = nullptr;
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(scf::ForOp op, VisitorInfo &info) {
    auto lowerBound = get<IndexValue>(op.getLowerBound());
    auto step = get<IndexValue>(op.getStep());
    auto upperBound = get<IndexValue>(op.getUpperBound());

    if (!lowerBound || !step || !upperBound)
      return op->emitOpError("can only elaborate index type iterator");

    // The loop never executes the body
    if (lowerBound.getIndex() >= upperBound.getIndex()) {
      for (auto [res, iterArg] : llvm::zip(op->getResults(), op.getInitArgs()))
        store(res, get(iterArg));
      return DeletionKind::Delete;
    }

    // First iteration
    if (!info.toElaborate) {
      store(op.getInductionVar(), lowerBound);
      for (auto [iterArg, initArg] :
           llvm::zip(op.getRegionIterArgs(), op.getInitArgs()))
        store(iterArg, get(initArg));
      info.toElaborate = op.getBody();
      return DeletionKind::Keep;
    }

    // Last time was the last execution
    size_t idx = get<IndexValue>(op.getInductionVar()).getIndex();
    if (idx + step.getIndex() >= upperBound.getIndex()) {
      for (auto [res, iterArg] : llvm::zip(
               op->getResults(), op.getBody()->getTerminator()->getOperands()))
        store(res, get(iterArg));

      info.toElaborate = nullptr;
      return DeletionKind::Delete;
    }

    // We are doing a regular iteration
    store(op.getInductionVar(), IndexValue(idx + step.getIndex()));
    for (auto [iterArg, prevIterArg] :
         llvm::zip(op.getRegionIterArgs(),
                   info.toElaborate->getTerminator()->getOperands()))
      store(iterArg, get(prevIterArg));

    info.toElaborate = op.getBody();
    return DeletionKind::Keep;
  }

  FailureOr<DeletionKind> visitOp(scf::YieldOp op, VisitorInfo &info) {
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(index::AddOp op, VisitorInfo &info) {
    size_t lhs = get<IndexValue>(op.getLhs()).getIndex();
    size_t rhs = get<IndexValue>(op.getRhs()).getIndex();
    store(op.getResult(), IndexValue(lhs + rhs));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(index::CmpOp op, VisitorInfo &info) {
    size_t lhs = get<IndexValue>(op.getLhs()).getIndex();
    size_t rhs = get<IndexValue>(op.getRhs()).getIndex();
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
    store(op.getResult(), BoolValue(result));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> dispatchOpVisitor(Operation *op, VisitorInfo &info) {
    if (op->hasTrait<OpTrait::ConstantLike>()) {
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
        store(op->getResult(0), IndexValue(intAttr.getInt()));
      else if (intAttr && intAttr.getType().isSignlessInteger(1))
        store(op->getResult(0), BoolValue(intAttr.getInt()));
      else
        store(op->getResult(0), AttributeValue(attr));

      return DeletionKind::Delete;
    }

    return TypeSwitch<Operation *, FailureOr<DeletionKind>>(op)
        .Case<
            // Index ops
            index::AddOp, index::CmpOp,
            // SCF ops
            scf::IfOp, scf::ForOp, scf::YieldOp>(
            [&](auto op) { return visitOp(op, info); })
        .Default([&](Operation *op) {
          return RTGBase::dispatchOpVisitor(op, info);
        });
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  LogicalResult elaborateBlock(Block *block, OpBuilder &builder,
                               IRMapping &mapping) {
    for (auto &op : *block) {
      VisitorInfo info;
      FailureOr<DeletionKind> result;
      do {
        result = dispatchOpVisitor(&op, info);
        if (failed(result))
          return failure();

        if (op.getNumRegions() != 0 && info.toElaborate == nullptr &&
            *result == DeletionKind::Keep)
          return op.emitOpError(
              "ops with nested regions must be elaborated away");

        // TODO: ideally use an itnerative approach, for now we just assume that
        // we don't have very deep nestings
        if (info.toElaborate) {
          IRMapping nestedMapping;
          if (failed(elaborateBlock(info.toElaborate, builder, nestedMapping)))
            return failure();
        }
      } while (info.toElaborate);

      if (*result == DeletionKind::Keep) {
        for (auto &operand : op.getOpOperands()) {
          if (mapping.contains(operand.get()))
            continue;

          auto emitError = [&]() {
            auto diag = op.emitError();
            diag.attachNote(op.getLoc())
                << "while materializing value for operand#"
                << operand.getOperandNumber();
            return diag;
          };
          Value val = materializer.materialize(get(operand.get()), op.getLoc(),
                                               worklist, emitError);
          if (!val)
            return failure();

          mapping.map(operand.get(), val);
        }

        builder.clone(op, mapping);
      }

      LLVM_DEBUG({
        llvm::dbgs() << "Elaborating " << op << " to\n[";

        llvm::interleaveComma(op.getResults(), llvm::dbgs(), [&](auto res) {
          if (state.contains(res))
            llvm::dbgs() << get(res);
          else
            llvm::dbgs() << "unknown";
        });

        llvm::dbgs() << "]\n\n";
      });
    }

    return success();
  }

  LogicalResult elaborate(SequenceOp family, SequenceOp dest,
                          ArrayRef<ElaboratorValue> args) {
    LLVM_DEBUG(llvm::dbgs() << "\n=== Elaborating " << family.getOperationName()
                            << " @" << family.getSymName() << " into @"
                            << dest.getSymName() << "\n\n");

    // Reduce max memory consumption and make sure the values cannot be accessed
    // anymore because we deleted the ops above. Clearing should lead to better
    // performance than having them as a local here and pass via function
    // argument.
    state.clear();
    materializer.reset(dest.getBody());
    IRMapping mapping;

    for (auto [arg, elabArg] :
         llvm::zip(family.getBody()->getArguments(), args))
      state[arg] = elabArg;

    OpBuilder builder = OpBuilder::atBlockEnd(dest.getBody());
    return elaborateBlock(family.getBody(), builder, mapping);
  }

  template <typename OpTy>
  LogicalResult elaborateInPlace(OpTy op) {
    LLVM_DEBUG(llvm::dbgs()
               << "\n=== Elaborating (in place) " << op.getOperationName()
               << " @" << op.getSymName() << "\n\n");

    // Reduce max memory consumption and make sure the values cannot be accessed
    // anymore because we deleted the ops above. Clearing should lead to better
    // performance than having them as a local here and pass via function
    // argument.
    state.clear();
    materializer.reset(op.getBody());

    SmallVector<Operation *> toDelete;
    for (auto &op : *op.getBody()) {
      VisitorInfo info;
      FailureOr<DeletionKind> result;
      OpBuilder builder(&op);
      do {
        result = dispatchOpVisitor(&op, info);
        if (failed(result))
          return failure();

        if (op.getNumRegions() != 0 && info.toElaborate == nullptr &&
            *result == DeletionKind::Keep)
          return op.emitOpError(
              "ops with nested regions must be elaborated away");

        // TODO: maybe worth asking the visitor if this is the last time we
        // evaluate that block and could thus inline it and elaborate in-place.
        if (info.toElaborate) {
          IRMapping mapping;
          if (failed(elaborateBlock(info.toElaborate, builder, mapping)))
            return failure();
        }
      } while (info.toElaborate);

      if (*result == DeletionKind::Keep) {
        for (auto &operand : op.getOpOperands()) {
          auto emitError = [&]() {
            auto diag = op.emitError();
            diag.attachNote(op.getLoc())
                << "while materializing value for operand#"
                << operand.getOperandNumber();
            return diag;
          };
          Value val = materializer.materialize(get(operand.get()), op.getLoc(),
                                               worklist, emitError);
          if (!val)
            return failure();
          operand.set(val);
        }
      } else { // DeletionKind::Delete
        toDelete.push_back(&op);
      }

      LLVM_DEBUG({
        llvm::dbgs() << "Elaborating " << op << " to\n[";

        llvm::interleaveComma(op.getResults(), llvm::dbgs(), [&](auto res) {
          if (state.contains(res))
            llvm::dbgs() << get(res);
          else
            llvm::dbgs() << "unknown";
        });

        llvm::dbgs() << "]\n\n";
      });
    }

    for (auto *op : llvm::reverse(toDelete))
      op->erase();

    return success();
  }

  LogicalResult inlineSequences(TestOp testOp) {
    OpBuilder builder(testOp);
    for (auto iter = testOp.getBody()->begin();
         iter != testOp.getBody()->end();) {
      auto invokeOp = dyn_cast<InvokeSequenceOp>(&*iter);
      if (!invokeOp) {
        ++iter;
        continue;
      }

      auto seqClosureOp =
          invokeOp.getSequence().getDefiningOp<SequenceClosureOp>();
      if (!seqClosureOp)
        return invokeOp->emitError(
            "sequence operand not directly defined by sequence_closure op");

      auto seqOp = table.lookup<SequenceOp>(seqClosureOp.getSequenceAttr());

      builder.setInsertionPointAfter(invokeOp);
      IRMapping mapping;
      for (auto &op : *seqOp.getBody())
        builder.clone(op, mapping);

      (iter++)->erase();

      if (seqClosureOp->use_empty())
        seqClosureOp->erase();
    }

    return success();
  }

  LogicalResult elaborateModule(ModuleOp moduleOp) {
    // Update the name cache
    names.clear();
    names.add(moduleOp);

    // Initialize the worklist with the test ops since they cannot be placed by
    // other ops.
    for (auto testOp : moduleOp.getOps<TestOp>())
      if (failed(elaborateInPlace(testOp)))
        return failure();

    // Do top-down BFS traversal such that elaborating a sequence further down
    // does not fix the outcome for multiple placements.
    while (!worklist.empty()) {
      auto curr = worklist.front();
      worklist.pop();

      if (table.lookup<SequenceOp>(curr.getName()))
        continue;

      auto familyOp = table.lookup<SequenceOp>(curr.getFamilyName());
      // TODO: use 'elaborateInPlace' and don't clone if this is the only
      // remaining reference to this sequence
      OpBuilder builder(familyOp);
      auto seqOp = builder.cloneWithoutRegions(familyOp);
      seqOp.getBodyRegion().emplaceBlock();
      seqOp.setSymName(curr.getName());
      table.insert(seqOp);
      assert(seqOp.getSymName() == curr.getName() &&
             "should not have been renamed");

      if (failed(elaborate(familyOp, seqOp, curr.getArgs())))
        return failure();
    }

    // Inline all sequences and remove the operations that place the sequences.
    for (auto testOp : moduleOp.getOps<TestOp>())
      if (failed(inlineSequences(testOp)))
        return failure();

    // Remove all sequences since they are not accessible from the outside and
    // are not needed anymore since we fully inlined them.
    for (auto seqOp : llvm::make_early_inc_range(moduleOp.getOps<SequenceOp>()))
      seqOp->erase();

    return success();
  }

private:
  std::mt19937 rng;
  SymbolTable &table;
  Namespace names;

  DenseMap<ElaboratorValue, SequenceOp> materializedSequences;

  /// The worklist used to keep track of the test and sequence operations to
  /// make sure they are processed top-down (BFS traversal).
  std::queue<SequenceValue> worklist;

  // A map from SSA values to a pointer of an interned elaborator value.
  DenseMap<Value, ElaboratorValue> state;

  // Allows us to materialize ElaboratorValues to the IR operations necessary to
  // obtain an SSA value representing that elaborated value.
  Materializer materializer;

  Internalizer internalizer;
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
};
} // namespace

void ElaborationPass::runOnOperation() {
  auto moduleOp = getOperation();
  SymbolTable table(moduleOp);

  cloneTargetsIntoTests(table);

  std::mt19937 rng(seed);
  Elaborator elaborator(table, rng);
  if (failed(elaborator.elaborateModule(moduleOp)))
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
