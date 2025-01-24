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
struct ElaboratorValue {
public:
  enum class ValueKind { Attribute, Set, Bag, Sequence, Index, Bool };

  ElaboratorValue(ValueKind kind) : kind(kind) {}
  virtual ~ElaboratorValue() {}

  virtual llvm::hash_code getHashValue() const = 0;
  virtual bool isEqual(const ElaboratorValue &other) const = 0;

#ifndef NDEBUG
  virtual void print(llvm::raw_ostream &os) const = 0;
#endif

  ValueKind getKind() const { return kind; }

private:
  const ValueKind kind;
};

/// Holds any typed attribute. Wrapping around an MLIR `Attribute` allows us to
/// use this elaborator value class for any values that have a corresponding
/// MLIR attribute rather than one per kind of attribute. We only support typed
/// attributes because for materialization we need to provide the type to the
/// dialect's materializer.
class AttributeValue : public ElaboratorValue {
public:
  AttributeValue(TypedAttr attr)
      : ElaboratorValue(ValueKind::Attribute), attr(attr) {
    assert(attr && "null attributes not allowed");
  }

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue *val) {
    return val->getKind() == ValueKind::Attribute;
  }

  llvm::hash_code getHashValue() const override {
    return llvm::hash_combine(attr);
  }

  bool isEqual(const ElaboratorValue &other) const override {
    auto *attrValue = dyn_cast<AttributeValue>(&other);
    if (!attrValue)
      return false;

    return attr == attrValue->attr;
  }

#ifndef NDEBUG
  void print(llvm::raw_ostream &os) const override {
    os << "<attr " << attr << " at " << this << ">";
  }
#endif

  TypedAttr getAttr() const { return attr; }

private:
  const TypedAttr attr;
};

/// Holds an evaluated value of a `IndexType`'d value.
class IndexValue : public ElaboratorValue {
public:
  IndexValue(size_t index) : ElaboratorValue(ValueKind::Index), index(index) {}

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue *val) {
    return val->getKind() == ValueKind::Index;
  }

  llvm::hash_code getHashValue() const override {
    return llvm::hash_value(index);
  }

  bool isEqual(const ElaboratorValue &other) const override {
    auto *indexValue = dyn_cast<IndexValue>(&other);
    if (!indexValue)
      return false;

    return index == indexValue->index;
  }

#ifndef NDEBUG
  void print(llvm::raw_ostream &os) const override {
    os << "<index " << index << " at " << this << ">";
  }
#endif

  size_t getIndex() const { return index; }

private:
  const size_t index;
};

/// Holds an evaluated value of an `i1` type'd value.
class BoolValue : public ElaboratorValue {
public:
  BoolValue(bool value) : ElaboratorValue(ValueKind::Bool), value(value) {}

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue *val) {
    return val->getKind() == ValueKind::Bool;
  }

  llvm::hash_code getHashValue() const override {
    return llvm::hash_value(value);
  }

  bool isEqual(const ElaboratorValue &other) const override {
    auto *val = dyn_cast<BoolValue>(&other);
    if (!val)
      return false;

    return value == val->value;
  }

#ifndef NDEBUG
  void print(llvm::raw_ostream &os) const override {
    os << "<bool " << (value ? "true" : "false") << " at " << this << ">";
  }
#endif

  bool getBool() const { return value; }

private:
  const bool value;
};

/// Holds an evaluated value of a `SetType`'d value.
class SetValue : public ElaboratorValue {
public:
  SetValue(SetVector<ElaboratorValue *> &&set, Type type)
      : ElaboratorValue(ValueKind::Set), set(std::move(set)), type(type),
        cachedHash(llvm::hash_combine(
            llvm::hash_combine_range(set.begin(), set.end()), type)) {}

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue *val) {
    return val->getKind() == ValueKind::Set;
  }

  llvm::hash_code getHashValue() const override { return cachedHash; }

  bool isEqual(const ElaboratorValue &other) const override {
    auto *otherSet = dyn_cast<SetValue>(&other);
    if (!otherSet)
      return false;

    if (cachedHash != otherSet->cachedHash)
      return false;

    // Make sure empty sets of different types are not considered equal
    return set == otherSet->set && type == otherSet->type;
  }

#ifndef NDEBUG
  void print(llvm::raw_ostream &os) const override {
    os << "<set {";
    llvm::interleaveComma(set, os, [&](ElaboratorValue *el) { el->print(os); });
    os << "} at " << this << ">";
  }
#endif

  const SetVector<ElaboratorValue *> &getSet() const { return set; }

  Type getType() const { return type; }

private:
  // We currently use a sorted vector to represent sets. Note that it is sorted
  // by the pointer value and thus non-deterministic.
  // We probably want to do some profiling in the future to see if a DenseSet or
  // other representation is better suited.
  const SetVector<ElaboratorValue *> set;

  // Store the set type such that we can materialize this evaluated value
  // also in the case where the set is empty.
  const Type type;

  // Compute the hash only once at constructor time.
  const llvm::hash_code cachedHash;
};

/// Holds an evaluated value of a `BagType`'d value.
class BagValue : public ElaboratorValue {
public:
  BagValue(MapVector<ElaboratorValue *, uint64_t> &&bag, Type type)
      : ElaboratorValue(ValueKind::Bag), bag(std::move(bag)), type(type),
        cachedHash(llvm::hash_combine(
            llvm::hash_combine_range(bag.begin(), bag.end()), type)) {}

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue *val) {
    return val->getKind() == ValueKind::Bag;
  }

  llvm::hash_code getHashValue() const override { return cachedHash; }

  bool isEqual(const ElaboratorValue &other) const override {
    auto *otherBag = dyn_cast<BagValue>(&other);
    if (!otherBag)
      return false;

    if (cachedHash != otherBag->cachedHash)
      return false;

    return llvm::equal(bag, otherBag->bag) && type == otherBag->type;
  }

#ifndef NDEBUG
  void print(llvm::raw_ostream &os) const override {
    os << "<bag {";
    llvm::interleaveComma(bag, os,
                          [&](std::pair<ElaboratorValue *, uint64_t> el) {
                            el.first->print(os);
                            os << " -> " << el.second;
                          });
    os << "} at " << this << ">";
  }
#endif

  const MapVector<ElaboratorValue *, uint64_t> &getBag() const { return bag; }

  Type getType() const { return type; }

private:
  // Stores the elaborated values of the bag.
  const MapVector<ElaboratorValue *, uint64_t> bag;

  // Store the type of the bag such that we can materialize this evaluated value
  // also in the case where the bag is empty.
  const Type type;

  // Compute the hash only once at constructor time.
  const llvm::hash_code cachedHash;
};

/// Holds an evaluated value of a `SequenceType`'d value.
class SequenceValue : public ElaboratorValue {
public:
  SequenceValue(StringRef name, StringAttr familyName,
                SmallVector<ElaboratorValue *> &&args)
      : ElaboratorValue(ValueKind::Sequence), name(name),
        familyName(familyName), args(std::move(args)),
        cachedHash(llvm::hash_combine(
            llvm::hash_combine_range(this->args.begin(), this->args.end()),
            name, familyName)) {}

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue *val) {
    return val->getKind() == ValueKind::Sequence;
  }

  llvm::hash_code getHashValue() const override { return cachedHash; }

  bool isEqual(const ElaboratorValue &other) const override {
    auto *otherSeq = dyn_cast<SequenceValue>(&other);
    if (!otherSeq)
      return false;

    if (cachedHash != otherSeq->cachedHash)
      return false;

    return name == otherSeq->name && familyName == otherSeq->familyName &&
           args == otherSeq->args;
  }

#ifndef NDEBUG
  void print(llvm::raw_ostream &os) const override {
    os << "<sequence @" << name << " derived from @" << familyName.getValue()
       << "(";
    llvm::interleaveComma(args, os,
                          [&](ElaboratorValue *val) { val->print(os); });
    os << ") at " << this << ">";
  }
#endif

  StringRef getName() const { return name; }
  StringAttr getFamilyName() const { return familyName; }
  ArrayRef<ElaboratorValue *> getArgs() const { return args; }

private:
  const StringRef name;
  const StringAttr familyName;
  const SmallVector<ElaboratorValue *> args;

  // Compute the hash only once at constructor time.
  const llvm::hash_code cachedHash;
};
} // namespace

#ifndef NDEBUG
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const ElaboratorValue &value) {
  value.print(os);
  return os;
}
#endif

//===----------------------------------------------------------------------===//
// Hash Map Helpers
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(readability-identifier-naming)
static llvm::hash_code hash_value(const ElaboratorValue &val) {
  return val.getHashValue();
}

namespace {
struct InternMapInfo : public DenseMapInfo<ElaboratorValue *> {
  static unsigned getHashValue(const ElaboratorValue *value) {
    assert(value != getTombstoneKey() && value != getEmptyKey());
    return hash_value(*value);
  }

  static bool isEqual(const ElaboratorValue *lhs, const ElaboratorValue *rhs) {
    if (lhs == rhs)
      return true;

    auto *tk = getTombstoneKey();
    auto *ek = getEmptyKey();
    if (lhs == tk || rhs == tk || lhs == ek || rhs == ek)
      return false;

    return lhs->isEqual(*rhs);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Main Elaborator Implementation
//===----------------------------------------------------------------------===//

namespace {

/// Construct an SSA value from a given elaborated value.
class Materializer {
public:
  Materializer(OpBuilder builder) : builder(builder) {}

  /// Materialize IR representing the provided `ElaboratorValue` and return the
  /// `Value` or a null value on failure.
  Value materialize(ElaboratorValue *val, Location loc,
                    std::queue<SequenceValue *> &elabRequests,
                    function_ref<InFlightDiagnostic()> emitError) {
    auto iter = materializedValues.find(val);
    if (iter != materializedValues.end())
      return iter->second;

    LLVM_DEBUG(llvm::dbgs() << "Materializing " << *val << "\n\n");

    return TypeSwitch<ElaboratorValue *, Value>(val)
        .Case<AttributeValue, IndexValue, BoolValue, SetValue, BagValue,
              SequenceValue>(
            [&](auto val) { return visit(val, loc, elabRequests, emitError); })
        .Default([](auto val) {
          assert(false && "all cases must be covered above");
          return Value();
        });
  }

  /// If `op` is not in the same region as the materializer insertion point, a
  /// clone is created at the materializer's insertion point by also
  /// materializing the `ElaboratorValue`s for each operand just before it.
  /// Otherwise, all operations after the materializer's insertion point are
  /// deleted until `op` is reached. An error is returned if the operation is
  /// before the insertion point.
  LogicalResult materialize(Operation *op,
                            DenseMap<Value, ElaboratorValue *> &state,
                            std::queue<SequenceValue *> &elabRequests) {
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
      auto ip = builder.getInsertionPoint();
      while (ip != builder.getBlock()->end() && &*ip != op) {
        LLVM_DEBUG(llvm::dbgs() << "Marking to be deleted: " << *ip << "\n\n");
        toDelete.push_back(&*ip);

        builder.setInsertionPointAfter(&*ip);
        ip = builder.getInsertionPoint();
      }

      if (ip == builder.getBlock()->end())
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
    for (auto *op : llvm::reverse(toDelete))
      op->erase();
  }

private:
  Value visit(AttributeValue *val, Location loc,
              std::queue<SequenceValue *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    auto attr = val->getAttr();

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

  Value visit(IndexValue *val, Location loc,
              std::queue<SequenceValue *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    Value res = builder.create<index::ConstantOp>(loc, val->getIndex());
    materializedValues[val] = res;
    return res;
  }

  Value visit(BoolValue *val, Location loc,
              std::queue<SequenceValue *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    Value res = builder.create<index::BoolConstantOp>(loc, val->getBool());
    materializedValues[val] = res;
    return res;
  }

  Value visit(SetValue *val, Location loc,
              std::queue<SequenceValue *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> elements;
    elements.reserve(val->getSet().size());
    for (auto *el : val->getSet()) {
      auto materialized = materialize(el, loc, elabRequests, emitError);
      if (!materialized)
        return Value();

      elements.push_back(materialized);
    }

    auto res = builder.create<SetCreateOp>(loc, val->getType(), elements);
    materializedValues[val] = res;
    return res;
  }

  Value visit(BagValue *val, Location loc,
              std::queue<SequenceValue *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> values, weights;
    values.reserve(val->getBag().size());
    weights.reserve(val->getBag().size());
    for (auto [val, weight] : val->getBag()) {
      auto materializedVal = materialize(val, loc, elabRequests, emitError);
      if (!materializedVal)
        return Value();

      auto iter = integerValues.find(weight);
      Value materializedWeight;
      if (iter != integerValues.end()) {
        materializedWeight = iter->second;
      } else {
        materializedWeight = builder.create<index::ConstantOp>(
            loc, builder.getIndexAttr(weight));
        integerValues[weight] = materializedWeight;
      }

      values.push_back(materializedVal);
      weights.push_back(materializedWeight);
    }

    auto res =
        builder.create<BagCreateOp>(loc, val->getType(), values, weights);
    materializedValues[val] = res;
    return res;
  }

  Value visit(SequenceValue *val, Location loc,
              std::queue<SequenceValue *> &elabRequests,
              function_ref<InFlightDiagnostic()> emitError) {
    elabRequests.push(val);
    return builder.create<SequenceClosureOp>(loc, val->getName(), ValueRange());
  }

private:
  /// Cache values we have already materialized to reuse them later. We start
  /// with an insertion point at the start of the block and cache the (updated)
  /// insertion point such that future materializations can also reuse previous
  /// materializations without running into dominance issues (or requiring
  /// additional checks to avoid them).
  DenseMap<ElaboratorValue *, Value> materializedValues;
  DenseMap<uint64_t, Value> integerValues;

  /// Cache the builder to continue insertions at their current insertion point
  /// for the reason stated above.
  OpBuilder builder;

  SmallVector<Operation *> toDelete;
};

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

  // A map used to intern elaborator values. We do this such that we can
  // compare pointers when, e.g., computing set differences, uniquing the
  // elements in a set, etc. Otherwise, we'd need to do a deep value comparison
  // in those situations.
  // Use a pointer as the key with custom MapInfo because of object slicing when
  // inserting an object of a derived class of ElaboratorValue.
  // The custom MapInfo makes sure that we do a value comparison instead of
  // comparing the pointers.
  DenseMap<ElaboratorValue *, std::unique_ptr<ElaboratorValue>, InternMapInfo>
      interned;

  /// The worklist used to keep track of the test and sequence operations to
  /// make sure they are processed top-down (BFS traversal).
  std::queue<SequenceValue *> worklist;
};

/// Interprets the IR to perform and lower the represented randomizations.
class Elaborator : public RTGOpVisitor<Elaborator, FailureOr<DeletionKind>> {
public:
  using RTGBase = RTGOpVisitor<Elaborator, FailureOr<DeletionKind>>;
  using RTGBase::visitOp;

  Elaborator(ElaboratorSharedState &sharedState, Materializer &materializer)
      : sharedState(sharedState), materializer(materializer) {}

  /// Helper to perform internalization and keep track of interpreted value for
  /// the given SSA value.
  template <typename ValueTy, typename... Args>
  void internalizeResult(Value val, Args &&...args) {
    // TODO: this isn't the most efficient way to internalize
    auto ptr = std::make_unique<ValueTy>(std::forward<Args>(args)...);
    auto *e = ptr.get();
    auto [iter, _] = sharedState.interned.insert({e, std::move(ptr)});
    state[val] = iter->second.get();
  }

  /// Print a nice error message for operations we don't support yet.
  FailureOr<DeletionKind> visitUnhandledOp(Operation *op) {
    return op->emitOpError("elaboration not supported");
  }

  FailureOr<DeletionKind> visitExternalOp(Operation *op) {
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
        internalizeResult<IndexValue>(op->getResult(0), intAttr.getInt());
      else if (intAttr && intAttr.getType().isSignlessInteger(1))
        internalizeResult<BoolValue>(op->getResult(0), intAttr.getInt());
      else
        internalizeResult<AttributeValue>(op->getResult(0), attr);

      return DeletionKind::Delete;
    }

    // TODO: we only have this to be able to write tests for this pass without
    // having to add support for more operations for now, so it should be
    // removed once it is not necessary anymore for writing tests
    if (op->use_empty())
      return DeletionKind::Keep;

    return visitUnhandledOp(op);
  }

  FailureOr<DeletionKind> visitOp(SequenceClosureOp op) {
    SmallVector<ElaboratorValue *> args;
    for (auto arg : op.getArgs())
      args.push_back(state.at(arg));

    auto familyName = op.getSequenceAttr();
    auto name = sharedState.names.newName(familyName.getValue());
    internalizeResult<SequenceValue>(op.getResult(), name, familyName,
                                     std::move(args));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(InvokeSequenceOp op) {
    return DeletionKind::Keep;
  }

  FailureOr<DeletionKind> visitOp(SetCreateOp op) {
    SetVector<ElaboratorValue *> set;
    for (auto val : op.getElements())
      set.insert(state.at(val));

    internalizeResult<SetValue>(op.getSet(), std::move(set),
                                op.getSet().getType());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SetSelectRandomOp op) {
    auto *set = cast<SetValue>(state.at(op.getSet()));

    size_t selected;
    if (auto intAttr =
            op->getAttrOfType<IntegerAttr>("rtg.elaboration_custom_seed")) {
      std::mt19937 customRng(intAttr.getInt());
      selected = getUniformlyInRange(customRng, 0, set->getSet().size() - 1);
    } else {
      selected =
          getUniformlyInRange(sharedState.rng, 0, set->getSet().size() - 1);
    }

    state[op.getResult()] = set->getSet()[selected];
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SetDifferenceOp op) {
    auto original = cast<SetValue>(state.at(op.getOriginal()))->getSet();
    auto diff = cast<SetValue>(state.at(op.getDiff()))->getSet();

    SetVector<ElaboratorValue *> result(original);
    result.set_subtract(diff);

    internalizeResult<SetValue>(op.getResult(), std::move(result),
                                op.getResult().getType());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SetUnionOp op) {
    SetVector<ElaboratorValue *> result;
    for (auto set : op.getSets())
      result.set_union(cast<SetValue>(state.at(set))->getSet());

    internalizeResult<SetValue>(op.getResult(), std::move(result),
                                op.getType());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(SetSizeOp op) {
    auto size = cast<SetValue>(state.at(op.getSet()))->getSet().size();
    auto sizeAttr = IntegerAttr::get(IndexType::get(op->getContext()), size);
    internalizeResult<AttributeValue>(op.getResult(), sizeAttr);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagCreateOp op) {
    MapVector<ElaboratorValue *, uint64_t> bag;
    for (auto [val, multiple] :
         llvm::zip(op.getElements(), op.getMultiples())) {
      auto *interpValue = state.at(val);
      // If the multiple is not stored as an AttributeValue, the elaboration
      // must have already failed earlier (since we don't have
      // unevaluated/opaque values).
      auto *interpMultiple = cast<IndexValue>(state.at(multiple));
      bag[interpValue] += interpMultiple->getIndex();
    }

    internalizeResult<BagValue>(op.getBag(), std::move(bag), op.getType());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagSelectRandomOp op) {
    auto *bag = cast<BagValue>(state.at(op.getBag()));

    SmallVector<std::pair<ElaboratorValue *, uint32_t>> prefixSum;
    prefixSum.reserve(bag->getBag().size());
    uint32_t accumulator = 0;
    for (auto [val, weight] : bag->getBag()) {
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
        [](uint32_t a, const std::pair<ElaboratorValue *, uint32_t> &b) {
          return a < b.second;
        });
    state[op.getResult()] = iter->first;
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagDifferenceOp op) {
    auto *original = cast<BagValue>(state.at(op.getOriginal()));
    auto *diff = cast<BagValue>(state.at(op.getDiff()));

    MapVector<ElaboratorValue *, uint64_t> result;
    for (const auto &el : original->getBag()) {
      if (!diff->getBag().contains(el.first)) {
        result.insert(el);
        continue;
      }

      if (op.getInf())
        continue;

      auto toDiff = diff->getBag().lookup(el.first);
      if (el.second <= toDiff)
        continue;

      result.insert({el.first, el.second - toDiff});
    }

    internalizeResult<BagValue>(op.getResult(), std::move(result),
                                op.getType());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagUnionOp op) {
    MapVector<ElaboratorValue *, uint64_t> result;
    for (auto bag : op.getBags()) {
      auto *val = cast<BagValue>(state.at(bag));
      for (auto [el, multiple] : val->getBag())
        result[el] += multiple;
    }

    internalizeResult<BagValue>(op.getResult(), std::move(result),
                                op.getType());
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(BagUniqueSizeOp op) {
    auto size = cast<BagValue>(state.at(op.getBag()))->getBag().size();
    auto sizeAttr = IntegerAttr::get(IndexType::get(op->getContext()), size);
    internalizeResult<AttributeValue>(op.getResult(), sizeAttr);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(scf::IfOp op) {
    bool cond = cast<BoolValue>(state.at(op.getCondition()))->getBool();
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
    auto *lowerBound = dyn_cast<IndexValue>(state.at(op.getLowerBound()));
    auto *step = dyn_cast<IndexValue>(state.at(op.getStep()));
    auto *upperBound = dyn_cast<IndexValue>(state.at(op.getUpperBound()));

    if (!lowerBound || !step || !upperBound)
      return op->emitOpError("can only elaborate index type iterator");

    // Prepare for first iteration by assigning the nested regions block
    // arguments. We can just reuse this elaborator because we need access to
    // values elaborated in the parent region anyway and materialize everything
    // inline (i.e., don't need a new materializer).
    state[op.getInductionVar()] = lowerBound;
    for (auto [iterArg, initArg] :
         llvm::zip(op.getRegionIterArgs(), op.getInitArgs()))
      state[iterArg] = state.at(initArg);

    // This loop performs the actual 'scf.for' loop iterations.
    for (size_t i = lowerBound->getIndex(); i < upperBound->getIndex();
         i += step->getIndex()) {
      if (failed(elaborate(op.getBodyRegion())))
        return failure();

      // Prepare for the next iteration by updating the mapping of the nested
      // regions block arguments
      internalizeResult<IndexValue>(op.getInductionVar(), i + step->getIndex());
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
    size_t lhs = cast<IndexValue>(state.at(op.getLhs()))->getIndex();
    size_t rhs = cast<IndexValue>(state.at(op.getRhs()))->getIndex();
    internalizeResult<IndexValue>(op.getResult(), lhs + rhs);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(index::CmpOp op) {
    size_t lhs = cast<IndexValue>(state.at(op.getLhs()))->getIndex();
    size_t rhs = cast<IndexValue>(state.at(op.getRhs()))->getIndex();
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
    internalizeResult<BoolValue>(op.getResult(), result);
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
                          ArrayRef<ElaboratorValue *> regionArguments = {}) {
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
            llvm::dbgs() << *state.at(res);
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

  // Allows us to materialize ElaboratorValues to the IR operations necessary to
  // obtain an SSA value representing that elaborated value.
  Materializer &materializer;

  // A map from SSA values to a pointer of an interned elaborator value.
  DenseMap<Value, ElaboratorValue *> state;
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
  for (auto testOp : moduleOp.getOps<TestOp>()) {
    LLVM_DEBUG(llvm::dbgs()
               << "\n=== Elaborating test @" << testOp.getSymName() << "\n\n");
    Materializer materializer(OpBuilder::atBlockBegin(testOp.getBody()));
    Elaborator elaborator(state, materializer);
    if (failed(elaborator.elaborate(testOp.getBodyRegion())))
      return failure();

    materializer.finalize();
  }

  // Do top-down BFS traversal such that elaborating a sequence further down
  // does not fix the outcome for multiple placements.
  while (!state.worklist.empty()) {
    auto *curr = state.worklist.front();
    state.worklist.pop();

    if (table.lookup<SequenceOp>(curr->getName()))
      continue;

    auto familyOp = table.lookup<SequenceOp>(curr->getFamilyName());
    // TODO: don't clone if this is the only remaining reference to this
    // sequence
    OpBuilder builder(familyOp);
    auto seqOp = builder.cloneWithoutRegions(familyOp);
    seqOp.getBodyRegion().emplaceBlock();
    seqOp.setSymName(curr->getName());
    table.insert(seqOp);
    assert(seqOp.getSymName() == curr->getName() &&
           "should not have been renamed");

    LLVM_DEBUG(llvm::dbgs()
               << "\n=== Elaborating sequence family @" << familyOp.getSymName()
               << " into @" << seqOp.getSymName() << "\n\n");

    Materializer materializer(OpBuilder::atBlockBegin(seqOp.getBody()));
    Elaborator elaborator(state, materializer);
    if (failed(elaborator.elaborate(familyOp.getBodyRegion(), curr->getArgs())))
      return failure();

    materializer.finalize();
  }

  // Inline all sequences and remove the operations that place the sequences.
  for (auto testOp : moduleOp.getOps<TestOp>())
    if (failed(inlineSequences(testOp, table)))
      return failure();

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
