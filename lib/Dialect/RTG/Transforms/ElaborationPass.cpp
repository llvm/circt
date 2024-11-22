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
#include "mlir/Dialect/Arith/IR/Arith.h"
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
  enum class ValueKind { Attribute, Set, Bag };

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
} // namespace

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
  Value materialize(ElaboratorValue *val, Location loc,
                    function_ref<InFlightDiagnostic()> emitError) {
    assert(block && "must call reset before calling this function");

    auto iter = materializedValues.find(val);
    if (iter != materializedValues.end())
      return iter->second;

    LLVM_DEBUG(llvm::dbgs() << "Materializing " << *val << "\n\n");

    OpBuilder builder(block, insertionPoint);
    return TypeSwitch<ElaboratorValue *, Value>(val)
        .Case<AttributeValue, SetValue, BagValue>(
            [&](auto val) { return visit(val, builder, loc, emitError); })
        .Default([](auto val) {
          assert(false && "all cases must be covered above");
          return Value();
        });
  }

  Materializer &reset(Block *block) {
    materializedValues.clear();
    integerValues.clear();
    this->block = block;
    insertionPoint = block->begin();
    return *this;
  }

private:
  Value visit(AttributeValue *val, OpBuilder &builder, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    auto attr = val->getAttr();

    // For integer attributes (and arithmetic operations on them) we use the
    // arith dialect.
    if (isa<IntegerAttr>(attr)) {
      Value res = builder.getContext()
                      ->getLoadedDialect<arith::ArithDialect>()
                      ->materializeConstant(builder, attr, attr.getType(), loc)
                      ->getResult(0);
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

  Value visit(SetValue *val, OpBuilder &builder, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> elements;
    elements.reserve(val->getSet().size());
    for (auto *el : val->getSet()) {
      auto materialized = materialize(el, loc, emitError);
      if (!materialized)
        return Value();

      elements.push_back(materialized);
    }

    auto res = builder.create<SetCreateOp>(loc, val->getType(), elements);
    materializedValues[val] = res;
    return res;
  }

  Value visit(BagValue *val, OpBuilder &builder, Location loc,
              function_ref<InFlightDiagnostic()> emitError) {
    SmallVector<Value> values, weights;
    values.reserve(val->getBag().size());
    weights.reserve(val->getBag().size());
    for (auto [val, weight] : val->getBag()) {
      auto materializedVal = materialize(val, loc, emitError);
      if (!materializedVal)
        return Value();

      auto iter = integerValues.find(weight);
      Value materializedWeight;
      if (iter != integerValues.end()) {
        materializedWeight = iter->second;
      } else {
        materializedWeight = builder.create<arith::ConstantOp>(
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

private:
  /// Cache values we have already materialized to reuse them later. We start
  /// with an insertion point at the start of the block and cache the (updated)
  /// insertion point such that future materializations can also reuse previous
  /// materializations without running into dominance issues (or requiring
  /// additional checks to avoid them).
  DenseMap<ElaboratorValue *, Value> materializedValues;
  DenseMap<uint64_t, Value> integerValues;

  /// Cache the builders to continue insertions at their current insertion point
  /// for the reason stated above.
  Block *block;
  Block::iterator insertionPoint;
};

/// Used to signal to the elaboration driver whether the operation should be
/// removed.
enum class DeletionKind { Keep, Delete };

/// Interprets the IR to perform and lower the represented randomizations.
class Elaborator : public RTGOpVisitor<Elaborator, FailureOr<DeletionKind>> {
public:
  using RTGBase = RTGOpVisitor<Elaborator, FailureOr<DeletionKind>>;
  using RTGBase::visitOp;
  using RTGBase::visitRegisterOp;

  Elaborator(SymbolTable &table, std::mt19937 &rng) : rng(rng), table(table) {}

  /// Helper to perform internalization and keep track of interpreted value for
  /// the given SSA value.
  template <typename ValueTy, typename... Args>
  void internalizeResult(Value val, Args &&...args) {
    // TODO: this isn't the most efficient way to internalize
    auto ptr = std::make_unique<ValueTy>(std::forward<Args>(args)...);
    auto *e = ptr.get();
    auto [iter, _] = interned.insert({e, std::move(ptr)});
    state[val] = iter->second.get();
  }

  /// Print a nice error message for operations we don't support yet.
  FailureOr<DeletionKind> visitUnhandledOp(Operation *op) {
    return op->emitOpError("elaboration not supported");
  }

  FailureOr<DeletionKind> visitExternalOp(Operation *op) {
    // TODO: we only have this to be able to write tests for this pass without
    // having to add support for more operations for now, so it should be
    // removed once it is not necessary anymore for writing tests
    if (op->use_empty()) {
      for (auto &operand : op->getOpOperands()) {
        auto emitError = [&]() {
          auto diag = op->emitError();
          diag.attachNote(op->getLoc())
              << "while materializing value for operand#"
              << operand.getOperandNumber();
          return diag;
        };
        Value val = materializer.materialize(state.at(operand.get()),
                                             op->getLoc(), emitError);
        if (!val)
          return failure();
        operand.set(val);
      }
      return DeletionKind::Keep;
    }

    return visitUnhandledOp(op);
  }

  FailureOr<DeletionKind> visitOp(SequenceClosureOp op) {
    auto originalSeqOp = table.lookup<SequenceOp>(op.getSequenceAttr());
    // TODO: don't clone if only user
    auto seqOp = cast<SequenceOp>(originalSeqOp->clone());
    auto nameAttr = table.insert(seqOp);

    for (auto [val, arg] :
         llvm::zip(op.getArgs(), seqOp.getBody()->getArguments())) {
      unsigned operandNumber = arg.getArgNumber();
      auto emitError = [&]() {
        auto diag = op->emitError();
        diag.attachNote(op->getLoc())
            << "while materializing value for operand#" << operandNumber;
        return diag;
      };

      Value repl = Materializer()
                       .reset(seqOp.getBody())
                       .materialize(state.at(val), seqOp.getLoc(), emitError);
      if (!repl)
        return failure();

      arg.replaceAllUsesWith(repl);
    }

    seqOp.getBody()->eraseArguments(0, seqOp.getBody()->getNumArguments());

    worklist.push(seqOp);

    internalizeResult<AttributeValue>(op.getResult(), nameAttr);
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind> visitOp(InvokeSequenceOp op) {
    auto *sequence = cast<AttributeValue>(state.at(op.getSequence()));
    OpBuilder builder(op);
    Value seqVal = builder.create<SequenceClosureOp>(
        op.getLoc(), cast<StringAttr>(sequence->getAttr()), ValueRange());
    op.getSequenceMutable().set(seqVal);
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
      selected = getUniformlyInRange(rng, 0, set->getSet().size() - 1);
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
      auto *interpMultiple = cast<AttributeValue>(state.at(multiple));
      uint64_t m = cast<IntegerAttr>(interpMultiple->getAttr()).getInt();
      bag[interpValue] += m;
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

    auto customRng = rng;
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

  FailureOr<DeletionKind> dispatchOpVisitor(Operation *op) {
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

      internalizeResult<AttributeValue>(op->getResult(0), attr);
      return DeletionKind::Delete;
    }

    return RTGBase::dispatchOpVisitor(op);
  }

  LogicalResult elaborate(Operation *op) {
    LLVM_DEBUG({
      if (auto testOp = dyn_cast<TestOp>(op))
        llvm::dbgs() << "\n=== Elaborating Test @" << testOp.getSymName()
                     << "\n\n";
      else if (auto seqOp = dyn_cast<SequenceOp>(op))
        llvm::dbgs() << "\n=== Elaborating Sequence @" << seqOp.getSymName()
                     << "\n\n";
    });

    Block *block = &op->getRegion(0).front();
    materializer.reset(block);

    SmallVector<Operation *> toDelete;
    for (auto &op : *block) {
      if (op.getNumRegions() != 0)
        return op.emitOpError("nested regions not supported");

      auto result = dispatchOpVisitor(&op);
      if (failed(result))
        return failure();

      LLVM_DEBUG({
        llvm::dbgs() << "Elaborating " << op << " to\n[";

        llvm::interleaveComma(op.getResults(), llvm::dbgs(), [&](auto res) {
          if (state.contains(res))
            llvm::dbgs() << *state.at(res);
          else
            llvm::dbgs() << "unknown";
        });

        llvm::dbgs() << "]\n\n";
      });

      if (*result == DeletionKind::Delete)
        toDelete.push_back(&op);
    }

    for (auto *op : llvm::reverse(toDelete))
      op->erase();

    // Reduce max memory consumption and make sure the values cannot be accessed
    // anymore because we deleted the ops above. Clearing should lead to better
    // performance than having them as a local here and pass via function
    // argument.
    state.clear();
    interned.clear();

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
    // Initialize the worklist with the test ops since they cannot be placed by
    // other ops.
    for (auto testOp : moduleOp.getOps<TestOp>())
      worklist.push(testOp);

    // Do top-down BFS traversal such that elaborating a sequence further down
    // does not fix the outcome for multiple placements.
    while (!worklist.empty()) {
      auto *curr = worklist.front();
      worklist.pop();

      auto result = elaborate(curr);
      if (failed(result))
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

  /// The worklist used to keep track of the test and sequence operations to
  /// make sure they are processed top-down (BFS traversal).
  std::queue<Operation *> worklist;

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

  // A map from SSA values to a pointer of an interned elaborator value.
  DenseMap<Value, ElaboratorValue *> state;

  // Allows us to materialize ElaboratorValues to the IR operations necessary to
  // obtain an SSA value representing that elaborated value.
  Materializer materializer;
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
