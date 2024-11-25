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
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include <deque>
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

#define DEBUG_TYPE "rtg-elaboration"

namespace {

//===----------------------------------------------------------------------===//
// Elaborator Values
//===----------------------------------------------------------------------===//

/// The base class for elaborated values. Using the base class directly
/// represents an opaque value, i.e., an SSA value which we cannot further
/// interpret.
/// Derived classes will also hold the opaque value such that we always have an
/// SSA value at hand that we can use as a replacement for the concrete
/// interpreted value when needed (alternatively we could materialize new IR,
/// but that's more expensive).
/// The derived classes are supposed to
/// * add fields necessary to hold the concrete value
/// * override the virtual methods to compute equivalence based on the
///   interpreted values
/// * implement LLVM's RTTI mechanism
class ElaboratorValue {
public:
  /// Creates an opaque value.
  ElaboratorValue(Value value) : ElaboratorValue(value, true) {
    assert(value && "null values not allowed");
  }

  virtual ~ElaboratorValue() {}

  Type getType() const { return value.getType(); }
  Value getOpaqueValue() const { return value; }
  bool isOpaqueValue() const { return isOpaque; }
  virtual bool containsOpaqueValue() const { return isOpaque; }

  virtual llvm::hash_code getHashValue() const {
    return llvm::hash_combine(value, isOpaque);
  }

  virtual bool isEqual(const ElaboratorValue &other) const {
    return isOpaque == other.isOpaque && value == other.value;
  }

  virtual std::string toString() const {
    std::string out;
    llvm::raw_string_ostream stream(out);
    stream << "<opaque ";
    value.print(stream);
    stream << " at " << this << ">";
    return out;
  }

protected:
  ElaboratorValue(Value value, bool isOpaque)
      : isOpaque(isOpaque), value(value) {}

private:
  const bool isOpaque;
  const Value value;
};

/// Holds an evaluated value of a `SetType`'d value.
class SetValue : public ElaboratorValue {
public:
  SetValue(Value value, SmallVector<ElaboratorValue *> &&set)
      : ElaboratorValue(value, false), debug(false), set(set) {
    assert(isa<SetType>(value.getType()));

    // Make sure the vector is sorted and has no duplicates.
    llvm::sort(this->set);
    this->set.erase(std::unique(this->set.begin(), this->set.end()),
                    this->set.end());
  }

  SetValue(Value value, SmallVector<ElaboratorValue *> &&set,
           SmallVector<ElaboratorValue *> &&debugMap)
      : ElaboratorValue(value, false), debug(true), set(set),
        debugMap(debugMap) {
    assert(isa<SetType>(value.getType()));

    // Make sure the vector is sorted and has no duplicates.
    llvm::sort(this->set);
    this->set.erase(std::unique(this->set.begin(), this->set.end()),
                    this->set.end());
  }

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue *val) {
    return !val->isOpaqueValue() && SetType::classof(val->getType());
  }

  bool containsOpaqueValue() const override {
    return llvm::any_of(set, [](auto el) { return el->containsOpaqueValue(); });
  }

  llvm::hash_code getHashValue() const override {
    return llvm::hash_combine_range(set.begin(), set.end());
  }

  bool isEqual(const ElaboratorValue &other) const override {
    auto *otherSet = dyn_cast<SetValue>(&other);
    if (!otherSet)
      return false;
    return set == otherSet->set;
  }

  std::string toString() const override {
    assert(debug && "must be in debug mode");

    std::string out;
    llvm::raw_string_ostream stream(out);
    stream << "<set {";
    DenseSet<ElaboratorValue *> visited;
    for (auto *val : debugMap) {
      if (visited.contains(val))
        continue;
      if (!visited.empty())
        stream << ", ";
      visited.insert(val);
      stream << val->toString();
    }
    stream << "} at " << this << ">";
    return out;
  }

  ArrayRef<ElaboratorValue *> getAsArrayRef() const { return set; }

  ArrayRef<ElaboratorValue *> getDebugMap() const {
    assert(debug && "must be in debug mode");
    return debugMap;
  }

private:
  // Whether this set was constructed in debug mode.
  const bool debug;

  // We currently use a sorted vector to represent sets. Note that it is sorted
  // by the pointer value and thus non-deterministic.
  // We probably want to do some profiling in the future to see if a DenseSet or
  // other representation is better suited.
  SmallVector<ElaboratorValue *> set;

  // A map to guarantee deterministic behavior when the 'rtg.elaboration'
  // attribute is used in debug mode.
  SmallVector<ElaboratorValue *> debugMap;
};

/// Holds an evaluated value of a `SequenceType`'d value.
class SequenceClosureValue : public ElaboratorValue {
public:
  SequenceClosureValue(Value value, StringAttr sequence,
                       ArrayRef<ElaboratorValue *> args)
      : ElaboratorValue(value, false), sequence(sequence), args(args) {
    assert(isa<SequenceType>(value.getType()));
  }

  // Implement LLVMs RTTI
  static bool classof(const ElaboratorValue *val) {
    return !val->isOpaqueValue() && SequenceType::classof(val->getType());
  }

  bool containsOpaqueValue() const override {
    return llvm::any_of(args,
                        [](auto el) { return el->containsOpaqueValue(); });
  }

  llvm::hash_code getHashValue() const override {
    return llvm::hash_combine(
        sequence, llvm::hash_combine_range(args.begin(), args.end()));
  }

  bool isEqual(const ElaboratorValue &other) const override {
    auto *seq = dyn_cast<SequenceClosureValue>(&other);
    if (!seq)
      return false;

    return sequence == seq->sequence && args == seq->args;
  }

  std::string toString() const override {
    std::string out;
    llvm::raw_string_ostream stream(out);
    stream << "<sequence @" << sequence.getValue() << "(";
    llvm::interleaveComma(args, stream,
                          [&](auto *val) { stream << val->toString(); });
    stream << ") at " << this << ">";
    return out;
  }

  StringAttr getSequence() const { return sequence; }
  ArrayRef<ElaboratorValue *> getArgs() const { return args; }

private:
  StringAttr sequence;
  SmallVector<ElaboratorValue *> args;
};

//===----------------------------------------------------------------------===//
// Hash Map Helpers
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(readability-identifier-naming)
inline llvm::hash_code hash_value(const ElaboratorValue &val) {
  return val.getHashValue();
}

struct InternMapInfo : public DenseMapInfo<ElaboratorValue *> {
  static unsigned getHashValue(const ElaboratorValue *value) {
    auto *tk = getTombstoneKey();
    auto *ek = getEmptyKey();
    if (value == tk || value == ek)
      return DenseMapInfo<ElaboratorValue *>::getHashValue(value);

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

//===----------------------------------------------------------------------===//
// Main Elaborator Implementation
//===----------------------------------------------------------------------===//

/// Used to signal to the elaboration driver whether the operation should be
/// removed.
enum class DeletionKind { Keep, Delete };

/// Interprets the IR to perform and lower the represented randomizations.
class Elaborator : public RTGOpVisitor<Elaborator, FailureOr<DeletionKind>,
                                       function_ref<void(Operation *)>> {
public:
  using RTGOpVisitor<Elaborator, FailureOr<DeletionKind>,
                     function_ref<void(Operation *)>>::visitOp;

  Elaborator(SymbolTable &table, const ElaborationOptions &options)
      : options(options), symTable(table) {

    // Initialize the RNG
    std::random_device dev;
    unsigned s = dev();
    if (options.seed.has_value())
      s = *options.seed;
    rng = std::mt19937(s);
  }

  /// Helper to perform internalization and keep track of interpreted value for
  /// the given SSA value.
  template <typename ValueTy, typename... Args>
  void internalizeResult(Value val, Args &&...args) {
    auto ptr = std::make_unique<ValueTy>(val, std::forward<Args>(args)...);
    auto *e = ptr.get();
    auto [iter, _] = interned.insert({e, std::move(ptr)});
    state[val] = iter->second.get();
  }

  /// Print a nice error message for operations we don't support yet.
  FailureOr<DeletionKind>
  visitUnhandledOp(Operation *op,
                   function_ref<void(Operation *)> addToWorklist) {
    return op->emitError("elaboration not supported");
  }

  FailureOr<DeletionKind>
  visitExternalOp(Operation *op,
                  function_ref<void(Operation *)> addToWorklist) {
    // Treat values defined by external ops as opaque, non-elaborated values.
    for (auto res : op->getResults())
      internalizeResult<ElaboratorValue>(res);

    return DeletionKind::Keep;
  }

  FailureOr<DeletionKind>
  visitOp(SequenceClosureOp op, function_ref<void(Operation *)> addToWorklist) {
    SmallVector<ElaboratorValue *> args;
    args.reserve(op.getArgs().size());
    for (auto arg : op.getArgs())
      args.push_back(state.at(arg));

    internalizeResult<SequenceClosureValue>(op.getResult(),
                                            op.getSequenceAttr(), args);

    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind>
  visitOp(InvokeSequenceOp op, function_ref<void(Operation *)> addToWorklist) {
    auto *sequenceClosure =
        cast<SequenceClosureValue>(state.at(op.getSequence()));

    IRRewriter rewriter(op);
    auto sequence = symTable.lookupNearestSymbolFrom<SequenceOp>(
        op->getParentOfType<ModuleOp>(), sequenceClosure->getSequence());
    auto *clone = sequence->clone();
    SmallVector<Value> args;
    args.reserve(sequenceClosure->getArgs().size());
    for (auto &arg : sequenceClosure->getArgs()) {
      Value val = arg->getOpaqueValue();
      // Note: If the value is defined inside the same block it must be before
      // this op as we would already have a dominance violation to start with.
      if (val.getParentBlock() != op->getBlock())
        return op.emitError("closure argument defined outside this block");

      args.push_back(val);
    }

    for (auto &op : clone->getRegion(0).front())
      addToWorklist(&op);

    rewriter.inlineBlockBefore(&clone->getRegion(0).front(), op, args);
    clone->erase();
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind>
  visitOp(SetCreateOp op, function_ref<void(Operation *)> addToWorklist) {
    SmallVector<ElaboratorValue *> set;
    for (auto val : op.getElements()) {
      auto *interpValue = state.at(val);
      if (interpValue->containsOpaqueValue())
        return op->emitError("cannot create a set of opaque values because "
                             "they cannot be reliably uniqued");
      set.emplace_back(interpValue);
    }

    if (options.debugMode) {
      SmallVector<ElaboratorValue *> debugMap(set);
      internalizeResult<SetValue>(op.getSet(), std::move(set),
                                  std::move(debugMap));
      return DeletionKind::Delete;
    }

    internalizeResult<SetValue>(op.getSet(), std::move(set));
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind>
  visitOp(SetSelectRandomOp op, function_ref<void(Operation *)> addToWorklist) {
    auto *set = cast<SetValue>(state.at(op.getSet()));

    ElaboratorValue *selected;
    if (options.debugMode) {
      auto intAttr = op->getAttrOfType<IntegerAttr>("rtg.elaboration");
      size_t originalSize = set->getDebugMap().size();
      if (originalSize != set->getAsArrayRef().size())
        op->emitWarning("set contained ")
            << (originalSize - set->getAsArrayRef().size())
            << " duplicate value(s), the value at index " << intAttr.getInt()
            << " might not be the intended one";
      if (originalSize <= intAttr.getValue().getZExtValue())
        return op->emitError("'rtg.elaboration' attribute value out of bounds, "
                             "must be between 0 (incl.) and ")
               << originalSize << " (excl.)";
      selected = set->getDebugMap()[intAttr.getInt()];
    } else {
      std::uniform_int_distribution<size_t> dist(
          0, set->getAsArrayRef().size() - 1);
      selected = set->getAsArrayRef()[dist(rng)];
    }

    state[op.getResult()] = selected;
    return DeletionKind::Delete;
  }

  FailureOr<DeletionKind>
  visitOp(SetDifferenceOp op, function_ref<void(Operation *)> addToWorklist) {
    auto *originalElaboratorValue = cast<SetValue>(state.at(op.getOriginal()));
    auto original = originalElaboratorValue->getAsArrayRef();
    auto diff = cast<SetValue>(state.at(op.getDiff()))->getAsArrayRef();

    SmallVector<ElaboratorValue *> result;
    std::set_difference(original.begin(), original.end(), diff.begin(),
                        diff.end(), std::inserter(result, result.end()));

    if (options.debugMode) {
      DenseSet<ElaboratorValue *> diffSet(diff.begin(), diff.end());
      SmallVector<ElaboratorValue *> debugMap;
      for (auto *el : originalElaboratorValue->getDebugMap())
        if (!diffSet.contains(el))
          debugMap.push_back(el);

      internalizeResult<SetValue>(op.getResult(), std::move(result),
                                  std::move(debugMap));
      return DeletionKind::Delete;
    }

    internalizeResult<SetValue>(op.getResult(), std::move(result));
    return DeletionKind::Delete;
  }

  LogicalResult elaborate(TestOp testOp) {
    LLVM_DEBUG(llvm::dbgs()
               << "\n=== Elaborating Test @" << testOp.getSymName() << "\n\n");

    DenseSet<Operation *> visited;
    std::deque<Operation *> worklist;
    DenseSet<Operation *> toDelete;
    for (auto &op : *testOp.getBody())
      if (op.use_empty())
        worklist.push_back(&op);

    while (!worklist.empty()) {
      auto *curr = worklist.back();
      if (visited.contains(curr)) {
        worklist.pop_back();
        continue;
      }

      if (curr->getNumRegions() != 0)
        return curr->emitError("nested regions not supported");

      bool addedSomething = false;
      for (auto val : curr->getOperands()) {
        if (state.contains(val))
          continue;

        auto *defOp = val.getDefiningOp();
        assert(defOp && "cannot be a BlockArgument here");
        if (!visited.contains(defOp)) {
          worklist.push_back(defOp);
          addedSomething = true;
        }
      }

      if (addedSomething)
        continue;

      auto addToWorklist = [&](Operation *op) {
        if (op->use_empty())
          worklist.push_front(op);
      };
      auto result = dispatchOpVisitor(curr, addToWorklist);
      if (failed(result))
        return failure();

      LLVM_DEBUG({
        if (options.debugMode) {
          llvm::dbgs() << "Elaborating " << *curr << " to\n[";

          llvm::interleaveComma(curr->getResults(), llvm::dbgs(),
                                [&](auto res) {
                                  if (state.contains(res))
                                    llvm::dbgs() << state.at(res)->toString();
                                  else
                                    llvm::dbgs() << "unknown";
                                });

          llvm::dbgs() << "]\n\n";
        }
      });

      if (*result == DeletionKind::Delete)
        toDelete.insert(curr);

      visited.insert(curr);
      worklist.pop_back();
    }

    // FIXME: this assumes that we didn't query the opaque value from an
    // interpreted elaborator value in a way that it can remain used in the IR.
    for (auto *op : toDelete) {
      op->dropAllUses();
      op->erase();
    }

    // Reduce max memory consumption and make sure the values cannot be accessed
    // anymore because we deleted the ops above.
    state.clear();
    interned.clear();

    return success();
  }

private:
  const ElaborationOptions &options;
  std::mt19937 rng;

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

  SymbolTable symTable;
};

//===----------------------------------------------------------------------===//
// Elaborator Pass
//===----------------------------------------------------------------------===//

class ElaborationPass : public rtg::impl::ElaborationPassBase<ElaborationPass> {
public:
  ElaborationPass() : ElaborationPassBase() {}
  ElaborationPass(const ElaborationPass &other) : ElaborationPassBase(other) {}
  ElaborationPass(const rtg::ElaborationOptions &options)
      : ElaborationPassBase(), options(options) {
    if (options.seed.has_value())
      seed = options.seed.value();
    debugMode = options.debugMode;
  }
  void runOnOperation() override;
  void cloneTargetsIntoTests();

private:
  ElaborationOptions options;
  Pass::Option<unsigned> seed{*this, "seed",
                              llvm::cl::desc("Seed for the RNG.")};
  Pass::Option<bool> debugMode{*this, "debug",
                               llvm::cl::desc("Enable debug mode."),
                               llvm::cl::init(false)};
};
} // end namespace

void ElaborationPass::runOnOperation() {
  if (seed.hasValue())
    options.seed = seed;
  options.debugMode = debugMode;

  cloneTargetsIntoTests();

  auto moduleOp = getOperation();
  SymbolTable table(moduleOp);
  Elaborator elaborator(table, options);

  for (auto testOp : moduleOp.getOps<TestOp>())
    if (failed(elaborator.elaborate(testOp)))
      return signalPassFailure();
}

void ElaborationPass::cloneTargetsIntoTests() {
  auto moduleOp = getOperation();
  for (auto target : llvm::make_early_inc_range(moduleOp.getOps<TargetOp>())) {
    for (auto test : moduleOp.getOps<TestOp>()) {
      // If the test requries nothing from a target, we can always run it.
      if (test.getTarget().getEntryTypes().empty())
        continue;

      // If the target requirements do not match, skip this test
      // TODO: allow target refinements, just not coarsening
      if (target.getTarget() != test.getTarget())
        continue;

      IRRewriter rewriter(test);
      // Create a new test for the matched target
      auto newTest = cast<TestOp>(rewriter.clone(*test));
      newTest.setSymName(test.getSymName().str() + "_" +
                         target.getSymName().str());

      // Copy the target body into the newly created test
      rewriter.setInsertionPoint(target);
      auto newTarget = cast<TargetOp>(rewriter.clone(*target));
      auto *yield = newTarget.getBody()->getTerminator();
      rewriter.inlineBlockBefore(newTarget.getBody(), newTest.getBody(),
                                 newTest.getBody()->begin());
      rewriter.replaceAllUsesWith(newTest.getBody()->getArguments(),
                                  yield->getOperands());
      newTest.getBody()->eraseArguments(0,
                                        newTest.getBody()->getNumArguments());
      rewriter.eraseOp(yield);
      rewriter.eraseOp(newTarget);
      newTest.setTarget(DictType::get(&getContext(), {}, {}));
    }

    target->erase();
  }

  // Erase all remaining non-matched tests.
  for (auto test : llvm::make_early_inc_range(moduleOp.getOps<TestOp>()))
    if (!test.getTarget().getEntryNames().empty())
      test->erase();
}

std::unique_ptr<Pass> rtg::createElaborationPass() {
  return std::make_unique<ElaborationPass>();
}

std::unique_ptr<Pass>
rtg::createElaborationPass(const rtg::ElaborationOptions &options) {
  return std::make_unique<ElaborationPass>(options);
}
