//===- TestPasses.cpp - Test passes for the analysis infrastructure -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements test passes for the analysis infrastructure.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/DebugAnalysis.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Analysis/OpCountAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/GatedClockConversion.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::dataflow;
using namespace circt;
using namespace circt::analysis;
using namespace circt::scheduling;

//===----------------------------------------------------------------------===//
// DebugAnalysis
//===----------------------------------------------------------------------===//

namespace {
struct TestDebugAnalysisPass
    : public PassWrapper<TestDebugAnalysisPass, OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDebugAnalysisPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-debug-analysis"; }
  StringRef getDescription() const override {
    return "Perform debug analysis and emit results as attributes";
  }
};
} // namespace

void TestDebugAnalysisPass::runOnOperation() {
  auto *context = &getContext();
  auto &analysis = getAnalysis<DebugAnalysis>();
  for (auto *op : analysis.debugOps) {
    op->setAttr("debug.only", UnitAttr::get(context));
  }
}

//===----------------------------------------------------------------------===//
// DependenceAnalysis
//===----------------------------------------------------------------------===//

namespace {
struct TestDependenceAnalysisPass
    : public PassWrapper<TestDependenceAnalysisPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDependenceAnalysisPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-dependence-analysis"; }
  StringRef getDescription() const override {
    return "Perform dependence analysis and emit results as attributes";
  }
};
} // namespace

void TestDependenceAnalysisPass::runOnOperation() {
  MLIRContext *context = &getContext();

  MemoryDependenceAnalysis analysis(getOperation());

  getOperation().walk([&](Operation *op) {
    if (!isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      return;

    SmallVector<Attribute> deps;

    for (auto dep : analysis.getDependences(op)) {
      if (dep.dependenceType != DependenceResult::HasDependence)
        continue;

      SmallVector<Attribute> comps;
      for (auto comp : dep.dependenceComponents) {
        SmallVector<Attribute> vector;
        vector.push_back(
            IntegerAttr::get(IntegerType::get(context, 64), *comp.lb));
        vector.push_back(
            IntegerAttr::get(IntegerType::get(context, 64), *comp.ub));
        comps.push_back(ArrayAttr::get(context, vector));
      }

      deps.push_back(ArrayAttr::get(context, comps));
    }

    auto dependences = ArrayAttr::get(context, deps);
    op->setAttr("dependences", dependences);
  });
}

//===----------------------------------------------------------------------===//
// SchedulingAnalysis
//===----------------------------------------------------------------------===//

namespace {
struct TestSchedulingAnalysisPass
    : public PassWrapper<TestSchedulingAnalysisPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSchedulingAnalysisPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-scheduling-analysis"; }
  StringRef getDescription() const override {
    return "Perform scheduling analysis and emit results as attributes";
  }
};
} // namespace

void TestSchedulingAnalysisPass::runOnOperation() {
  MLIRContext *context = &getContext();

  CyclicSchedulingAnalysis analysis = getAnalysis<CyclicSchedulingAnalysis>();

  getOperation().walk([&](AffineForOp forOp) {
    if (isa<AffineForOp>(forOp.getBody()->front()))
      return;
    CyclicProblem problem = analysis.getProblem(forOp);
    forOp.getBody()->walk([&](Operation *op) {
      for (auto dep : problem.getDependences(op)) {
        assert(!dep.isInvalid());
        if (dep.isAuxiliary())
          op->setAttr("dependence", UnitAttr::get(context));
      }
    });
  });
}

//===----------------------------------------------------------------------===//
// InstanceGraph
//===----------------------------------------------------------------------===//

namespace {
struct InferTopModulePass
    : public PassWrapper<InferTopModulePass, OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferTopModulePass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-infer-top-level"; }
  StringRef getDescription() const override {
    return "Perform top level module inference and emit results as attributes "
           "on the enclosing module.";
  }
};
} // namespace

void InferTopModulePass::runOnOperation() {
  circt::hw::InstanceGraph &analysis = getAnalysis<circt::hw::InstanceGraph>();
  auto res = analysis.getInferredTopLevelNodes();
  if (failed(res)) {
    signalPassFailure();
    return;
  }

  llvm::SmallVector<Attribute, 4> attrs;
  for (auto *node : *res)
    attrs.push_back(node->getModule().getModuleNameAttr());

  analysis.getParent()->setAttr("test.top",
                                ArrayAttr::get(&getContext(), attrs));
}

//===----------------------------------------------------------------------===//
// FIRRTL Instance Info
//===----------------------------------------------------------------------===//

namespace {
struct FIRRTLInstanceInfoPass
    : public PassWrapper<FIRRTLInstanceInfoPass,
                         OperationPass<firrtl::CircuitOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FIRRTLInstanceInfoPass)

  void runOnOperation() override;
  StringRef getArgument() const override { return "test-firrtl-instance-info"; }
  StringRef getDescription() const override {
    return "Run firrtl::InstanceInfo analysis and show the results.  This pass "
           "is intended to be used for testing purposes only.";
  }
};
} // namespace

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const bool a) {
  if (a)
    return os << "true";
  return os << "false";
}

static void printCircuitInfo(firrtl::CircuitOp op,
                             firrtl::InstanceInfo &iInfo) {
  OpPrintingFlags flags;
  flags.skipRegions();
  llvm::errs() << "  - operation: ";
  op->print(llvm::errs(), flags);
  llvm::errs() << "\n"
               << "    hasDut: " << iInfo.hasDut() << "\n"
               << "    dut: ";
  if (auto dutNode = iInfo.getDut())
    dutNode->print(llvm::errs(), flags);
  else
    llvm::errs() << "null";
  llvm::errs() << "\n"
               << "    effectiveDut: ";
  iInfo.getEffectiveDut()->print(llvm::errs(), flags);
  llvm::errs() << "\n";
}

static void printModuleInfo(igraph::ModuleOpInterface op,
                            firrtl::InstanceInfo &iInfo) {
  OpPrintingFlags flags;
  flags.skipRegions();
  llvm::errs() << "  - operation: ";
  op->print(llvm::errs(), flags);
  llvm::errs()
      << "\n"
      << "    isDut: " << iInfo.isDut(op) << "\n"
      << "    anyInstanceUnderDut: " << iInfo.anyInstanceUnderDut(op) << "\n"
      << "    allInstancesUnderDut: " << iInfo.allInstancesUnderDut(op) << "\n"
      << "    anyInstanceUnderEffectiveDut: "
      << iInfo.anyInstanceUnderEffectiveDut(op) << "\n"
      << "    allInstancesUnderEffectiveDut: "
      << iInfo.allInstancesUnderEffectiveDut(op) << "\n"
      << "    anyInstanceUnderLayer: " << iInfo.anyInstanceUnderLayer(op)
      << "\n"
      << "    allInstancesUnderLayer: " << iInfo.allInstancesUnderLayer(op)
      << "\n"
      << "    anyInstanceInDesign: " << iInfo.anyInstanceInDesign(op) << "\n"
      << "    allInstancesInDesign: " << iInfo.allInstancesInDesign(op) << "\n"
      << "    anyInstanceInEffectiveDesign: "
      << iInfo.anyInstanceInEffectiveDesign(op) << "\n"
      << "    allInstancesInEffectiveDesign: "
      << iInfo.allInstancesInEffectiveDesign(op) << "\n"
      << "    anyInstanceInInstanceChoice: "
      << iInfo.anyInstanceInInstanceChoice(op) << "\n"
      << "    moduleContainsProperties: " << iInfo.moduleContainsProperties(op)
      << "\n";
}

void FIRRTLInstanceInfoPass::runOnOperation() {
  auto &iInfo = getAnalysis<firrtl::InstanceInfo>();

  printCircuitInfo(getOperation(), iInfo);
  for (auto op :
       getOperation().getBodyBlock()->getOps<igraph::ModuleOpInterface>())
    printModuleInfo(op, iInfo);
}

//===----------------------------------------------------------------------===//
// FIRRTL GatedClockConversion
//===----------------------------------------------------------------------===//

namespace {
/// A brute-force oracle for `ClockAliasAnalysis`: trace every clock back to
/// its source along every instance path, and check that each claimed alias
/// holds on every pair of paths that the claim relates.
class ClockAliasOracle {
public:
  ClockAliasOracle(firrtl::CircuitOp circuit, firrtl::InstanceGraph &ig,
                   const firrtl::GatedClockConversion::ClockAliasAnalysis &aa)
      : circuit(circuit), ig(ig), aa(aa) {}

  /// Returns the number of violations found, each reported as an error.
  unsigned verify();

private:
  // An instance path, interned. Path 0..numRoots-1 are the root modules.
  struct Path {
    unsigned parent;         // ~0u for a root.
    firrtl::InstanceOp inst; // null for a root.
    firrtl::FModuleOp mod;   // The module this path instantiates.
    unsigned depth;
  };
  // (value, path) pairs identify a concrete signal.
  using Signal = std::pair<Value, unsigned>;

  unsigned extend(unsigned parent, firrtl::InstanceOp inst,
                  firrtl::FModuleOp mod);
  void enumeratePaths();
  std::optional<Signal> step(Signal s);
  Signal source(Signal s);
  /// The prefix of path `p` that ends at module `d`, or ~0u.
  unsigned prefixAt(unsigned p, firrtl::FModuleOp d) const;
  /// Modules on every path of `mod`, in path order.
  SmallVector<firrtl::FModuleOp> dominators(firrtl::FModuleOp mod) const;
  firrtl::FModuleOp moduleOf(Value v) const;
  std::string name(Value v) const;
  unsigned checkPair(Value a, Value b);

  firrtl::CircuitOp circuit;
  firrtl::InstanceGraph &ig;
  const firrtl::GatedClockConversion::ClockAliasAnalysis &aa;

  SmallVector<Path> paths;
  DenseMap<std::pair<unsigned, Operation *>, unsigned> pathIds;
  DenseMap<Operation *, SmallVector<unsigned>> pathsOf;
  DenseMap<Signal, Signal> sources;
  bool truncated = false;
};
} // namespace

unsigned ClockAliasOracle::extend(unsigned parent, firrtl::InstanceOp inst,
                                  firrtl::FModuleOp mod) {
  auto [it, inserted] = pathIds.try_emplace({parent, inst}, paths.size());
  if (inserted) {
    paths.push_back({parent, inst, mod, paths[parent].depth + 1});
    pathsOf[mod].push_back(it->second);
  }
  return it->second;
}

void ClockAliasOracle::enumeratePaths() {
  SmallVector<unsigned> worklist;
  for (auto *node : ig)
    if (node->noUses())
      if (auto mod = dyn_cast_or_null<firrtl::FModuleOp>(
              node->getModule().getOperation())) {
        paths.push_back({~0u, {}, mod, 0});
        pathsOf[mod].push_back(paths.size() - 1);
        worklist.push_back(paths.size() - 1);
      }
  while (!worklist.empty()) {
    unsigned p = worklist.pop_back_val();
    if (paths.size() > 20000) {
      truncated = true;
      return;
    }
    firrtl::FModuleOp mod = paths[p].mod;
    mod.walk([&](firrtl::InstanceOp inst) {
      auto child = dyn_cast_or_null<firrtl::FModuleOp>(
          inst.getReferencedModule(ig).getOperation());
      if (child)
        worklist.push_back(extend(p, inst, child));
    });
  }
}

static Value driverOf(Value v) {
  for (auto *user : v.getUsers())
    if (auto connect = dyn_cast<firrtl::FConnectLike>(user))
      if (connect.getDest() == v)
        return connect.getSrc();
  return Value();
}

std::optional<ClockAliasOracle::Signal> ClockAliasOracle::step(Signal s) {
  auto [v, p] = s;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    auto mod = dyn_cast<firrtl::FModuleOp>(arg.getOwner()->getParentOp());
    if (!mod)
      return std::nullopt;
    if (mod.getPortDirection(arg.getArgNumber()) == firrtl::Direction::In) {
      if (paths[p].parent == ~0u)
        return std::nullopt;
      return Signal{paths[p].inst.getResult(arg.getArgNumber()),
                    paths[p].parent};
    }
    if (Value d = driverOf(v))
      return Signal{d, p};
    return std::nullopt;
  }
  Operation *op = v.getDefiningOp();
  if (auto inst = dyn_cast<firrtl::InstanceOp>(op)) {
    unsigned idx = cast<OpResult>(v).getResultNumber();
    if (inst.getPortDirection(idx) == firrtl::Direction::Out) {
      auto child = dyn_cast_or_null<firrtl::FModuleOp>(
          inst.getReferencedModule(ig).getOperation());
      if (!child)
        return std::nullopt;
      auto it = pathIds.find({p, inst});
      if (it == pathIds.end())
        return std::nullopt;
      return Signal{child.getBodyBlock()->getArgument(idx), it->second};
    }
    if (Value d = driverOf(v))
      return Signal{d, p};
    return std::nullopt;
  }
  if (isa<firrtl::WireOp>(op)) {
    if (Value d = driverOf(v))
      return Signal{d, p};
    return std::nullopt;
  }
  if (auto node = dyn_cast<firrtl::NodeOp>(op))
    return Signal{node.getInput(), p};
  if (isa<firrtl::AsUIntPrimOp, firrtl::AsSIntPrimOp, firrtl::AsClockPrimOp,
          firrtl::AsAsyncResetPrimOp>(op))
    return Signal{op->getOperand(0), p};
  if (auto gate = dyn_cast<firrtl::ClockGateIntrinsicOp>(op))
    return Signal{gate.getInput(), p};
  return std::nullopt;
}

ClockAliasOracle::Signal ClockAliasOracle::source(Signal s) {
  SmallVector<Signal> chain;
  DenseMap<Signal, unsigned> onChain;
  Signal cur = s;
  Signal result;
  while (true) {
    if (auto it = sources.find(cur); it != sources.end()) {
      result = it->second;
      break;
    }
    auto [it, inserted] = onChain.try_emplace(cur, chain.size());
    if (!inserted) {
      // A loop: every signal on it is the same; pick a canonical one.
      result = cur;
      for (unsigned i = it->second; i < chain.size(); ++i)
        if (std::make_pair(chain[i].first.getAsOpaquePointer(),
                           chain[i].second) <
            std::make_pair(result.first.getAsOpaquePointer(), result.second))
          result = chain[i];
      break;
    }
    chain.push_back(cur);
    auto next = step(cur);
    if (!next) {
      result = cur;
      break;
    }
    cur = *next;
  }
  for (Signal c : chain)
    sources[c] = result;
  return result;
}

firrtl::FModuleOp ClockAliasOracle::moduleOf(Value v) const {
  if (auto arg = dyn_cast<BlockArgument>(v))
    return dyn_cast<firrtl::FModuleOp>(arg.getOwner()->getParentOp());
  return v.getDefiningOp()->getParentOfType<firrtl::FModuleOp>();
}

unsigned ClockAliasOracle::prefixAt(unsigned p, firrtl::FModuleOp d) const {
  for (; p != ~0u; p = paths[p].parent)
    if (paths[p].mod == d)
      return p;
  return ~0u;
}

SmallVector<firrtl::FModuleOp>
ClockAliasOracle::dominators(firrtl::FModuleOp mod) const {
  auto it = pathsOf.find(mod);
  if (it == pathsOf.end() || it->second.empty())
    return {};
  auto modulesOn = [&](unsigned p) {
    SmallVector<firrtl::FModuleOp> mods;
    for (; p != ~0u; p = paths[p].parent)
      mods.push_back(paths[p].mod);
    std::reverse(mods.begin(), mods.end());
    return mods;
  };
  SmallVector<firrtl::FModuleOp> result = modulesOn(it->second.front());
  for (unsigned p : it->second) {
    auto mods = modulesOn(p);
    DenseSet<Operation *> on;
    for (auto m : mods)
      on.insert(m);
    llvm::erase_if(result, [&](firrtl::FModuleOp m) { return !on.count(m); });
  }
  return result;
}

std::string ClockAliasOracle::name(Value v) const {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << moduleOf(v).getModuleName() << ".";
  if (auto arg = dyn_cast<BlockArgument>(v))
    os << moduleOf(v).getPortName(arg.getArgNumber());
  else
    v.printAsOperand(os, OpPrintingFlags());
  return s;
}

unsigned ClockAliasOracle::checkPair(Value a, Value b) {
  firrtl::FModuleOp ma = moduleOf(a), mb = moduleOf(b);
  if (!ma || !mb)
    return 0;
  // The deepest module that dominates both.
  auto domA = dominators(ma), domB = dominators(mb);
  DenseSet<Operation *> inB;
  for (auto m : domB)
    inB.insert(m);
  firrtl::FModuleOp d;
  for (auto m : domA)
    if (inB.count(m))
      d = m;
  if (!d)
    return 0;
  // Group the paths of each module by their instance of `d`.
  DenseMap<unsigned, SmallVector<unsigned>> byPrefixB;
  for (unsigned q : pathsOf.lookup(mb))
    byPrefixB[prefixAt(q, d)].push_back(q);
  for (unsigned p : pathsOf.lookup(ma)) {
    unsigned prefix = prefixAt(p, d);
    for (unsigned q : byPrefixB.lookup(prefix)) {
      Signal sa = source({a, p}), sb = source({b, q});
      if (sa != sb) {
        mlir::emitError(a.getLoc())
            << "clock alias oracle: " << name(a) << " and " << name(b)
            << " are claimed to alias but come from " << name(sa.first)
            << " and " << name(sb.first);
        return 1;
      }
    }
  }
  return 0;
}

unsigned ClockAliasOracle::verify() {
  enumeratePaths();
  if (truncated)
    return 0;
  unsigned errors = 0;

  // Every value of the IR, to catch a tracked value that no longer exists.
  DenseSet<Value> live;
  SmallVector<Value> tracked;
  circuit.walk([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          live.insert(arg);
    for (Value result : op->getResults())
      live.insert(result);
  });
  circuit.walk([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          if (aa.isTracked(arg))
            tracked.push_back(arg);
    for (Value result : op->getResults())
      if (aa.isTracked(result))
        tracked.push_back(result);
  });

  // Group by representative and check the class API is consistent.
  llvm::MapVector<Value, SmallVector<Value>> classes;
  for (Value v : tracked) {
    Value rep = aa.getRepresentative(v);
    if (!rep || !aa.isTracked(rep) || !aa.alias(v, rep) || !live.count(rep)) {
      mlir::emitError(v.getLoc()) << "clock alias oracle: bad representative";
      ++errors;
      continue;
    }
    classes[rep].push_back(v);
  }
  for (auto &[rep, members] : classes) {
    auto set = aa.aliasSet(rep);
    for (Value m : set)
      if (!live.count(m)) {
        mlir::emitError(rep.getLoc())
            << "clock alias oracle: class contains a value not in the IR";
        ++errors;
      }
    if (set.size() != members.size()) {
      mlir::emitError(rep.getLoc())
          << "clock alias oracle: aliasSet has " << set.size()
          << " members, but " << members.size() << " IR values map to it";
      ++errors;
    }
    Value base = aa.getBaseClock(rep);
    if (base && !aa.alias(base, rep)) {
      mlir::emitError(rep.getLoc())
          << "clock alias oracle: base clock is not in its class";
      ++errors;
    }
    for (unsigned i = 0; i < members.size(); ++i)
      for (unsigned j = i + 1; j < members.size(); ++j)
        errors += checkPair(members[i], members[j]);
  }
  return errors;
}

namespace {
struct FIRRTLGatedClockConversionPass
    : public PassWrapper<FIRRTLGatedClockConversionPass,
                         OperationPass<firrtl::CircuitOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FIRRTLGatedClockConversionPass)

  FIRRTLGatedClockConversionPass() = default;
  FIRRTLGatedClockConversionPass(const FIRRTLGatedClockConversionPass &other)
      : PassWrapper(other) {}

  void runOnOperation() override;
  StringRef getArgument() const override {
    return "test-firrtl-gated-clock-conversion";
  }
  StringRef getDescription() const override {
    return "Run firrtl::GatedClockConversion utility and show the results.  "
           "This pass is intended to be used for testing purposes only.";
  }

  Option<bool> printClockAliases{
      *this, "print-clock-aliases",
      llvm::cl::desc("Print the clock alias classes after the conversion"),
      llvm::cl::init(false)};
  Option<bool> verifyClockAliases{
      *this, "verify-clock-aliases",
      llvm::cl::desc("Check every clock alias against a brute-force oracle"),
      llvm::cl::init(false)};
};
} // namespace

void FIRRTLGatedClockConversionPass::runOnOperation() {
  auto circuit = getOperation();
  auto &instanceGraph = getAnalysis<firrtl::InstanceGraph>();
  firrtl::GatedClockConversion converter(instanceGraph);

  // Collect all register and ref force/release operations
  circuit.walk([&](Operation *op) {
    if (isa<firrtl::RegOp, firrtl::RegResetOp, firrtl::RefForceOp,
            firrtl::RefReleaseOp>(op)) {
      if (failed(converter.addRoot(op)))
        return signalPassFailure();
    }
  });

  // Run the conversion
  if (failed(converter.run()))
    return signalPassFailure();

  if (printClockAliases && converter.hasClockAliases()) {
    llvm::outs() << "clock-aliases @" << circuit.getName() << "\n";
    converter.getClockAliases().print(llvm::outs());
  }
  if (verifyClockAliases && converter.hasClockAliases() &&
      ClockAliasOracle(circuit, instanceGraph, converter.getClockAliases())
          .verify())
    return signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Comb IntRange Analysis
//===----------------------------------------------------------------------===//

namespace {
struct TestCombIntegerRangeAnalysisPass
    : public PassWrapper<TestCombIntegerRangeAnalysisPass,
                         OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCombIntegerRangeAnalysisPass)

  void runOnOperation() override;
  StringRef getArgument() const override {
    return "test-comb-int-range-analysis";
  }
  StringRef getDescription() const override {
    return "Perform integer range analysis on comb dialect and set results as "
           "attributes.";
  }
};
} // namespace

void TestCombIntegerRangeAnalysisPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *ctx = op->getContext();
  DataFlowSolver solver;
  solver.load<DeadCodeAnalysis>();
  solver.load<IntegerRangeAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();

  // Append the integer range analysis as an operation attribute.
  op->walk([&](Operation *op) {
    for (auto value : op->getResults()) {
      if (auto *range = solver.lookupState<IntegerValueRangeLattice>(value)) {
        // All analyzed comb operations should return a single result.
        assert(op->getResults().size() == 1 &&
               "Expected a single result for the operation analysis");
        assert(!range->getValue().isUninitialized() &&
               "Expected a valid range for the value");
        auto interval = range->getValue().getValue();
        auto smax = interval.smax();
        auto smaxAttr =
            IntegerAttr::get(IntegerType::get(ctx, smax.getBitWidth()), smax);
        op->setAttr("smax", smaxAttr);
        auto smin = interval.smin();
        auto sminAttr =
            IntegerAttr::get(IntegerType::get(ctx, smin.getBitWidth()), smin);
        op->setAttr("smin", sminAttr);
        auto umax = interval.umax();
        auto umaxAttr = IntegerAttr::get(
            IntegerType::get(ctx, umax.getBitWidth(), IntegerType::Unsigned),
            umax);
        op->setAttr("umax", umaxAttr);
        auto umin = interval.umin();
        auto uminAttr = IntegerAttr::get(
            IntegerType::get(ctx, umin.getBitWidth(), IntegerType::Unsigned),
            umin);
        op->setAttr("umin", uminAttr);
      }
    }
  });
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace circt {
namespace test {
void registerAnalysisTestPasses() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<TestDependenceAnalysisPass>();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<TestSchedulingAnalysisPass>();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<TestDebugAnalysisPass>();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<InferTopModulePass>();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<FIRRTLInstanceInfoPass>();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<FIRRTLGatedClockConversionPass>();
  });
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<TestCombIntegerRangeAnalysisPass>();
  });
}
} // namespace test
} // namespace circt
