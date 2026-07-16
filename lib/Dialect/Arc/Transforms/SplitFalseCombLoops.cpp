//===- SplitFalseCombLoops.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ConvertToArcs' fan-in analysis detects combinational loops at whole-SSA-value
// granularity. Aggregate update chains of the form "write element i, read
// element j" (unrolled per-way cache state updates, register files, struct
// field updates) and narrow bitwise feedback (bit 0 depends on bit 1) form
// SSA cycles in the `hw.module` graph region even though they are perfectly
// acyclic per array element, struct field, or bit. This pass finds strongly
// connected components among non-arc-breaking combinational ops and, as long
// as every op in a component is decomposable, splits the participating values
// one type level at a time (arrays into elements, structs into fields, and --
// capped by an option -- integers into bits). Genuine loops are left in place
// for ConvertToArcs to diagnose.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "arc-split-false-comb-loops"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_SPLITFALSECOMBLOOPS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;
using llvm::SmallDenseMap;

/// Mirror of ConvertToArcs' isArcBreakingOp: ops whose results do not carry a
/// combinational dependency for the purposes of the fan-in loop analysis.
static bool isCombBreaker(mlir::Operation *op) {
  if (isa<arc::TapOp>(op))
    return false;
  if (isa<llhd::ProbeOp>(op))
    return true;
  if (llvm::any_of(op->getResults(), [](mlir::Value v) {
        return isa<llhd::RefType>(v.getType());
      }))
    return true;
  return op->hasTrait<mlir::OpTrait::ConstantLike>() ||
         isa<hw::InstanceOp, seq::CompRegOp, arc::MemoryOp,
             arc::MemoryReadPortOp, arc::ClockedOpInterface, seq::InitialOp,
             seq::ClockGateOp, sim::DPICallOp>(op) ||
         op->getNumResults() > 1 || op->getNumRegions() > 0 ||
         !mlir::isMemoryEffectFree(op);
}

namespace {

/// The number of scalar leaves of a type under full recursive aggregate
/// decomposition. Arrays contribute elements times their element's leaves,
/// structs the sum of their fields' leaves, and any other type is one leaf
/// (integers are only decomposed in dedicated bitwise rounds).
static uint64_t leafCount(mlir::Type type) {
  if (auto arrayTy = dyn_cast<hw::ArrayType>(type))
    return arrayTy.getNumElements() * leafCount(arrayTy.getElementType());
  if (auto structTy = dyn_cast<hw::StructType>(type)) {
    uint64_t count = 0;
    for (auto field : structTy.getElements())
      count += leafCount(field.type);
    return count;
  }
  return 1;
}

/// Whether a type contains a union anywhere; unions are not decomposable.
static bool containsUnion(mlir::Type type) {
  if (isa<hw::UnionType>(type))
    return true;
  if (auto arrayTy = dyn_cast<hw::ArrayType>(type))
    return containsUnion(arrayTy.getElementType());
  if (auto structTy = dyn_cast<hw::StructType>(type))
    return llvm::any_of(structTy.getElements(),
                        [](auto field) { return containsUnion(field.type); });
  return false;
}

/// The type of leaf `idx` under full recursive decomposition. Leaves are
/// numbered element 0 first for arrays and in declaration order for structs.
static mlir::Type leafType(mlir::Type type, uint64_t idx) {
  if (auto arrayTy = dyn_cast<hw::ArrayType>(type)) {
    uint64_t perElement = leafCount(arrayTy.getElementType());
    return leafType(arrayTy.getElementType(), idx % perElement);
  }
  if (auto structTy = dyn_cast<hw::StructType>(type)) {
    for (auto field : structTy.getElements()) {
      uint64_t count = leafCount(field.type);
      if (idx < count)
        return leafType(field.type, idx);
      idx -= count;
    }
    llvm_unreachable("leaf index out of range");
  }
  return type;
}

struct SplitFalseCombLoopsPass
    : public arc::impl::SplitFalseCombLoopsBase<SplitFalseCombLoopsPass> {
  using SplitFalseCombLoopsBase::SplitFalseCombLoopsBase;

  void runOnOperation() override;
  bool runOneRound(mlir::Block &body);
  bool splitAggregateSCC(llvm::ArrayRef<mlir::Operation *> scc);
  bool splitBitwiseSCC(llvm::ArrayRef<mlir::Operation *> scc);
};

} // namespace

//===----------------------------------------------------------------------===//
// SCC discovery
//===----------------------------------------------------------------------===//

/// Compute SCCs of size > 1 (or with a self edge) among the non-breaker ops of
/// `body`. Edges run from defining op to user; operands defined by breaker ops
/// contribute no edge, mirroring ConvertToArcs' traversal.
static void findCombSCCs(
    mlir::Block &body,
    llvm::SmallVectorImpl<llvm::SmallVector<mlir::Operation *>> &sccs) {
  // Iterative Tarjan.
  struct NodeInfo {
    unsigned index = 0;
    unsigned lowlink = 0;
    bool onStack = false;
    bool visited = false;
  };
  SmallDenseMap<mlir::Operation *, NodeInfo> info;
  llvm::SmallVector<mlir::Operation *> stack;
  unsigned counter = 0;

  auto isNode = [](mlir::Operation *op) { return !isCombBreaker(op); };

  auto forEachSucc = [&](mlir::Operation *op, auto callback) {
    for (auto operand : op->getOperands()) {
      auto *def = operand.getDefiningOp();
      if (!def || def->getBlock() != op->getBlock() || !isNode(def))
        continue;
      callback(def);
    }
  };

  // DFS over "operand-defining op" edges; SCCs found are identical regardless
  // of edge direction.
  struct Frame {
    mlir::Operation *op;
    llvm::SmallVector<mlir::Operation *, 4> succs;
    unsigned nextSucc = 0;
  };

  for (auto &rootOp : body) {
    if (!isNode(&rootOp) || info[&rootOp].visited)
      continue;
    llvm::SmallVector<Frame> dfs;
    auto pushNode = [&](mlir::Operation *op) {
      auto &ni = info[op];
      ni.visited = true;
      ni.index = ni.lowlink = counter++;
      ni.onStack = true;
      stack.push_back(op);
      Frame frame;
      frame.op = op;
      forEachSucc(op,
                  [&](mlir::Operation *succ) { frame.succs.push_back(succ); });
      dfs.push_back(std::move(frame));
    };
    pushNode(&rootOp);
    while (!dfs.empty()) {
      auto &frame = dfs.back();
      if (frame.nextSucc < frame.succs.size()) {
        auto *succ = frame.succs[frame.nextSucc++];
        auto &si = info[succ];
        if (!si.visited) {
          pushNode(succ);
          continue;
        }
        if (si.onStack) {
          auto &ni = info[frame.op];
          ni.lowlink = std::min(ni.lowlink, si.index);
        }
        continue;
      }
      // Pop.
      auto *op = frame.op;
      auto &ni = info[op];
      dfs.pop_back();
      if (!dfs.empty()) {
        auto &pi = info[dfs.back().op];
        pi.lowlink = std::min(pi.lowlink, ni.lowlink);
      }
      if (ni.lowlink == ni.index) {
        llvm::SmallVector<mlir::Operation *> scc;
        while (true) {
          auto *member = stack.pop_back_val();
          info[member].onStack = false;
          scc.push_back(member);
          if (member == op)
            break;
        }
        if (scc.size() > 1) {
          sccs.push_back(std::move(scc));
        } else {
          // Keep single nodes only if they have a self edge (possible in the
          // graph region of an hw.module).
          auto *single = scc.front();
          bool selfEdge =
              llvm::any_of(single->getOperands(), [&](mlir::Value v) {
                return v.getDefiningOp() == single;
              });
          if (selfEdge)
            sccs.push_back(std::move(scc));
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Piece-level acyclicity pre-check
//===----------------------------------------------------------------------===//

namespace {
/// Checks whether the dependency graph of an SCC is acyclic at *piece*
/// granularity, i.e. after decomposing every member value into its array
/// elements, struct fields, or bits. Only then is the loop false and the
/// rewrite guaranteed to terminate with all backedges resolvable.
struct PieceGraph {
  /// Offset into the flat node space and arity per member value.
  llvm::SmallDenseMap<mlir::Value, std::pair<unsigned, unsigned>> members;
  unsigned numNodes = 0;
  /// Adjacency: for each piece node, the piece nodes it depends on.
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>> deps;

  /// Add a member value with the given decomposition arity.
  void addValue(mlir::Value v, unsigned arity) {
    members[v] = {numNodes, arity};
    numNodes += arity;
  }

  bool isMember(mlir::Value v) const { return members.count(v); }

  void allocate() { deps.resize(numNodes); }

  unsigned node(mlir::Value v, unsigned piece) const {
    return members.lookup(v).first + piece;
  }

  /// result piece `i` depends on piece `j` of member operand `x`.
  void addDep(mlir::Value result, unsigned i, mlir::Value x, unsigned j) {
    deps[node(result, i)].push_back(node(x, j));
  }

  /// result piece `i` depends on all pieces of operand `x` (no-op for
  /// non-member operands).
  void addDepAll(mlir::Value result, unsigned i, mlir::Value x) {
    auto it = members.find(x);
    if (it == members.end())
      return;
    auto [offset, arity] = it->second;
    for (unsigned j = 0; j < arity; ++j)
      deps[node(result, i)].push_back(offset + j);
  }

  /// Iterative three-color DFS cycle check.
  bool isAcyclic() const {
    enum : uint8_t { White, Gray, Black };
    llvm::SmallVector<uint8_t> color(numNodes, White);
    llvm::SmallVector<std::pair<unsigned, unsigned>> stack;
    for (unsigned root = 0; root < numNodes; ++root) {
      if (color[root] != White)
        continue;
      color[root] = Gray;
      stack.push_back({root, 0});
      while (!stack.empty()) {
        auto &[n, next] = stack.back();
        if (next < deps[n].size()) {
          unsigned succ = deps[n][next++];
          if (color[succ] == Gray)
            return false;
          if (color[succ] == White) {
            color[succ] = Gray;
            stack.push_back({succ, 0});
          }
          continue;
        }
        color[n] = Black;
        stack.pop_back();
      }
    }
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Aggregate splitting
//===----------------------------------------------------------------------===//

namespace {
/// Shared machinery for rewriting one SCC by decomposing every member value
/// one level.
struct SCCRewriter {
  llvm::SmallVector<mlir::Operation *> ops;
  llvm::SmallDenseSet<mlir::Operation *> opSet;
  mlir::OpBuilder builder;
  BackedgeBuilder backedges;
  /// Backedge placeholders for each SCC result value's pieces.
  SmallDenseMap<mlir::Value, llvm::SmallVector<Backedge>> pieces;
  /// Resolved piece values, filled as ops get rewritten.
  SmallDenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> resolved;
  /// Materialized whole values for SCC results.
  SmallDenseMap<mlir::Value, mlir::Value> wholes;

  SCCRewriter(llvm::ArrayRef<mlir::Operation *> scc, mlir::MLIRContext *ctx)
      : ops(scc.begin(), scc.end()), opSet(scc.begin(), scc.end()),
        builder(ctx), backedges(builder, scc.front()->getLoc()) {}

  bool isSCCValue(mlir::Value v) {
    auto *def = v.getDefiningOp();
    return def && opSet.contains(def);
  }

  void finalize() {
    // A piece may resolve directly to another piece's placeholder. Since the
    // stored `resolved` values are captured before any placeholder is
    // replaced, chase placeholder-to-placeholder chains to their final value
    // first (the piece-level acyclicity pre-check guarantees termination);
    // otherwise a placeholder replaced early would be re-introduced by a
    // later `setValue` through its stale handle.
    SmallDenseMap<mlir::Value, std::pair<mlir::Value, unsigned>> targets;
    for (auto &[value, edges] : pieces)
      for (auto [i, edge] : llvm::enumerate(edges))
        targets[(mlir::Value)edge] = {value, (unsigned)i};
    auto chase = [&](mlir::Value val) {
      while (true) {
        auto it = targets.find(val);
        if (it == targets.end())
          return val;
        val = resolved[it->second.first][it->second.second];
      }
    };
    // Resolve all placeholders.
    for (auto &[value, edges] : pieces) {
      auto &res = resolved[value];
      assert(res.size() == edges.size() && "unresolved pieces");
      for (auto [edge, val] : llvm::zip(edges, res))
        edge.setValue(chase(val));
    }
    // Detach the old (mutually referencing) ops from each other first so that
    // only genuinely external uses remain, then re-materialize wholes for
    // those and erase the old ops. A single-leaf value's whole may be a
    // cached placeholder captured before resolution; chase it as well, or the
    // replacement would re-introduce uses of the dead placeholder.
    for (auto *op : ops)
      op->dropAllReferences();
    for (auto *op : ops) {
      auto result = op->getResult(0);
      if (!result.use_empty())
        result.replaceAllUsesWith(chase(getWhole(result)));
    }
    for (auto *op : ops)
      op->erase();
  }

  virtual mlir::Value getWhole(mlir::Value v) = 0;
  virtual ~SCCRewriter() = default;
};

struct AggregateSCCRewriter : public SCCRewriter {
  using SCCRewriter::SCCRewriter;

  /// Cache for materialized leaves of non-member values.
  llvm::DenseMap<std::pair<mlir::Value, uint64_t>, mlir::Value> externalLeaves;

  /// Materialize an access chain to leaf `idx` of the (non-member) value `x`.
  mlir::Value materializeLeaf(mlir::Value x, uint64_t idx) {
    auto type = x.getType();
    if (auto arrayTy = dyn_cast<hw::ArrayType>(type)) {
      uint64_t perElement = leafCount(arrayTy.getElementType());
      unsigned idxWidth =
          std::max(1u, (unsigned)llvm::Log2_64_Ceil(arrayTy.getNumElements()));
      auto cst = hw::ConstantOp::create(
          builder, x.getLoc(), llvm::APInt(idxWidth, idx / perElement));
      auto element = hw::ArrayGetOp::create(builder, x.getLoc(), x, cst);
      return materializeLeaf(element, idx % perElement);
    }
    if (auto structTy = dyn_cast<hw::StructType>(type)) {
      for (auto [fieldIdx, field] : llvm::enumerate(structTy.getElements())) {
        uint64_t count = leafCount(field.type);
        if (idx < count) {
          auto fieldValue =
              hw::StructExtractOp::create(builder, x.getLoc(), field.type, x,
                                          builder.getI32IntegerAttr(fieldIdx));
          return materializeLeaf(fieldValue, idx);
        }
        idx -= count;
      }
      llvm_unreachable("leaf index out of range");
    }
    assert(idx == 0 && "non-aggregate values have a single leaf");
    return x;
  }

  /// Leaf `idx` of value `x`: a placeholder for member values, a materialized
  /// access chain otherwise.
  mlir::Value getPiece(mlir::Value x, uint64_t idx) {
    if (isSCCValue(x))
      return pieces[x][idx];
    auto [it, inserted] = externalLeaves.try_emplace({x, idx});
    if (inserted)
      it->second = materializeLeaf(x, idx);
    return it->second;
  }

  /// Recursively reassemble a value of `type` from its leaves.
  mlir::Value buildWhole(mlir::Type type, llvm::ArrayRef<mlir::Value> leaves,
                         mlir::Location loc) {
    if (auto arrayTy = dyn_cast<hw::ArrayType>(type)) {
      uint64_t perElement = leafCount(arrayTy.getElementType());
      // array_create takes elements MSB-first (operand 0 = highest index).
      llvm::SmallVector<mlir::Value> elements;
      for (uint64_t e = arrayTy.getNumElements(); e > 0; --e)
        elements.push_back(
            buildWhole(arrayTy.getElementType(),
                       leaves.slice((e - 1) * perElement, perElement), loc));
      return hw::ArrayCreateOp::create(builder, loc, elements);
    }
    if (auto structTy = dyn_cast<hw::StructType>(type)) {
      llvm::SmallVector<mlir::Value> fields;
      uint64_t offset = 0;
      for (auto field : structTy.getElements()) {
        uint64_t count = leafCount(field.type);
        fields.push_back(
            buildWhole(field.type, leaves.slice(offset, count), loc));
        offset += count;
      }
      return hw::StructCreateOp::create(builder, loc, structTy, fields);
    }
    assert(leaves.size() == 1 && "non-aggregate values have a single leaf");
    return leaves[0];
  }

  mlir::Value getWhole(mlir::Value v) override {
    if (!isSCCValue(v))
      return v;
    auto it = wholes.find(v);
    if (it != wholes.end())
      return it->second;
    auto &vp = pieces[v];
    llvm::SmallVector<mlir::Value> leaves(vp.begin(), vp.end());
    auto whole = buildWhole(v.getType(), leaves, v.getLoc());
    wholes[v] = whole;
    return whole;
  }

  /// Rewrite a single op, filling `resolved[result]`.
  void rewriteOp(mlir::Operation *op) {
    builder.setInsertionPoint(op);
    auto result = op->getResult(0);
    auto loc = op->getLoc();
    auto &res = resolved[result];
    uint64_t numLeaves = pieces[result].size();

    if (auto mux = dyn_cast<comb::MuxOp>(op)) {
      auto cond = getWhole(mux.getCond());
      for (uint64_t i = 0; i < numLeaves; ++i)
        res.push_back(comb::MuxOp::create(
            builder, loc, cond, getPiece(mux.getTrueValue(), i),
            getPiece(mux.getFalseValue(), i), mux.getTwoState()));
      return;
    }
    if (auto inject = dyn_cast<hw::ArrayInjectOp>(op)) {
      auto arrayTy = cast<hw::ArrayType>(inject.getType());
      uint64_t perElement = leafCount(arrayTy.getElementType());
      auto index = getWhole(inject.getIndex());
      llvm::APInt constIdx;
      bool isConst = mlir::matchPattern(index, mlir::m_ConstantInt(&constIdx));
      mlir::Value eq;
      for (uint64_t i = 0; i < numLeaves; ++i) {
        uint64_t element = i / perElement, sub = i % perElement;
        if (isConst) {
          res.push_back(constIdx == element ? getPiece(inject.getElement(), sub)
                                            : getPiece(inject.getInput(), i));
          continue;
        }
        if (sub == 0) {
          auto cst = hw::ConstantOp::create(
              builder, loc,
              llvm::APInt(cast<mlir::IntegerType>(index.getType()).getWidth(),
                          element));
          eq = comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq,
                                    index, cst);
        }
        res.push_back(comb::MuxOp::create(builder, loc, eq,
                                          getPiece(inject.getElement(), sub),
                                          getPiece(inject.getInput(), i),
                                          /*twoState=*/false));
      }
      return;
    }
    if (auto get = dyn_cast<hw::ArrayGetOp>(op)) {
      auto arrayTy = cast<hw::ArrayType>(get.getInput().getType());
      uint64_t perElement = leafCount(arrayTy.getElementType());
      auto index = getWhole(get.getIndex());
      llvm::APInt constIdx;
      if (mlir::matchPattern(index, mlir::m_ConstantInt(&constIdx))) {
        uint64_t base = constIdx.getZExtValue() * perElement;
        for (uint64_t i = 0; i < numLeaves; ++i)
          res.push_back(getPiece(get.getInput(), base + i));
        return;
      }
      // Dynamic read: reassemble the base (acyclic per the piece-level
      // pre-check) and read from it like before.
      auto whole =
          hw::ArrayGetOp::create(builder, loc, getWhole(get.getInput()), index);
      for (uint64_t i = 0; i < numLeaves; ++i)
        res.push_back(materializeLeaf(whole, i));
      return;
    }
    if (auto create = dyn_cast<hw::ArrayCreateOp>(op)) {
      // Operands are MSB-first: operand 0 is element N-1.
      auto inputs = create.getInputs();
      uint64_t perElement = numLeaves / inputs.size();
      for (uint64_t i = 0; i < numLeaves; ++i)
        res.push_back(getPiece(inputs[inputs.size() - 1 - i / perElement],
                               i % perElement));
      return;
    }
    if (auto create = dyn_cast<hw::StructCreateOp>(op)) {
      auto structTy = cast<hw::StructType>(create.getType());
      for (auto [fieldIdx, field] : llvm::enumerate(structTy.getElements())) {
        uint64_t count = leafCount(field.type);
        for (uint64_t s = 0; s < count; ++s)
          res.push_back(getPiece(create.getInput()[fieldIdx], s));
      }
      return;
    }
    if (auto extract = dyn_cast<hw::StructExtractOp>(op)) {
      auto structTy = cast<hw::StructType>(extract.getInput().getType());
      uint64_t offset = 0;
      for (unsigned f = 0; f < extract.getFieldIndex(); ++f)
        offset += leafCount(structTy.getElements()[f].type);
      for (uint64_t i = 0; i < numLeaves; ++i)
        res.push_back(getPiece(extract.getInput(), offset + i));
      return;
    }
    if (auto inject = dyn_cast<hw::StructInjectOp>(op)) {
      auto structTy = cast<hw::StructType>(inject.getType());
      uint64_t offset = 0;
      for (unsigned f = 0; f < inject.getFieldIndex(); ++f)
        offset += leafCount(structTy.getElements()[f].type);
      uint64_t count =
          leafCount(structTy.getElements()[inject.getFieldIndex()].type);
      for (uint64_t i = 0; i < numLeaves; ++i) {
        if (i >= offset && i < offset + count)
          res.push_back(getPiece(inject.getNewValue(), i - offset));
        else
          res.push_back(getPiece(inject.getInput(), i));
      }
      return;
    }
    llvm_unreachable("unexpected op in aggregate SCC");
  }
};
} // namespace

bool SplitFalseCombLoopsPass::splitAggregateSCC(
    llvm::ArrayRef<mlir::Operation *> scc) {
  // Every op must be decomposable and at least one member value must be an
  // aggregate for a split to make progress.
  bool anyAggregate = false;
  uint64_t totalPieces = 0;
  for (auto *op : scc) {
    if (!isa<comb::MuxOp, hw::ArrayInjectOp, hw::ArrayGetOp, hw::ArrayCreateOp,
             hw::StructCreateOp, hw::StructExtractOp, hw::StructInjectOp>(op))
      return false;
    auto type = op->getResult(0).getType();
    if (isa<hw::ArrayType, hw::StructType>(type))
      anyAggregate = true;
    if (containsUnion(type))
      return false;
    totalPieces += leafCount(type);
    if (totalPieces > (1ull << 20))
      return false;
  }
  if (!anyAggregate)
    return false;

  // Pre-check: the loop must actually be false at leaf granularity, i.e. the
  // leaf-level dependency graph must be acyclic. This rejects genuine loops
  // (e.g. an element read-modify-write feeding back into itself) for which
  // the rewrite could not resolve its placeholders.
  PieceGraph graph;
  for (auto *op : scc)
    graph.addValue(op->getResult(0), leafCount(op->getResult(0).getType()));
  graph.allocate();
  for (auto *op : scc) {
    auto result = op->getResult(0);
    uint64_t numLeaves = leafCount(result.getType());
    if (auto mux = dyn_cast<comb::MuxOp>(op)) {
      for (uint64_t i = 0; i < numLeaves; ++i) {
        graph.addDepAll(result, i, mux.getCond());
        if (graph.isMember(mux.getTrueValue()))
          graph.addDep(result, i, mux.getTrueValue(), i);
        if (graph.isMember(mux.getFalseValue()))
          graph.addDep(result, i, mux.getFalseValue(), i);
      }
    } else if (auto inject = dyn_cast<hw::ArrayInjectOp>(op)) {
      uint64_t perElement =
          leafCount(cast<hw::ArrayType>(inject.getType()).getElementType());
      llvm::APInt constIdx;
      bool isConst =
          mlir::matchPattern(inject.getIndex(), mlir::m_ConstantInt(&constIdx));
      for (uint64_t i = 0; i < numLeaves; ++i) {
        uint64_t element = i / perElement, sub = i % perElement;
        if (isConst) {
          if (constIdx == element) {
            if (graph.isMember(inject.getElement()))
              graph.addDep(result, i, inject.getElement(), sub);
          } else if (graph.isMember(inject.getInput())) {
            graph.addDep(result, i, inject.getInput(), i);
          }
          continue;
        }
        if (graph.isMember(inject.getElement()))
          graph.addDep(result, i, inject.getElement(), sub);
        if (graph.isMember(inject.getInput()))
          graph.addDep(result, i, inject.getInput(), i);
        graph.addDepAll(result, i, inject.getIndex());
      }
    } else if (auto get = dyn_cast<hw::ArrayGetOp>(op)) {
      uint64_t perElement = leafCount(
          cast<hw::ArrayType>(get.getInput().getType()).getElementType());
      llvm::APInt constIdx;
      bool isConst =
          mlir::matchPattern(get.getIndex(), mlir::m_ConstantInt(&constIdx));
      for (uint64_t i = 0; i < numLeaves; ++i) {
        if (isConst) {
          if (graph.isMember(get.getInput()))
            graph.addDep(result, i, get.getInput(),
                         constIdx.getZExtValue() * perElement + i);
        } else {
          graph.addDepAll(result, i, get.getInput());
          graph.addDepAll(result, i, get.getIndex());
        }
      }
    } else if (auto create = dyn_cast<hw::ArrayCreateOp>(op)) {
      auto inputs = create.getInputs();
      uint64_t perElement = numLeaves / inputs.size();
      for (uint64_t i = 0; i < numLeaves; ++i) {
        auto operand = inputs[inputs.size() - 1 - i / perElement];
        if (graph.isMember(operand))
          graph.addDep(result, i, operand, i % perElement);
      }
    } else if (auto create = dyn_cast<hw::StructCreateOp>(op)) {
      auto structTy = cast<hw::StructType>(create.getType());
      uint64_t i = 0;
      for (auto [fieldIdx, field] : llvm::enumerate(structTy.getElements())) {
        uint64_t count = leafCount(field.type);
        for (uint64_t s = 0; s < count; ++s, ++i)
          if (graph.isMember(create.getInput()[fieldIdx]))
            graph.addDep(result, i, create.getInput()[fieldIdx], s);
      }
    } else if (auto extract = dyn_cast<hw::StructExtractOp>(op)) {
      auto structTy = cast<hw::StructType>(extract.getInput().getType());
      uint64_t offset = 0;
      for (unsigned f = 0; f < extract.getFieldIndex(); ++f)
        offset += leafCount(structTy.getElements()[f].type);
      for (uint64_t i = 0; i < numLeaves; ++i)
        if (graph.isMember(extract.getInput()))
          graph.addDep(result, i, extract.getInput(), offset + i);
    } else if (auto inject = dyn_cast<hw::StructInjectOp>(op)) {
      auto structTy = cast<hw::StructType>(inject.getType());
      uint64_t offset = 0;
      for (unsigned f = 0; f < inject.getFieldIndex(); ++f)
        offset += leafCount(structTy.getElements()[f].type);
      uint64_t count =
          leafCount(structTy.getElements()[inject.getFieldIndex()].type);
      for (uint64_t i = 0; i < numLeaves; ++i) {
        if (i >= offset && i < offset + count) {
          if (graph.isMember(inject.getNewValue()))
            graph.addDep(result, i, inject.getNewValue(), i - offset);
        } else if (graph.isMember(inject.getInput())) {
          graph.addDep(result, i, inject.getInput(), i);
        }
      }
    }
  }
  if (!graph.isAcyclic())
    return false;

  AggregateSCCRewriter rewriter(scc, &getContext());
  // Allocate placeholders for every member value's leaves.
  for (auto *op : scc) {
    auto result = op->getResult(0);
    rewriter.builder.setInsertionPoint(op);
    auto &edges = rewriter.pieces[result];
    uint64_t numLeaves = leafCount(result.getType());
    for (uint64_t i = 0; i < numLeaves; ++i)
      edges.push_back(
          rewriter.backedges.get(leafType(result.getType(), i), op->getLoc()));
  }
  for (auto *op : scc)
    rewriter.rewriteOp(op);
  rewriter.finalize();
  ++numAggregateLoopsSplit;
  return true;
}

//===----------------------------------------------------------------------===//
// Bitwise splitting
//===----------------------------------------------------------------------===//

namespace {
struct BitwiseSCCRewriter : public SCCRewriter {
  using SCCRewriter::SCCRewriter;

  mlir::Value getPiece(mlir::Value x, unsigned idx) {
    if (isSCCValue(x))
      return pieces[x][idx];
    auto width = cast<mlir::IntegerType>(x.getType()).getWidth();
    if (width == 1) {
      assert(idx == 0 && "single bit");
      return x;
    }
    return comb::ExtractOp::create(builder, x.getLoc(), x, /*lowBit=*/idx,
                                   /*bitWidth=*/1);
  }

  mlir::Value getWhole(mlir::Value v) override {
    if (!isSCCValue(v))
      return v;
    auto it = wholes.find(v);
    if (it != wholes.end())
      return it->second;
    auto &vp = pieces[v];
    mlir::Value whole;
    if (vp.size() == 1) {
      whole = vp[0];
    } else {
      // concat takes bits MSB-first.
      llvm::SmallVector<mlir::Value> bits;
      for (unsigned i = vp.size(); i > 0; --i)
        bits.push_back(vp[i - 1]);
      whole = comb::ConcatOp::create(builder, v.getLoc(), bits);
    }
    wholes[v] = whole;
    return whole;
  }

  void rewriteOp(mlir::Operation *op) {
    builder.setInsertionPoint(op);
    auto result = op->getResult(0);
    auto loc = op->getLoc();
    auto &res = resolved[result];
    unsigned width = pieces[result].size();

    if (auto mux = dyn_cast<comb::MuxOp>(op)) {
      auto cond = getWhole(mux.getCond());
      for (unsigned i = 0; i < width; ++i)
        res.push_back(comb::MuxOp::create(
            builder, loc, cond, getPiece(mux.getTrueValue(), i),
            getPiece(mux.getFalseValue(), i), mux.getTwoState()));
      return;
    }
    if (auto extract = dyn_cast<comb::ExtractOp>(op)) {
      for (unsigned i = 0; i < width; ++i)
        res.push_back(getPiece(extract.getInput(), extract.getLowBit() + i));
      return;
    }
    if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
      // Operands are MSB-first.
      llvm::SmallVector<mlir::Value> bits(width);
      unsigned bit = 0;
      for (auto operand : llvm::reverse(concat.getInputs())) {
        unsigned w = cast<mlir::IntegerType>(operand.getType()).getWidth();
        for (unsigned i = 0; i < w; ++i)
          bits[bit++] = getPiece(operand, i);
      }
      res.append(bits.begin(), bits.end());
      return;
    }
    // Variadic bitwise ops.
    for (unsigned i = 0; i < width; ++i) {
      llvm::SmallVector<mlir::Value> operands;
      for (auto operand : op->getOperands())
        operands.push_back(getPiece(operand, i));
      mlir::Value bit;
      if (auto andOp = dyn_cast<comb::AndOp>(op))
        bit = comb::AndOp::create(builder, loc, operands, andOp.getTwoState());
      else if (auto orOp = dyn_cast<comb::OrOp>(op))
        bit = comb::OrOp::create(builder, loc, operands, orOp.getTwoState());
      else if (auto xorOp = dyn_cast<comb::XorOp>(op))
        bit = comb::XorOp::create(builder, loc, operands, xorOp.getTwoState());
      else
        llvm_unreachable("unexpected op in bitwise SCC");
      res.push_back(bit);
    }
  }
};
} // namespace

bool SplitFalseCombLoopsPass::splitBitwiseSCC(
    llvm::ArrayRef<mlir::Operation *> scc) {
  bool anyMultiBit = false;
  for (auto *op : scc) {
    if (!isa<comb::AndOp, comb::OrOp, comb::XorOp, comb::MuxOp, comb::ConcatOp,
             comb::ExtractOp>(op))
      return false;
    auto intTy = dyn_cast<mlir::IntegerType>(op->getResult(0).getType());
    if (!intTy || intTy.getWidth() == 0 || intTy.getWidth() > maxBitSplitWidth)
      return false;
    // All operands must be integers as well (mux condition is i1).
    for (auto operand : op->getOperands())
      if (!isa<mlir::IntegerType>(operand.getType()))
        return false;
    if (intTy.getWidth() > 1)
      anyMultiBit = true;
  }
  if (!anyMultiBit)
    return false;

  // Pre-check acyclicity at bit granularity (see splitAggregateSCC).
  auto width = [](mlir::Value v) {
    return cast<mlir::IntegerType>(v.getType()).getWidth();
  };
  PieceGraph graph;
  for (auto *op : scc)
    graph.addValue(op->getResult(0), width(op->getResult(0)));
  graph.allocate();
  for (auto *op : scc) {
    auto result = op->getResult(0);
    unsigned w = width(result);
    if (auto mux = dyn_cast<comb::MuxOp>(op)) {
      for (unsigned i = 0; i < w; ++i) {
        graph.addDepAll(result, i, mux.getCond());
        if (graph.isMember(mux.getTrueValue()))
          graph.addDep(result, i, mux.getTrueValue(), i);
        if (graph.isMember(mux.getFalseValue()))
          graph.addDep(result, i, mux.getFalseValue(), i);
      }
    } else if (auto extract = dyn_cast<comb::ExtractOp>(op)) {
      for (unsigned i = 0; i < w; ++i)
        if (graph.isMember(extract.getInput()))
          graph.addDep(result, i, extract.getInput(), extract.getLowBit() + i);
    } else if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
      unsigned bit = 0;
      for (auto operand : llvm::reverse(concat.getInputs())) {
        for (unsigned i = 0; i < width(operand); ++i, ++bit)
          if (graph.isMember(operand))
            graph.addDep(result, bit, operand, i);
      }
    } else {
      // Variadic bitwise ops.
      for (unsigned i = 0; i < w; ++i)
        for (auto operand : op->getOperands())
          if (graph.isMember(operand))
            graph.addDep(result, i, operand, i);
    }
  }
  if (!graph.isAcyclic())
    return false;

  BitwiseSCCRewriter rewriter(scc, &getContext());
  for (auto *op : scc) {
    auto result = op->getResult(0);
    rewriter.builder.setInsertionPoint(op);
    auto &edges = rewriter.pieces[result];
    unsigned width = cast<mlir::IntegerType>(result.getType()).getWidth();
    for (unsigned i = 0; i < width; ++i)
      edges.push_back(rewriter.backedges.get(rewriter.builder.getIntegerType(1),
                                             op->getLoc()));
  }
  for (auto *op : scc)
    rewriter.rewriteOp(op);
  rewriter.finalize();
  ++numBitwiseLoopsSplit;
  return true;
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

bool SplitFalseCombLoopsPass::runOneRound(mlir::Block &body) {
  llvm::SmallVector<llvm::SmallVector<mlir::Operation *>> sccs;
  findCombSCCs(body, sccs);
  bool changed = false;
  for (auto &scc : sccs) {
    if (splitAggregateSCC(scc)) {
      changed = true;
      continue;
    }
    if (splitBitwiseSCC(scc))
      changed = true;
  }
  return changed;
}

void SplitFalseCombLoopsPass::runOnOperation() {
  auto module = getOperation();
  auto *body = module.getBodyBlock();
  if (!body)
    return;
  // Each round decomposes the cycle-participating values by one type level;
  // the nesting depth of the involved types bounds the number of useful
  // rounds. The cap is a safety net.
  for (unsigned round = 0; round < 32; ++round)
    if (!runOneRound(*body))
      break;
}
