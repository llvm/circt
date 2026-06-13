#include "circt/Dialect/Resource/Passes/BRAMAnalysisAffine.h"
#include "circt/Dialect/Resource/Interfaces/MemoryInterface.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <map>
#include <string>


#define DEBUG_TYPE "hls-analysis-affine"

using namespace mlir;
using namespace mlir::affine;
using namespace hls;

namespace {

DenseMap<StringRef, hls::PartitionSpec>
getPartitionSpecs(Value memref) {
  DenseMap<StringRef, hls::PartitionSpec> specs;
  Operation *def = memref.getDefiningOp();
  if (!def)
    return specs;

  auto arr = def->getAttrOfType<ArrayAttr>("hls.array_partition");
  if (!arr)
    return specs;

  for (Attribute a : arr) {
    auto dict = llvm::dyn_cast<DictionaryAttr>(a);
    if (!dict)
      continue;

    auto dimAttr = dict.getAs<IntegerAttr>("dim");
    auto factorAttr = dict.getAs<IntegerAttr>("factor");
    auto kindAttr = dict.getAs<StringAttr>("kind");
    auto nameAttr = dict.getAs<StringAttr>("variable");
    if (!dimAttr || !kindAttr || !nameAttr) {
      llvm::outs() << "Array partition missing keys\n";
      continue;
    }
  

    unsigned hlsDim = dimAttr.getInt(); // 1-indexed
    if (hlsDim == 0)
      continue; // dim=0 means "all dims" in HLS; handle if you need it
    unsigned d = hlsDim - 1; // -> MLIR 0-indexed

    PartitionSpec s;
    StringRef kind = kindAttr.getValue();
    if (kind == "cyclic")
      s.kind = PartitionSpec::Cyclic;
    else if (kind == "block")
      s.kind = PartitionSpec::Block;
    else
      s.kind = PartitionSpec::Complete;
    // complete partitioning: every element its own bank; factor irrelevant.
    s.factor = factorAttr ? factorAttr.getInt() : 1;
    s.dim = d;
    specs[nameAttr] = s;
  }
  return specs;
}

std::optional<StringRef> resolveArrayName(const MemRefAccess &acc) {
  Operation *def = acc.memref.getDefiningOp();
  if (!def)
    return std::nullopt;

  // If the alloca is partitioned on exactly one variable, the access
  // unambiguously belongs to it. Multi-entry (struct) allocas need offset
  // resolution that isn't coded yet -> bail for now.
  auto arr = def->getAttrOfType<ArrayAttr>("hls.array_partition");
  if (arr && arr.size() == 1) {
    if (auto d = llvm::dyn_cast<DictionaryAttr>(arr[0]))
      if (auto v = d.getAs<StringAttr>("variable"))
        return v.getValue();
  }

  // Fallback: single-array alloca with a plain varname.
  if (auto vn = def->getAttrOfType<StringAttr>("polygeist.varname"))
    if (!arr || arr.size() <= 1)
      return vn.getValue();

  // Struct with multiple partitioned fields: offset resolution TODO.
  return std::nullopt;
}

bool sameBank(MemRefAccess &A, MemRefAccess &B,
              int64_t N, unsigned d,
              bool sameIteration, hls::PartitionSpec::Kind Kind) {
  assert(A.memref == B.memref && "caller already grouped by memref");
  if (N <= 1)
    return true; // unpartitioned -> single bank

  AffineValueMap amA, amB;
  A.getAccessMap(&amA);
  B.getAccessMap(&amB);

  unsigned numResultsA = amA.getNumResults();
  unsigned numResultsB = amB.getNumResults();
  llvm::errs() << "rank A=" << amA.getNumResults()
      << " rank B=" << amB.getNumResults()
      << " d=" << d << "\n";
  if (d >= numResultsA || d >= numResultsB) {
    llvm::errs() << "cyclicSameBank: dim " << d
        << " out of range (A has " << numResultsA
        << " results, B has " << numResultsB << ")\n";
    return true; // conservative
  }

  AffineExpr eA = amA.getResult(d);
  AffineExpr eB = amB.getResult(d);

  AffineExpr diff = simplifyAffineExpr(eA - eB, amA.getNumDims(),
                                       amA.getNumSymbols());
  // ---- Fast path: same iteration, indices differ by a constant ----
  // This is your i vs i+2 case: same-bank <=> N | (cB - cA). No solver.
  if (sameIteration && amA.getOperands() == amB.getOperands()) {
    llvm::outs() << "Simple access case\n";
    if (auto c = llvm::dyn_cast<AffineConstantExpr>(diff)) {
      return (c.getValue() % N) == 0; // Euclidean, sign-safe
    }
  }

  // ---- General path: integer emptiness check ----
  FlatAffineValueConstraints cst;

  // Loop-bound domains for both accesses, into one Value-aware system.
  // Shared IV Values merge by identity -> that *is* same-iteration coupling.
  SmallVector<AffineForOp, 4> loopsA, loopsB;
  getAffineForIVs(*A.opInst, &loopsA);
  getAffineForIVs(*B.opInst, &loopsB);

  SmallVector<AffineForOp, 8> allLoops(loopsA.begin(), loopsA.end());
  allLoops.append(loopsB.begin(), loopsB.end());

  SmallVector<Operation *, 8> allLoopOps;
  for (AffineForOp f : allLoops)
    allLoopOps.push_back(f.getOperation());
  if (failed(getIndexSet(allLoopOps, &cst)))
    return true;

  // Flatten each index expr to coefficients over cst's current columns.
  auto flatten = [&](AffineExpr e, AffineValueMap &am,
                     SmallVectorImpl<int64_t> &out) -> LogicalResult {
    // map operands of `am` to their column in `cst`, then flatten.
    // getFlattenedAffineExpr fills [dims..., symbols..., locals..., const].
    return getFlattenedAffineExpr(e, am.getNumDims(), am.getNumSymbols(),
                                  &out);
  };
  SmallVector<int64_t, 8> fA, fB;
  if (failed(flatten(eA, amA, fA)) || failed(flatten(eB, amB, fB)))
    return true;

  switch (Kind) {
  case hls::PartitionSpec::Kind::Cyclic: {
    unsigned rA = cst.addLocalModulo(fA, N); // rA column == fA mod N
    unsigned rB = cst.addLocalModulo(fB, N); // rB column == fB mod N
    SmallVector<int64_t, 8> eq(cst.getNumCols(), 0);
    eq[rA] = 1;
    eq[rB] = -1;
    cst.addEquality(eq); // bank_A == bank_B
    break;
  }
  case hls::PartitionSpec::Kind::Block: {
    unsigned rA = cst.addLocalFloorDiv(fA, N); // rA column == fA / N
    unsigned rB = cst.addLocalModulo(fB, N); // rB column == fB / N
    SmallVector<int64_t, 8> eq(cst.getNumCols(), 0);
    eq[rA] = 1;
    eq[rB] = -1;
    cst.addEquality(eq); // bank_A == bank_B
    break;
  }
  default:
    llvm_unreachable("Unexpected kind");
  }
  return !cst.isEmpty();
}

struct BRAMAnalysisAffinePass
    : PassWrapper<BRAMAnalysisAffinePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BRAMAnalysisAffinePass)

  StringRef getArgument() const final { return "bram-affine-analysis"; }

  StringRef getDescription() const final {
    return "Walks affine dialect machinery: loop nests, bounds, trip counts, "
        "memory accesses, integer sets";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, func::FuncDialect>();
  }

  // --------------------------------------------------------------===//
  // affine.load / affine.store: MemRefAccess + AffineValueMap
  //===--------------------------------------------------------------------===//
  void visitAccess(Operation *op) {
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp func) {
      llvm::outs() << "=== function @" << func.getName() << " ===\n";

      // block (contention region) -> arrayName (resource) -> accesses
      DenseMap<Block *, llvm::StringMap<SmallVector<MemRefAccess>>>
          byBlockByArray;

      func.walk([&](Operation *op) {
        if (!isa<AffineLoadOp, AffineStoreOp>(op))
          return;
        MemRefAccess access(op);
        if (!isa_and_nonnull<memref::AllocaOp>(access.memref.getDefiningOp()))
          return;
        auto name = resolveArrayName(access);
        if (!name)
          return; // can't attribute -> skip for now
        byBlockByArray[op->getBlock()][*name].push_back(access);
      });

      for (auto &[block, byArray] : byBlockByArray) {
        for (auto &[arrayName, accs] : byArray) {
          Value memref = accs.front().memref;
          auto specs = getPartitionSpecs(memref);
          auto it = specs.find(arrayName);

          // Not in the partition list -> unpartitioned -> single bank.
          if (it == specs.end()) {
            llvm::outs() << arrayName << ": unpartitioned (single bank)\n";
            continue;
          }
          const PartitionSpec &spec = it->second;

          for (unsigned i = 0; i < accs.size(); ++i)
            for (unsigned j = i + 1; j < accs.size(); ++j) {
              llvm::outs() << "Comparing in " << arrayName << "\n";
              accs[i].opInst->dumpPretty();
              accs[j].opInst->dumpPretty();
              if (spec.kind == PartitionSpec::Complete) {
                // every element its own bank: conflict iff same element.
                // offset != 0 within the same array -> different bank.
                llvm::outs() << arrayName << ": complete (per-element banks)\n";
                continue; // wire up element-level check later
              }
              PartitionSpec::Kind kind = spec.kind;
              bool same = sameBank(accs[i], accs[j], spec.factor, spec.dim,
                                   /*sameIteration=*/true, kind);
              llvm::outs() << arrayName << ": "
                  << (same ? "Same bank\n" : "Different bank\n");
            }
        }
      }
      llvm::outs() << "\n";
      markAllAnalysesPreserved();
    });
  }
};

} // namespace

std::unique_ptr<Pass> hls::createBRAMAffineAnalysis() {
  return std::make_unique<BRAMAnalysisAffinePass>();
}

void hls::registerBRAMAffineAnalysisPass() {
  PassRegistration<BRAMAnalysisAffinePass>();
}