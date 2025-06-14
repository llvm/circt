#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;
using namespace circt;
using namespace moore;

namespace {

struct ScalarAssignGroup {
  moore::ExtractRefOp extractRef;
  moore::ExtractOp extract;
  moore::ContinuousAssignOp assign;
  int index;
};

struct ValueComparator {
  bool operator()(mlir::Value lhs, mlir::Value rhs) const {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  }
};

using IndexedGroupMap = std::map<int, ScalarAssignGroup>;
using SourceGroupMap = std::map<mlir::Value, IndexedGroupMap, ValueComparator>;
using AssignTree = std::map<mlir::Value, SourceGroupMap, ValueComparator>;

void vectorizeGroup(std::vector<ScalarAssignGroup> &group) {
  if (group.empty())
    return;

  auto builder = OpBuilder(group.front().assign.getContext());
  builder.setInsertionPoint(group.front().assign);

  auto dst = group.front().extractRef.getOperand();
  auto src = group.front().extract.getOperand();

  builder.create<moore::ContinuousAssignOp>(group.front().assign.getLoc(), dst, src);

  for (auto &g : group) {
    g.assign.erase();
    g.extractRef.erase();
    g.extract.erase();
  }
}

struct SimpleVectorizationPass
    : public mlir::PassWrapper<SimpleVectorizationPass, mlir::OperationPass<mlir::ModuleOp>> {

  void runOnOperation() override {
    auto module = getOperation();
    AssignTree assignTree;

    module.walk([&](moore::ContinuousAssignOp assign) {
      auto lhs = assign.getDst();
      auto rhs = assign.getSrc();

      auto extractRef = dyn_cast_or_null<moore::ExtractRefOp>(lhs.getDefiningOp());
      auto extract = dyn_cast_or_null<moore::ExtractOp>(rhs.getDefiningOp());
      if (!extractRef || !extract)
        return;

      auto indexRefAttr = extractRef->getAttrOfType<mlir::IntegerAttr>("lowBit");
      auto indexAttr = extract->getAttrOfType<mlir::IntegerAttr>("lowBit");
      if (!indexRefAttr || !indexAttr)
        return;

      int index = indexRefAttr.getInt();
      if (index != indexAttr.getInt())
        return;

      assignTree[extractRef.getOperand()][extract.getOperand()][index] =
          {extractRef, extract, assign, index};
    });

    for (auto &[dst, srcMap] : assignTree) {
      for (auto &[src, indexMap] : srcMap) {
        std::vector<int> sortedIndices;
        for (const auto &[index, _] : indexMap)
          sortedIndices.push_back(index);
        std::sort(sortedIndices.begin(), sortedIndices.end());

        std::vector<ScalarAssignGroup> group;
        for (size_t i = 0; i < sortedIndices.size(); ++i) {
          if (!group.empty() && sortedIndices[i] != sortedIndices[i - 1] + 1) {
            if (group.size() > 1)
              vectorizeGroup(group);
            group.clear();
          }
          group.push_back(indexMap[sortedIndices[i]]);
        }
        if (group.size() > 1)
          vectorizeGroup(group);
      }
    }
  }

  StringRef getArgument() const override { return "simple-vec"; }

  StringRef getDescription() const override {
    return "Simple Vectorization Pass using tree structure";
  }
};

} 

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {
      MLIR_PLUGIN_API_VERSION,
      "SimpleVec",
      LLVM_VERSION_STRING,
      []() {
        PassPipelineRegistration<>(
            "simple-vec", "Simple Vectorization Pass",
            [](OpPassManager &pm) {
              pm.addPass(std::make_unique<SimpleVectorizationPass>());
            });
      }};
}

MLIR_DECLARE_EXPLICIT_TYPE_ID(SimpleVectorizationPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(SimpleVectorizationPass)
