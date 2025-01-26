#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Regex.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWTREESHAKE
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

struct HWTreeShakePass : circt::hw::impl::HWTreeShakeBase<HWTreeShakePass> {
  void runOnOperation() override;
};

void HWTreeShakePass::runOnOperation() {
  auto root = getOperation();
  llvm::DenseMap<mlir::StringRef, circt::hw::HWModuleLike> allModules;
  root.walk(
      [&](circt::hw::HWModuleLike mod) { allModules[mod.getName()] = mod; });

  llvm::DenseSet<circt::hw::HWModuleLike> visited;
  auto visit = [&allModules, &visited](auto &self,
                                       circt::hw::HWModuleLike mod) -> void {
    if (visited.contains(mod))
      return;
    visited.insert(mod);
    mod.walk([&](circt::hw::InstanceOp inst) {
      auto modName = inst.getModuleName();
      self(self, allModules.at(modName));
    });
  };

  for (const auto &kept : keep) {
    auto lookup = allModules.find(kept);
    if (lookup == allModules.end())
      continue; // Silently ignore missing modules
    visit(visit, lookup->getSecond());
  }

  for (auto &mod : allModules) {
    if (!visited.contains(mod.getSecond())) {
      mod.getSecond()->remove();
    }
  }
}

std::unique_ptr<mlir::Pass> circt::hw::createHWTreeShakePass() {
  return std::make_unique<HWTreeShakePass>();
}