#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

namespace {
// A test pass that simply replaces all wire names with foo_<n>
struct FooWiresPass : FooWiresBase<FooWiresPass> {
  void runOnOperation() override;
};
} // namespace

void FooWiresPass::runOnOperation() {
  size_t nWires = 0;                 // Counts the number of wires modified
  module.walk([&](hw::WireOp wire) { // Walk over every wire in the module
    wire.setName("foo_" + std::to_string(nWires++)); // Rename said wire
  });
}

std::unique_ptr<mlir::Pass> circt::hw::createFooWiresPass() {
  return std::make_unique<FooWiresPass>();
}
