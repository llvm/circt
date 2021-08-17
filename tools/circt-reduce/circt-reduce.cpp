//===- circt-reduce.cpp - The circt-reduce driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-reduce' tool, which is the circt analog of
// mlir-reduce, used to drive test case reduction.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/InitAllDialects.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Reducer/Tester.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

#define DEBUG_TYPE "circt-reduce"

using namespace llvm;
using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::Required);

static cl::opt<std::string>
    outputFilename("o", cl::init("-"),
                   cl::desc("Output filename for the reduced test case"));

static cl::opt<bool>
    keepBest("keep-best", cl::init(false),
             cl::desc("Keep overwriting the output with better reductions"));

static cl::opt<std::string> testerCommand(
    "test", cl::Required,
    cl::desc("A command or script to check if output is interesting"));

static cl::list<std::string>
    testerArgs("test-arg", cl::ZeroOrMore,
               cl::desc("Additional arguments to the test"));

//===----------------------------------------------------------------------===//
// Tool Implementation
//===----------------------------------------------------------------------===//

/// An abstract reduction pattern.
struct Reduction {
  virtual ~Reduction() {}

  /// Check if the reduction can apply to a specific operation.
  virtual bool match(Operation *op) const = 0;

  /// Apply the reduction to a specific operation. If the returned result
  /// indicates that the application failed, the resulting module is treated the
  /// same as if the tester marked it as uninteresting.
  virtual LogicalResult rewrite(Operation *op) const = 0;

  /// Return a human-readable name for this reduction pattern.
  virtual std::string getName() const = 0;

  /// Return true if the tool should accept the transformation this reduction
  /// performs on the module even if the overall size of the output increases.
  /// This can be handy for patterns that reduce the complexity of the IR at the
  /// cost of some verbosity.
  virtual bool acceptSizeIncrease() const { return false; }
};

/// A reduction pattern that applies an `mlir::Pass`.
struct PassReduction : public Reduction {
  PassReduction(MLIRContext *context, std::unique_ptr<Pass> pass,
                bool canIncreaseSize = false)
      : context(context), canIncreaseSize(canIncreaseSize) {
    passName = pass->getArgument();
    if (passName.empty())
      passName = pass->getName();

    if (auto opName = pass->getOpName())
      pm = std::make_unique<PassManager>(context, *opName);
    else
      pm = std::make_unique<PassManager>(context);
    pm->addPass(std::move(pass));
  }
  bool match(Operation *op) const override {
    return op->getName().getIdentifier() == pm->getOpName(*context);
  }
  LogicalResult rewrite(Operation *op) const override { return pm->run(op); }
  std::string getName() const override { return passName.str(); }
  bool acceptSizeIncrease() const override { return canIncreaseSize; }

protected:
  MLIRContext *const context;
  std::unique_ptr<PassManager> pm;
  StringRef passName;
  bool canIncreaseSize;
};

/// A sample reduction pattern that maps `firrtl.module` to `firrtl.extmodule`.
struct ModuleExternalizer : public Reduction {
  bool match(Operation *op) const override {
    return isa<firrtl::FModuleOp>(op);
  }
  LogicalResult rewrite(Operation *op) const override {
    auto module = cast<firrtl::FModuleOp>(op);
    OpBuilder builder(module);
    builder.create<firrtl::FExtModuleOp>(
        module->getLoc(),
        module->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()),
        firrtl::getModulePortInfo(module), StringRef(),
        module.annotationsAttr());
    module->erase();
    return success();
  }
  std::string getName() const override { return "module-externalizer"; }
};

/// Starting at the given `op`, traverse through it and its operands and erase
/// operations that have no more uses.
static void pruneUnusedOps(Operation *initialOp) {
  SmallVector<Operation *> worklist;
  worklist.push_back(initialOp);
  while (!worklist.empty()) {
    auto op = worklist.pop_back_val();
    if (!op->use_empty())
      continue;
    for (auto arg : op->getOperands())
      if (auto argOp = arg.getDefiningOp())
        worklist.push_back(argOp);
    op->erase();
  }
}

/// A sample reduction pattern that replaces the right-hand-side of
/// `firrtl.connect` and `firrtl.partialconnect` operations with a
/// `firrtl.invalidvalue`. This removes uses from the fanin cone to these
/// connects and creates opportunities for reduction in DCE/CSE.
struct ConnectInvalidator : public Reduction {
  bool match(Operation *op) const override {
    return isa<firrtl::ConnectOp, firrtl::PartialConnectOp>(op) &&
           !op->getOperand(1).getDefiningOp<firrtl::InvalidValueOp>();
  }
  LogicalResult rewrite(Operation *op) const override {
    assert(match(op));
    auto rhs = op->getOperand(1);
    OpBuilder builder(op);
    auto invOp =
        builder.create<firrtl::InvalidValueOp>(rhs.getLoc(), rhs.getType());
    op->setOperand(1, invOp);
    if (auto rhsOp = rhs.getDefiningOp())
      pruneUnusedOps(rhsOp);
    return success();
  }
  std::string getName() const override { return "connect-invalidator"; }
};

/// A sample reduction pattern that removes operations which either produce no
/// results or their results have no users.
struct OperationPruner : public Reduction {
  bool match(Operation *op) const override {
    return !isa<ModuleOp>(op) &&
           !op->hasAttr(SymbolTable::getSymbolAttrName()) &&
           (op->getNumResults() == 0 || op->use_empty());
  }
  LogicalResult rewrite(Operation *op) const override {
    assert(match(op));
    pruneUnusedOps(op);
    return success();
  }
  std::string getName() const override { return "operation-pruner"; }
};

/// A sample reduction pattern that replaces instances of `firrtl.extmodule`
/// with wires.
struct ExtmoduleInstanceRemover : public Reduction {
  bool match(Operation *op) const override {
    if (auto instOp = dyn_cast<firrtl::InstanceOp>(op))
      return isa<firrtl::FExtModuleOp>(instOp.getReferencedModule());
    return false;
  }
  LogicalResult rewrite(Operation *op) const override {
    auto instOp = cast<firrtl::InstanceOp>(op);
    auto portInfo = firrtl::getModulePortInfo(instOp.getReferencedModule());
    ImplicitLocOpBuilder builder(instOp.getLoc(), instOp);
    SmallVector<Value> replacementWires;
    for (firrtl::ModulePortInfo info : portInfo) {
      auto wire = builder.create<firrtl::WireOp>(
          info.type, (Twine(instOp.name()) + "_" + info.getName()).str());
      if (info.isOutput()) {
        auto inv = builder.create<firrtl::InvalidValueOp>(info.type);
        builder.create<firrtl::ConnectOp>(wire, inv);
      }
      replacementWires.push_back(wire);
    }
    instOp.replaceAllUsesWith(std::move(replacementWires));
    instOp->erase();
    return success();
  }
  std::string getName() const override { return "extmodule-instance-remover"; }
  bool acceptSizeIncrease() const override { return true; }
};

/// Helper function that writes the current MLIR module to the configured output
/// file. Called for intermediate states if the `keepBest` options has been set,
/// or at least at the very end of the run.
static LogicalResult writeOutput(ModuleOp module) {
  std::string errorMessage;
  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    mlir::emitError(UnknownLoc::get(module.getContext()),
                    "unable to open output file \"")
        << outputFilename << "\": " << errorMessage << "\n";
    return failure();
  }
  module.print(output->os());
  output->keep();
  return success();
}

static std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = false;
  return createCanonicalizerPass(config);
}

/// Execute the main chunk of work of the tool. This function reads the input
/// module and iteratively applies the reduction strategies until no options
/// make it smaller.
static LogicalResult execute(MLIRContext &context) {
  std::string errorMessage;

  // Parse the input file.
  LLVM_DEBUG(llvm::dbgs() << "Reading input\n");
  OwningModuleRef module = parseSourceFile(inputFilename, &context);
  if (!module)
    return failure();

  // Evaluate the unreduced input.
  LLVM_DEBUG(llvm::dbgs() << "Testing input\n");
  LLVM_DEBUG(llvm::dbgs() << "The test is `" << testerCommand << "`\n");
  for (auto &arg : testerArgs)
    LLVM_DEBUG(llvm::dbgs() << "  with argument `" << arg << "`\n");
  Tester tester(testerCommand, testerArgs);
  auto initialTest = tester.isInteresting(module.get());
  if (initialTest.first != Tester::Interestingness::True) {
    mlir::emitError(UnknownLoc::get(&context), "input is not interesting");
    return failure();
  }
  auto bestSize = initialTest.second;
  LLVM_DEBUG(llvm::dbgs() << "Initial module has size " << bestSize << "\n");

  // Gather a list of reduction patterns that we should try. Ideally these are
  // sorted by decreasing reduction potential/benefit. For example, things that
  // can knock out entire modules while being cheap should be tried first,
  // before trying to tweak operands of individual arithmetic ops.
  SmallVector<std::unique_ptr<Reduction>> patterns;
  patterns.push_back(std::make_unique<ModuleExternalizer>());
  patterns.push_back(
      std::make_unique<PassReduction>(&context, firrtl::createInlinerPass()));
  patterns.push_back(std::make_unique<PassReduction>(
      &context, createSimpleCanonicalizerPass()));
  patterns.push_back(
      std::make_unique<PassReduction>(&context, createCSEPass()));
  patterns.push_back(std::make_unique<ConnectInvalidator>());
  patterns.push_back(std::make_unique<OperationPruner>());
  patterns.push_back(std::make_unique<ExtmoduleInstanceRemover>());

  // Iteratively reduce the input module by applying the current reduction
  // pattern to successively smaller subsets of the operations until we find one
  // that retains the interesting behavior.
  // ModuleExternalizer pattern;
  for (unsigned patternIdx = 0; patternIdx < patterns.size();) {
    Reduction &pattern = *patterns[patternIdx];
    LLVM_DEBUG(llvm::dbgs()
               << "Trying reduction `" << pattern.getName() << "`\n");
    size_t rangeBase = 0;
    size_t rangeLength = -1;
    bool patternDidReduce = false;
    while (rangeLength > 0) {
      // Apply the pattern to the subset of operations selected by `rangeBase`
      // and `rangeLength`.
      size_t opIdx = 0;
      OwningModuleRef newModule = module->clone();
      newModule->walk([&](Operation *op) {
        if (!pattern.match(op))
          return;
        auto i = opIdx++;
        if (i < rangeBase || i - rangeBase >= rangeLength)
          return;
        (void)pattern.rewrite(op);
      });
      if (opIdx == 0) {
        LLVM_DEBUG(llvm::dbgs() << "- No more ops where the pattern applies\n");
        break;
      }

      // Check if this reduced module is still interesting, and its overall size
      // is smaller than what we had before.
      // LLVM_DEBUG(llvm::dbgs() << "Trying: " << **newModule << "\n");
      auto test = tester.isInteresting(newModule.get());
      if (test.first == Tester::Interestingness::True &&
          (test.second < bestSize || pattern.acceptSizeIncrease())) {
        // Make this reduced module the new baseline and reset our search
        // strategy to start again from the beginning, since this reduction may
        // have created additional opportunities.
        patternDidReduce = true;
        bestSize = test.second;
        LLVM_DEBUG(llvm::dbgs()
                   << "- Accepting module of size " << bestSize << "\n");
        module = std::move(newModule);

        // If this was already a run across all operations, no need to restart
        // again at the top. We're done at this point.
        if (rangeLength == (size_t)-1) {
          rangeLength = 0;
        } else {
          rangeBase = 0;
          rangeLength = -1;
        }

        // Write the current state to disk if the user asked for it.
        if (keepBest)
          if (failed(writeOutput(module.get())))
            return failure();
      } else {
        // Try the pattern on the next `rangeLength` number of operations. If we
        // go past the end of the input, reduce the size of the chunk of
        // operations we're reducing and start again from the top.
        rangeBase += rangeLength;
        if (rangeBase >= opIdx) {
          // Exhausted all subsets of this size. Try to go smaller.
          rangeLength = std::min(rangeLength, opIdx) / 2;
          rangeBase = 0;
          if (rangeLength > 0)
            LLVM_DEBUG(llvm::dbgs()
                       << "- Trying " << rangeLength << " ops at once\n");
        }
      }
    }

    // If the pattern provided a successful reduction, restart with the first
    // pattern again, since we might have uncovered additional reduction
    // opportunities. Otherwise we just keep going to try the next pattern.
    if (patternDidReduce && patternIdx > 0) {
      LLVM_DEBUG(llvm::dbgs() << "- Reduction `" << pattern.getName()
                              << "` was successful, starting at the top\n\n");
      patternIdx = 0;
    } else {
      ++patternIdx;
    }
  }

  // Write the reduced test case to the output.
  LLVM_DEBUG(llvm::dbgs() << "All reduction strategies exhausted\n");
  return writeOutput(module.get());
}

/// The entry point for the `circt-reduce` tool. Configures and parses the
/// command line options, registers all dialects with a context, and calls the
/// `execute` function to do the actual work.
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Parse the command line options provided by the user.
  registerMLIRContextCLOptions();
  registerAsmPrinterCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "CIRCT test case reduction tool\n");

  // Register all the dialects and create a context to work wtih.
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  mlir::MLIRContext context(registry);

  // Do the actual processing and use `exit` to avoid the slow teardown of the
  // context.
  exit(failed(execute(context)));
}
