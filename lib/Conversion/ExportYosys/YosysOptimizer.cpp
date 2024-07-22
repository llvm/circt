//===- LowerFirMem.cpp - Seq FIRRTL memory lowering -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translate Seq FirMem ops to instances of HW generated modules.
//
//===----------------------------------------------------------------------===//
#include "../PassDetail.h"
#include "RTLILConverterInternal.h"
#include "circt/Conversion/ExportRTLIL.h"
#include "circt/Dialect/Comb/CombVisitors.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqVisitor.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"

// Yosys headers.
#include "backends/rtlil/rtlil_backend.h"
#include "kernel/rtlil.h"
#include "kernel/yosys.h"

#define DEBUG_TYPE "export-yosys"

using namespace circt;
using namespace hw;
using namespace comb;
using namespace Yosys;
using namespace rtlil;

namespace {
#define GEN_PASS_DEF_EXPORTYOSYS
#define GEN_PASS_DEF_EXPORTYOSYSPARALLEL
#include "circt/Conversion/Passes.h.inc"

struct ExportYosysPass : public impl::ExportYosysBase<ExportYosysPass> {
  void runOnOperation() override;
};
struct ExportYosysParallelPass
    : public impl::ExportYosysParallelBase<ExportYosysParallelPass> {
  void runOnOperation() override;
};

} // namespace

void ExportYosysPass::runOnOperation() {
  init_yosys(redirectLog.getValue());
  auto &theInstanceGraph = getAnalysis<hw::InstanceGraph>();

  SmallVector<hw::HWModuleLike> modules(
      getOperation().getOps<hw::HWModuleLike>());
  auto theDesign =
      circt::rtlil::exportRTLILDesign(modules, {}, theInstanceGraph);

  auto *design = theDesign->get();

  if (!topModule.empty()) {
    std::string command = "hierarchy -top ";
    command += topModule;
    Yosys::run_pass(command, design);
  }

  for (auto pass : passes)
    Yosys::run_pass(pass, design);

  // Erase all ops before importing them.
  // TODO: We should preserve unrelated parts.
  while (getOperation().begin() != getOperation().end())
    getOperation().begin()->erase();

  auto result = circt::rtlil::importRTLILDesign(design, getOperation());
  if (failed(result))
    return signalPassFailure();
}

static llvm::DenseSet<mlir::StringAttr> designSet(InstanceGraph &instanceGraph,
                                                  StringAttr dut) {
  auto dutModule = instanceGraph.lookup(dut);
  if (!dutModule)
    return {};
  SmallVector<circt::igraph::InstanceGraphNode *, 8> worklist{dutModule};
  DenseSet<StringAttr> visited;
  while (!worklist.empty()) {
    auto *mod = worklist.pop_back_val();
    if (!mod)
      continue;
    if (!visited.insert(mod->getModule().getModuleNameAttr()).second)
      continue;
    for (auto inst : *mod) {
      if (!inst)
        continue;

      igraph::InstanceOpInterface t = inst->getInstance();
      assert(t);
      if (t->hasAttr("doNotPrint") || t->getNumResults() == 0)
        continue;
      worklist.push_back(inst->getTarget());
    }
  }
  return visited;
}

LogicalResult runYosys(Location loc, StringRef inputFilePath,
                       std::string command) {
  auto yosysPath = llvm::sys::findProgramByName("yosys");
  if (!yosysPath) {
    return mlir::emitError(loc) << "cannot find 'yosys' executable. Please add "
                                   "yosys to PATH. Error message='"
                                << yosysPath.getError().message() << "'";
  }
  StringRef commands[] = {"-q", "-p", command, "-f", "rtlil", inputFilePath};
  auto exitCode = llvm::sys::ExecuteAndWait(yosysPath.get(), commands);
  return success(exitCode == 0);
}

void ExportYosysParallelPass::runOnOperation() {
  // Set up yosys.
  init_yosys(false);
  auto &theInstanceGraph = getAnalysis<hw::InstanceGraph>();
  auto &table = getAnalysis<mlir::SymbolTable>();

  if (topModule.empty()) {
    getOperation().emitError() << "top module must be specified";
    return signalPassFailure();
  }
  auto dut = StringAttr::get(&getContext(), topModule.getValue());
  DenseSet<StringAttr> designs;
  if (table.lookup(dut))
    designs = designSet(theInstanceGraph, dut);
  auto isInDesign = [&](StringAttr mod) -> bool {
    if (designs.empty())
      return true;
    return designs.count(mod);
  };
  SmallVector<std::pair<hw::HWModuleOp, std::string>> results;
  for (auto op : getOperation().getOps<hw::HWModuleOp>()) {
    auto *node = theInstanceGraph.lookup(op.getModuleNameAttr());
    if (!isInDesign(op.getModuleNameAttr()))
      continue;

    SmallVector<hw::HWModuleLike> blackboxes;
    for (auto instance : *node) {
      auto mod = instance->getTarget()->getModule<hw::HWModuleLike>();
      blackboxes.push_back(mod);
    }

    auto theDesign =
        circt::rtlil::exportRTLILDesign({op}, blackboxes, theInstanceGraph);
    if (failed(theDesign))
      return signalPassFailure();

    auto *design = theDesign->get();

    std::error_code ec;
    SmallString<128> fileName;
    if ((ec = llvm::sys::fs::createTemporaryFile("yosys", "rtlil", fileName))) {
      op.emitError() << ec.message();
      return signalPassFailure();
    }

    results.emplace_back(op, fileName);

    {
      std::ofstream myfile(fileName.c_str());
      RTLIL_BACKEND::dump_design(myfile, design, false);
      myfile.close();
    }
  }
  llvm::sys::SmartMutex<true> mutex;
  auto logger = [&mutex](auto callback) {
    llvm::sys::SmartScopedLock<true> lock(mutex);
    callback();
  };
  std::string commands;
  bool first = true;
  for (auto i : passes) {
    if (first) {
      first = false;
    } else {
      commands += "; ";
    }
    commands += i;
  }
  if (failed(mlir::failableParallelForEachN(
          &getContext(), 0, results.size(), [&](auto i) {
            auto &[op, test] = results[i];
            logger([&] {
              llvm::errs() << "[yosys-optimizer] Running [" << i + 1 << "/"
                           << results.size() << "] " << op.getModuleName()
                           << "\n";
            });

            auto result = runYosys(op.getLoc(), test, commands);

            logger([&] {
              llvm::errs() << "[yosys-optimizer] Finished [" << i + 1 << "/"
                           << results.size() << "] " << op.getModuleName()
                           << "\n";
            });

            // Remove temporary rtlil if success.
            if (succeeded(result))
              llvm::sys::fs::remove(test);
            else
              op.emitError() << "Found error in yosys"
                             << "\n";

            return result;
          })))
    return signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> circt::createExportYosys() {
  return std::make_unique<ExportYosysPass>();
}

std::unique_ptr<mlir::Pass> circt::createExportYosysParallel() {
  return std::make_unique<ExportYosysParallelPass>();
}