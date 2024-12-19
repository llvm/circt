//===- circt-bmc.cpp - The circt-bmc bounded model checker ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-bmc' tool
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/HW//HWOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/Passes.h"

#include "circt/Support/Version.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace circt;
using namespace llvm;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

using STREAM_TYPE = mlir::raw_indented_ostream &;

static cl::OptionCategory mainCategory("circt-bmc Options");

static cl::opt<std::string>
    moduleName("module",
               cl::desc("Specify a named module to verify properties over."),
               cl::value_desc("module name"), cl::cat(mainCategory));

static cl::list<std::string> inputFilenames(cl::Positional, cl::OneOrMore,
                                            cl::desc("<input files>"),
                                            cl::cat(mainCategory));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

enum Kind { Module, Decl, Port };
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os, Kind kind) {
  switch (kind) {
  case Kind::Module:
    return os << "module";
  case Kind::Decl:
    return os << "decl";
  case Kind::Port:
    return os << "port";
  }
}

struct Diff {
  virtual void reportImpl(STREAM_TYPE &os) = 0;
  StringAttr name;
  LocationAttr loc;
  Kind kind;
  Diff(StringAttr name, LocationAttr loc, Kind kind)
      : name(name), loc(loc), kind(kind) {}
  virtual ~Diff() {}
  void report(STREAM_TYPE &os) {
    os << kind << " " << name << " @ " << loc << "\n";
    reportImpl(os);
    os << '\n';
  }
};

struct NonExist : Diff {
  size_t index;
  NonExist(StringAttr name, LocationAttr loc, Kind kind, size_t index)
      : Diff(name, loc, kind), index(index) {}
  void reportImpl(STREAM_TYPE &os) override {
    os << "not exists in input " << index;
  }
};

struct TypeMismatch : Diff {
  Type lhs, rhs;
  TypeMismatch(StringAttr name, LocationAttr loc, Kind kind, Type lhs, Type rhs)
      : Diff(name, loc, kind), lhs(lhs), rhs(rhs) {}
  void reportImpl(STREAM_TYPE &os) override {
    os << "type mismatch " << lhs << " vs " << rhs;
  }
  ~TypeMismatch() {}
};

struct OrderMismatch : Diff {
  void reportImpl(STREAM_TYPE &os) override { os << "order is not same"; }
};

struct HWModuleDiff : Diff {
  hw::HWModuleOp lhs, rhs;
  HWModuleDiff(hw::HWModuleOp lhs, hw::HWModuleOp rhs,
               SmallVector<std::unique_ptr<Diff>> ports)
      : Diff(lhs.getModuleNameAttr(), lhs->getLoc(), Kind::Module), lhs(lhs),
        rhs(rhs), ports(std::move(ports)) {}
  SmallVector<std::unique_ptr<Diff>> ports;
  SmallVector<std::unique_ptr<Diff>> decls;
  static const size_t LHS_ID = 0, RHS_ID = 1;

  void reportImpl(STREAM_TYPE &os) override {
    if (ports.empty() && decls.empty()) {
      os << "diff not found";
      return;
    }
    if (!ports.empty()) {
      auto scope = os.scope("", "");
      for (auto &port : ports) {
        os << '\n';
        port->report(os);
      }
      for (auto &decl : decls) {
        os << '\n';
        decl->report(os);
      }
    }
  }
  static std::unique_ptr<Diff> compare(hw::HWModuleOp lhs, hw::HWModuleOp rhs) {
    // Check ports.
    SmallVector<std::unique_ptr<Diff>> ports;
    SmallVector<std::unique_ptr<Diff>> decls;
    if (lhs.getNumOutputPorts() == 0 &&
        rhs.getNumOutputPorts() == 0) { // skip for now.
      return nullptr;
    }

    if (lhs.getModuleType() != rhs.getModuleType()) {
      // If a module type is different, there must be port difference.
      llvm::MapVector<StringAttr, hw::PortInfo> lhsPorts, rhsPorts;
      const auto &lhsPortLists = lhs.getPortList();
      const auto &rhsPortLists = rhs.getPortList();
      for (const auto &port : lhsPortLists)
        lhsPorts.insert(
            {StringAttr::get(lhs.getContext(), port.getVerilogName()), port});
      for (auto port : rhsPortLists)
        rhsPorts.insert(
            {StringAttr::get(rhs.getContext(), port.getVerilogName()), port});
      for (auto &[port, info] : lhsPorts) {
        auto it = rhsPorts.find(port);
        if (it == rhsPorts.end()) {
          ports.push_back(
              std::make_unique<NonExist>(port, info.loc, Kind::Port, RHS_ID));
          continue;
        }

        if (it->second.type != info.type) {
          ports.push_back(std::make_unique<TypeMismatch>(
              port, info.loc, Kind::Port, info.type, it->second.type));
          continue;
        }
      }
      for (auto &[port, info] : rhsPorts) {
        auto it = lhsPorts.find(port);
        if (it == rhsPorts.end()) {
          ports.push_back(
              std::make_unique<NonExist>(port, info.loc, Kind::Port, LHS_ID));
          continue;
        }

        if (it->second.type != info.type) {
          ports.push_back(std::make_unique<TypeMismatch>(
              port, info.loc, Kind::Port, info.type, it->second.type));
          continue;
        }
      }
    }

    if (ports.empty())
      return nullptr;

    return std::make_unique<HWModuleDiff>(lhs, rhs, std::move(ports));
  }
};

struct Comparer {
  SmallVector<std::unique_ptr<Diff>> diffs;
  void addDiff(std::unique_ptr<Diff> diff) { diffs.push_back(std::move(diff)); }
  bool empty() { return diffs.empty(); }
};

static void compare(mlir::ModuleOp module1, mlir::ModuleOp module2) {
  SymbolTable symTbl(module2);
  Comparer comparer;
  size_t numModules = 0;
  for (auto moduleOp : module1.getOps<hw::HWModuleOp>()) {
    auto op = dyn_cast_or_null<hw::HWModuleOp>(
        symTbl.lookup(moduleOp.getSymNameAttr()));
    if (!op){
      continue;
    }
    ++numModules;
    auto diff = HWModuleDiff::compare(moduleOp, op);
    if (diff)
      comparer.addDiff((std::move(diff)));
  }

  auto os = mlir::raw_indented_ostream(llvm::errs());
  if (comparer.empty()) {
    llvm::errs() << "inspected " << numModules
                 << " modules but diff was not found!\n";
    return;
  }
  for (auto &diff : comparer.diffs) {
    diff->report(os);
  }
}

/// This function initializes the various components of the tool and
/// orchestrates the work to be done.
static LogicalResult executeBMC(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  OwningOpRef<ModuleOp> module1, module2;
  {
    context.disableMultithreading();
    auto parserTimer = ts.nest("Parse MLIR input");
    // Parse the provided input files.
    module1 = parseSourceFile<ModuleOp>(inputFilenames[0], &context);
    if (!module1)
      return failure();

    module2 = parseSourceFile<ModuleOp>(inputFilenames[1], &context);
    if (!module2)
      return failure();

    context.enableMultithreading();
  }

  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  std::string errorMessage;
  // Create an output file.
  outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
  if (!outputFile.value()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

  if (verbosePassExecutions)
    pm.addInstrumentation(
        std::make_unique<VerbosePassInstrumentation<mlir::ModuleOp>>(
            "circt-bmc"));
  compare(module1.get(), module2.get());

  return success();
}

/// The entry point for the `circt-bmc` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeBMC` function to do the actual work.
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(
      argc, argv,
      "circt-bmc - bounded model checker\n\n"
      "\tThis tool checks all possible executions of a hardware module up "
      "to a "
      "given time bound to check whether any asserted properties can be "
      "violated.\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work
  // with.
  DialectRegistry registry;
  registry
      .insert<circt::comb::CombDialect, circt::hw::HWDialect,
              circt::seq::SeqDialect, mlir::BuiltinDialect,
              circt::sv::SVDialect, circt::om::OMDialect, emit::EmitDialect,
              ltl::LTLDialect, sim::SimDialect, circt::verif::VerifDialect>();

  MLIRContext context(registry);
  context.allowUnregisteredDialects(true);

  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  // Perform the logical equivalence checking; using `exit` to avoid the
  // slow teardown of the MLIR context.
  exit(failed(executeBMC(context)));
}
