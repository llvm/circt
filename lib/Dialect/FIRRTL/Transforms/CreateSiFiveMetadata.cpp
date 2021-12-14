//===- CreateSiFiveMetadata.cpp - Create various metadata -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CreateSiFiveMetadata pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"

#define DEBUG_TYPE "create-sifive-metadata"

using namespace circt;
using namespace firrtl;

static const char dutAnnoClass[] = "sifive.enterprise.firrtl.MarkDUTAnnotation";
/// Attribute that indicates where some json files should be dumped.
static const char metadataDirectoryAnnoClass[] =
    "sifive.enterprise.firrtl.MetadataDirAnnotation";

static const char moduleReplacementAnnoClass[] =
    "sifive.enterprise.grandcentral.ModuleReplacementAnnotation";

namespace {
// A helper struct that keeps track of config file lines and their interpolated
// symbols and symbol indices. It stores the list of symbols that were
// interpolated in each line and uses those symbol names to sort and dedup the
// lines before emitting them as a verbatim op. This ensures that config file
// lines are deterministic. Note, this will dedup lines that have the same list
// of interpolated symbols, even if their contents might be different.
struct ConfigFileLines {
  // mapping of symbol name to interpolated line
  SmallVector<std::pair<SmallVector<StringRef>, SmallString<64>>> keyLinePairs;

  // the list of symbol names that have been interpolated in the current line
  SmallVector<StringRef> currKey;

  // the string value of the current interpolated line
  SmallString<64> currLine;
  void append(StringRef str) { currLine.append(str); }

  SmallDenseMap<Attribute, unsigned> symbolIndices;
  SmallVector<Attribute> symbols;
  void addSymbolImpl(Attribute symbol) {
    unsigned id;
    auto it = symbolIndices.find(symbol);
    if (it != symbolIndices.end()) {
      id = it->second;
    } else {
      id = symbols.size();
      symbols.push_back(symbol);
      symbolIndices.insert({symbol, id});
    }
    SmallString<8> str;
    ("{{" + Twine(id) + "}}").toVector(str);
    currLine.append(str);
  }
  // `instName` is used as the symbol key instead of the acual inner ref name,
  // since the inner ref name is not the actual instance name.
  void addSymbol(StringRef instName, hw::InnerRefAttr symbol) {
    currKey.push_back(instName);
    addSymbolImpl(symbol);
  }
  void addSymbol(FlatSymbolRefAttr symbol) {
    currKey.push_back(symbol.getValue());
    addSymbolImpl(symbol);
  }
  void addSymbol(Operation *op) {
    addSymbol(FlatSymbolRefAttr::get(SymbolTable::getSymbolName(op)));
  }

  void newLine() {
    keyLinePairs.push_back({std::move(currKey), std::move(currLine)});
    currKey.clear();
    currLine.clear();
  }

  // create a `VerbatimOp` with lines sorted and deduped
  sv::VerbatimOp verbatimOp(MLIRContext *context, OpBuilder &builder,
                            const mlir::Twine &filename) {
    // sort lines by their keys
    std::sort(keyLinePairs.begin(), keyLinePairs.end(),
              [](const std::pair<SmallVector<StringRef>, SmallString<64>> &a,
                 const std::pair<SmallVector<StringRef>, SmallString<64>> &b)
                  -> bool {
                for (auto [a, b] : llvm::zip(a.first, b.first)) {
                  if (a < b)
                    return true;
                  if (a > b)
                    return false;
                }
                return a.first.size() < b.first.size();
              });

    // dedup lines that have the same content
    keyLinePairs.erase(
        std::unique(
            keyLinePairs.begin(), keyLinePairs.end(),
            [](const std::pair<SmallVector<StringRef>, SmallString<64>> &a,
               const std::pair<SmallVector<StringRef>, SmallString<64>> &b)
                -> bool { return a.second == b.second; }),
        keyLinePairs.end());

    std::string buffer = "";
    llvm::raw_string_ostream os(buffer);
    for (auto [keys, line] : keyLinePairs) {
      os << line << '\n';
    }
    auto verbatimOp =
        builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), buffer);
    auto fileAttr = hw::OutputFileAttr::getFromFilename(
        context, filename, /*excludeFromFilelist=*/true);
    verbatimOp->setAttr("output_file", fileAttr);
    verbatimOp.symbolsAttr(ArrayAttr::get(context, symbols));
    return verbatimOp;
  }
};

struct ModuleReplacement {
  std::string circuitPackage;
  DenseSet<NonLocalAnchor> nlas;
  SmallVector<Operation *> ops;
};

class CreateSiFiveMetadataPass
    : public CreateSiFiveMetadataBase<CreateSiFiveMetadataPass> {
  LogicalResult emitRetimeModulesMetadata();
  LogicalResult emitSitestBlackboxMetadata();
  LogicalResult emitMemoryMetadata();
  /// writes VCS-specific runtime and compiletime config files for module
  /// swapping
  LogicalResult emitVCSConfigFiles();
  /// writes file lists for replaced modules
  LogicalResult emitVerilatorConfigFiles();
  void getDependentDialects(mlir::DialectRegistry &registry) const override;
  void runOnOperation() override;

  // The set of all modules underneath the design under test module.
  DenseSet<Operation *> dutModuleSet;
  // The design under test module.
  FModuleOp dutMod;
  /// mapping of annotation id to ModuleReplacement
  SmallDenseMap<IntegerAttr, ModuleReplacement> replacements;
  /// Returns an operation's `inner_sym`, adding one if necessary.
  StringAttr getOrAddInnerSym(Operation *op);
  /// Obtain an inner reference to an operation, possibly adding an `inner_sym`
  /// to that operation.
  hw::InnerRefAttr getInnerRefTo(Operation *op);
  SymbolTable *symtbl;
  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;
  /// Get the cached namespace for a module.
  ModuleNamespace &getModuleNamespace(FModuleLike module) {
    auto it = moduleNamespaces.find(module);
    if (it != moduleNamespaces.end())
      return it->second;
    return moduleNamespaces.insert({module, ModuleNamespace(module)})
        .first->second;
  }

public:
  CreateSiFiveMetadataPass(bool _replSeqMem, StringRef _replSeqMemCircuit,
                           StringRef _replSeqMemFile,
                           StringRef _moduleSwappingConfigOutputDir) {
    replSeqMem = _replSeqMem;
    replSeqMemCircuit = _replSeqMemCircuit.str();
    replSeqMemFile = _replSeqMemFile.str();
    moduleSwappingConfigOutputDir = _moduleSwappingConfigOutputDir.str();
  }
};
} // end anonymous namespace

/// This function collects all the firrtl.mem ops and creates a verbatim op with
/// the relevant memory attributes.
LogicalResult CreateSiFiveMetadataPass::emitMemoryMetadata() {
  if (!replSeqMem)
    return success();

  // Lambda to get the number of read, write and read-write ports corresponding
  // to a MemOp.

  CircuitOp circuitOp = getOperation();
  // The instance graph analysis will be required to print the hierarchy names
  // of the memory.
  auto instancePathCache = InstancePathCache(getAnalysis<InstanceGraph>());

  // This lambda, writes to the given Json stream all the relevant memory
  // attributes. Also adds the memory attrbutes to the string for creating the
  // memmory conf file.
  auto createMemMetadata = [&](SmallVector<MemOp> &memList,
                               llvm::json::OStream &jsonStream,
                               std::string &seqMemConfStr) {
    if (memList.empty())
      return;
    // All the MemOp in the memList refer to the same FIRRTL memory. So just get
    // the summary for the first MemoOp
    auto memSummary = (*memList.begin()).getSummary();
    // Get the memory data width.
    auto width = memSummary.dataWidth;
    // Metadata needs to be printed for memories which are candidates for
    // macro replacement. The requirements for macro replacement::
    // 1. read latency and write latency of one.
    // 2. only one readwrite port or write port.
    // 3. zero or one read port.
    // 4. undefined read-under-write behavior.
    if (!((memSummary.readLatency == 1 && memSummary.writeLatency == 1) &&
          (memSummary.numWritePorts + memSummary.numReadWritePorts == 1) &&
          (memSummary.numReadPorts <= 1) && width > 0))
      return;

    // Compute the mask granularity.
    auto maskGran = width / memSummary.maskBits;
    // Now create the config string for the memory.
    std::string portStr;
    if (memSummary.numWritePorts)
      portStr += "mwrite";
    if (memSummary.numReadPorts) {
      if (!portStr.empty())
        portStr += ",";
      portStr += "read";
    }
    if (memSummary.numReadWritePorts)
      portStr = "mrw";
    auto memExtName = memSummary.getFirMemoryName();
    seqMemConfStr += "name " + memExtName + " depth " +
                     std::to_string(memSummary.depth) + " width " +
                     std::to_string(width) + " ports " + portStr +
                     " mask_gran " + std::to_string(maskGran) + "\n";
    // This adds a Json array element entry corresponding to this memory.
    jsonStream.object([&] {
      jsonStream.attribute("module_name", memExtName);
      jsonStream.attribute("depth", (int64_t)memSummary.depth);
      jsonStream.attribute("width", (int64_t)width);
      jsonStream.attribute("masked", "true");
      jsonStream.attribute("read", memSummary.numReadPorts ? "true" : "false");
      jsonStream.attribute("write",
                           memSummary.numWritePorts ? "true" : "false");
      jsonStream.attribute("readwrite",
                           memSummary.numReadWritePorts ? "true" : "false");
      jsonStream.attribute("mask_granularity", (int64_t)maskGran);
      jsonStream.attributeArray("extra_ports", [&] {});
      // Record all the hierarchy names.
      SmallVector<std::string> hierNames;
      jsonStream.attributeArray("hierarchy", [&] {
        for (auto memOp : memList) {
          // Get the absolute path for the parent memory, to create the
          // hierarchy names.
          auto paths = instancePathCache.getAbsolutePaths(
              memOp->getParentOfType<FModuleOp>());
          for (auto p : paths) {
            if (p.empty())
              continue;
            const InstanceOp &x = p.front();
            std::string hierName =
                x->getParentOfType<FModuleOp>().getName().str();
            for (InstanceOp inst : p) {
              hierName = hierName + "." + inst.name().str();
            }
            hierName = hierName + "." + memOp.name().str();
            hierNames.push_back(hierName);
            jsonStream.value(hierName);
          }
        }
      });
    });
  };
  std::string testBenchJsonBuffer;
  llvm::raw_string_ostream testBenchOs(testBenchJsonBuffer);
  llvm::json::OStream testBenchJson(testBenchOs);
  std::string dutJsonBuffer;
  llvm::raw_string_ostream dutOs(dutJsonBuffer);
  llvm::json::OStream dutJson(dutOs);
  llvm::StringMap<SmallVector<MemOp>> dutMems;
  llvm::StringMap<SmallVector<MemOp>> tbMems;

  for (auto mod : circuitOp.getOps<FModuleOp>()) {
    bool isDut = dutModuleSet.contains(mod);
    for (auto memOp : mod.getBody()->getOps<MemOp>()) {
      auto firMem = memOp.getSummary();
      auto name = firMem.getFirMemoryName();
      if (isDut)
        dutMems[name].push_back(memOp);
      else
        tbMems[name].push_back(memOp);
    }
  }
  std::string seqMemConfStr;
  dutJson.array([&] {
    for (auto &dutM : dutMems)
      createMemMetadata(dutM.second, dutJson, seqMemConfStr);
  });
  testBenchJson.array([&] {
    // The tbConfStr is populated here, but unused, it will not be printed to
    // file.
    for (auto &tbM : tbMems)
      createMemMetadata(tbM.second, testBenchJson, seqMemConfStr);
  });

  auto *context = &getContext();
  auto builder = OpBuilder::atBlockEnd(circuitOp.getBody());
  AnnotationSet annos(circuitOp);
  auto dirAnno = annos.getAnnotation(metadataDirectoryAnnoClass);
  StringRef metadataDir = "metadata";
  if (dirAnno)
    if (auto dir = dirAnno.getMember<StringAttr>("dirname"))
      metadataDir = dir.getValue();

  if (testBenchJsonBuffer != "[]") {
    // Use unknown loc to avoid printing the location in the metadata files.
    auto tbVerbatimOp = builder.create<sv::VerbatimOp>(builder.getUnknownLoc(),
                                                       testBenchJsonBuffer);
    auto fileAttr = hw::OutputFileAttr::getFromDirectoryAndFilename(
        context, metadataDir, "tb_seq_mems.json", /*excludeFromFilelist=*/true);
    tbVerbatimOp->setAttr("output_file", fileAttr);
  }
  if (dutJsonBuffer != "[]") {
    auto dutVerbatimOp =
        builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), dutJsonBuffer);
    auto fileAttr = hw::OutputFileAttr::getFromDirectoryAndFilename(
        context, metadataDir, "seq_mems.json", /*excludeFromFilelist=*/true);
    dutVerbatimOp->setAttr("output_file", fileAttr);
  }
  if (!seqMemConfStr.empty()) {
    auto confVerbatimOp =
        builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), seqMemConfStr);
    if (replSeqMemFile.empty()) {
      circuitOp->emitError("metadata emission failed, the option "
                           "`-repl-seq-mem-file=<filename>` is mandatory for "
                           "specifying a valid seq mem metadata file");
      return failure();
    }

    auto fileAttr = hw::OutputFileAttr::getFromFilename(
        context, replSeqMemFile, /*excludeFromFilelist=*/true);
    confVerbatimOp->setAttr("output_file", fileAttr);
  }

  return success();
}
/// This will search for a target annotation and remove it from the operation.
/// If the annotation has a filename, it will be returned in the output
/// argument.  If the annotation is missing the filename member, or if more than
/// one matching annotation is attached, it will print an error and return
/// failure.
static LogicalResult removeAnnotationWithFilename(Operation *op,
                                                  StringRef annoClass,
                                                  StringRef &filename) {
  filename = "";
  bool error = false;
  AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
    // If there was a previous error or its not a match, continue.
    if (error || !anno.isClass(annoClass))
      return false;

    // If we have already found a matching annotation, error.
    if (!filename.empty()) {
      op->emitError("more than one ") << annoClass << " annotation attached";
      error = true;
      return false;
    }

    // Get the filename from the annotation.
    auto filenameAttr = anno.getMember<StringAttr>("filename");
    if (!filenameAttr) {
      op->emitError(annoClass) << " requires a filename";
      error = true;
      return false;
    }

    // Require a non-empty filename.
    filename = filenameAttr.getValue();
    if (filename.empty()) {
      op->emitError(annoClass) << " requires a non-empty filename";
      error = true;
      return false;
    }

    return true;
  });

  // If there was a problem above, return failure.
  return failure(error);
}

/// This function collects the name of each module annotated and prints them
/// all as a JSON array.
LogicalResult CreateSiFiveMetadataPass::emitRetimeModulesMetadata() {

  // Circuit level annotation.
  auto *retimeModulesAnnoClass =
      "sifive.enterprise.firrtl.RetimeModulesAnnotation";
  // Per module annotation.
  auto *retimeModuleAnnoClass =
      "sifive.enterprise.firrtl.RetimeModuleAnnotation";

  auto *context = &getContext();
  auto circuitOp = getOperation();

  // Get the filename, removing the annotation from the circuit.
  StringRef filename;
  if (failed(removeAnnotationWithFilename(circuitOp, retimeModulesAnnoClass,
                                          filename)))
    return failure();

  if (filename.empty())
    return success();

  // Create a string buffer for the json data.
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  llvm::json::OStream j(os);

  // The output is a json array with each element a module name.
  unsigned index = 0;
  SmallVector<Attribute> symbols;
  SmallString<3> placeholder;
  j.array([&] {
    for (auto module : circuitOp.getBody()->getOps<FModuleLike>()) {
      // The annotation has no supplemental information, just remove it.
      if (!AnnotationSet::removeAnnotations(module, retimeModuleAnnoClass))
        continue;

      // We use symbol substitution to make sure we output the correct thing
      // when the module goes through renaming.
      j.value(("{{" + Twine(index++) + "}}").str());
      symbols.push_back(SymbolRefAttr::get(context, module.moduleName()));
    }
  });

  // Put the retime information in a verbatim operation.
  auto builder = OpBuilder::atBlockEnd(circuitOp.getBody());
  auto verbatimOp = builder.create<sv::VerbatimOp>(
      circuitOp.getLoc(), buffer, ValueRange(), builder.getArrayAttr(symbols));
  auto fileAttr = hw::OutputFileAttr::getFromFilename(
      context, filename, /*excludeFromFilelist=*/true);
  verbatimOp->setAttr("output_file", fileAttr);
  return success();
}

/// This function finds all external modules which will need to be generated for
/// the test harness to run.
LogicalResult CreateSiFiveMetadataPass::emitSitestBlackboxMetadata() {
  auto *dutBlackboxAnnoClass =
      "sifive.enterprise.firrtl.SitestBlackBoxAnnotation";
  auto *testBlackboxAnnoClass =
      "sifive.enterprise.firrtl.SitestTestHarnessBlackBoxAnnotation";

  // Any extmodule with these annotations or one of these ScalaClass classes
  // should be excluded from the blackbox list.
  auto *scalaClassAnnoClass = "sifive.enterprise.firrtl.ScalaClassAnnotation";
  std::array<StringRef, 3> classBlackList = {
      "freechips.rocketchip.util.BlackBoxedROM", "chisel3.shim.CloneModule",
      "sifive.enterprise.grandcentral.MemTap"};
  std::array<StringRef, 6> blackListedAnnos = {
      "firrtl.transforms.BlackBox",
      "firrtl.transforms.BlackBoxInlineAnno",
      "firrtl.transforms.BlackBoxResourceAnno",
      "sifive.enterprise.grandcentral.DataTapsAnnotation",
      "sifive.enterprise.grandcentral.MemTapAnnotation",
      "sifive.enterprise.grandcentral.transforms.SignalMappingAnnotation"};

  auto *context = &getContext();
  auto circuitOp = getOperation();

  // Get the filenames from the annotations.
  StringRef dutFilename, testFilename;
  if (failed(removeAnnotationWithFilename(circuitOp, dutBlackboxAnnoClass,
                                          dutFilename)) ||
      failed(removeAnnotationWithFilename(circuitOp, testBlackboxAnnoClass,
                                          testFilename)))
    return failure();

  // If we don't have either annotation, no need to run this pass.
  if (dutFilename.empty() && testFilename.empty())
    return success();

  // Find all extmodules in the circuit. Check if they are black-listed from
  // being included in the list. If they are not, separate them into two groups
  // depending on if theyre in the DUT or the test harness.
  SmallVector<StringRef> dutModules;
  SmallVector<StringRef> testModules;
  for (auto extModule : circuitOp.getBody()->getOps<FExtModuleOp>()) {
    // If the module doesn't have a defname, then we can't record it properly.
    // Just skip it.
    if (!extModule.defname())
      continue;

    // If its a generated blackbox, skip it.
    AnnotationSet annos(extModule);
    if (llvm::any_of(blackListedAnnos, [&](auto blackListedAnno) {
          return annos.hasAnnotation(blackListedAnno);
        }))
      continue;

    // If its a blacklisted scala class, skip it.
    if (auto scalaAnno = annos.getAnnotation(scalaClassAnnoClass)) {
      auto scalaClass = scalaAnno.getMember<StringAttr>("className");
      if (scalaClass &&
          llvm::is_contained(classBlackList, scalaClass.getValue()))
        continue;
    }

    // Record the defname of the module.
    if (dutModuleSet.contains(extModule)) {
      dutModules.push_back(*extModule.defname());
    } else {
      testModules.push_back(*extModule.defname());
    }
  }

  // This is a helper to create the verbatim output operation.
  auto createOutput = [&](SmallVectorImpl<StringRef> &names,
                          StringRef filename) {
    if (filename.empty())
      return;

    // Sort and remove duplicates.
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());

    // The output is a json array with each element a module name. The
    // defname of a module can't change so we can output them verbatim.
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    llvm::json::OStream j(os);
    j.array([&] {
      for (auto &name : names)
        j.value(name);
    });

    auto *body = circuitOp.getBody();
    // Put the information in a verbatim operation.
    auto builder = OpBuilder::atBlockEnd(body);
    auto verbatimOp =
        builder.create<sv::VerbatimOp>(circuitOp.getLoc(), buffer);
    auto fileAttr = hw::OutputFileAttr::getFromFilename(
        context, filename, /*excludeFromFilelist=*/true);
    verbatimOp->setAttr("output_file", fileAttr);
  };

  createOutput(testModules, testFilename);
  createOutput(dutModules, dutFilename);
  return success();
}

LogicalResult CreateSiFiveMetadataPass::emitVerilatorConfigFiles() {
  if (replacements.empty())
    return success();

  MLIRContext *context = &getContext();
  CircuitOp circuitOp = getOperation();

  ConfigFileLines originalModules;

  auto builder = OpBuilder(circuitOp);
  for (auto [_, replacement] : replacements) {
    for (auto nla : replacement.nlas) {
      auto module = nla.modpath().getValue().back().cast<FlatSymbolRefAttr>();
      originalModules.addSymbol(module);
      originalModules.newLine();
    }
    for (auto *op : replacement.ops) {
      FModuleLike module = dyn_cast<FModuleLike>(op);
      assert(module);
      originalModules.addSymbol(op);
      originalModules.newLine();
    }
    auto subcircuitDir = (Twine(moduleSwappingConfigOutputDir) + "/" +
                          replacement.circuitPackage)
                             .str();

    // TODO: emit file list for modules in the replacement circuit when we are
    // able to read in multiple circuits
    originalModules.verbatimOp(context, builder,
                               subcircuitDir + "/original_modules.F");
  }
  return success();
}

LogicalResult CreateSiFiveMetadataPass::emitVCSConfigFiles() {
  if (replacements.empty())
    return success();

  MLIRContext *context = &getContext();
  CircuitOp circuitOp = getOperation();

  ConfigFileLines runtimeConfig;
  ConfigFileLines compileConfig;
  InstancePathCache instancePaths(getAnalysis<InstanceGraph>());
  auto builder = OpBuilder(circuitOp);
  for (auto [_, replacement] : replacements) {
    for (auto nla : replacement.nlas) {
      // nla instance path not including the path of the root module
      SmallVector<std::pair<StringRef, hw::InnerRefAttr>> tail;

      // build tail instance path
      hw::InnerRefAttr instRef;
      StringRef instName;
      for (auto modAndName :
           llvm::zip(nla.modpath().getValue(), nla.namepath().getValue())) {
        auto symAttr = std::get<0>(modAndName).cast<FlatSymbolRefAttr>();
        auto nameAttr = std::get<1>(modAndName).cast<StringAttr>();
        Operation *module = symtbl->lookup(symAttr.getValue());
        assert(module);
        if (instRef) {
          tail.push_back({instName, instRef});
        }

        // Find an instance with the given name in this module. Ensure it has a
        // symbol that we can refer to.
        instRef = {};
        instName = "";
        module->walk([&](InstanceOp instOp) {
          if (instOp.nameAttr() != nameAttr)
            return;
          LLVM_DEBUG(llvm::dbgs()
                     << "Marking NLA-participating instance " << nameAttr
                     << " in module " << symAttr << " as dont-touch\n");
          AnnotationSet::addDontTouch(instOp);
          instRef = getInnerRefTo(instOp);
          instName = instOp.name();
        });
      }

      // find all instances of the root module
      auto rootMod = nla.modpath().getValue().front().cast<FlatSymbolRefAttr>();
      Operation *module = symtbl->lookup(rootMod.getValue());
      assert(module);

      auto paths = instancePaths.getAbsolutePaths(module);
      if (paths.empty()) {
        nla->emitError("nla root targets uninstantiated component `")
            << rootMod.getValue() << "`";
        return failure();
      }

      // for each instance of the root module, construct an absolute path with
      // `tail`
      for (auto path : paths) {
        runtimeConfig.addSymbol(circuitOp.getMainModule());
        for (auto instance : path) {
          LLVM_DEBUG(llvm::dbgs() << "Marking NLA-participating instance "
                                  << instance.name() << " as dont-touch\n");
          AnnotationSet::addDontTouch(instance);
          runtimeConfig.append(".");
          runtimeConfig.addSymbol(instance.name(), getInnerRefTo(instance));
        }
        for (auto [name, symbol] : tail) {
          runtimeConfig.append(".");
          runtimeConfig.addSymbol(name, symbol);
        }
        runtimeConfig.newLine();
      }

      // add a replacement for the Leaf Module
      auto lastMod = nla.modpath().getValue().back().cast<FlatSymbolRefAttr>();
      compileConfig.append("replace module {");
      compileConfig.addSymbol(lastMod);
      compileConfig.append("} with module {");
      // TODO: use acutal module names once we support reading in multiple
      // circuits
      compileConfig.append(replacement.circuitPackage);
      compileConfig.append("}\n");
      compileConfig.newLine();
    }

    for (auto *op : replacement.ops) {
      FModuleLike module = dyn_cast<FModuleLike>(op);
      assert(module);
      runtimeConfig.addSymbol(module);
      runtimeConfig.newLine();

      compileConfig.append("replace module {");
      compileConfig.addSymbol(module);
      compileConfig.append("} with module {");
      // TODO: use acutal module names once we support reading in multiple
      // circuits
      compileConfig.append(replacement.circuitPackage);
      compileConfig.append("}\n");
      compileConfig.newLine();
    }

    auto subcircuitDir = (Twine(moduleSwappingConfigOutputDir) + "/" +
                          replacement.circuitPackage)
                             .str();
    runtimeConfig.verbatimOp(context, builder,
                             subcircuitDir + "/runtime_config.txt");
    compileConfig.verbatimOp(context, builder,
                             subcircuitDir + "/compile_config.txt");
  }
  return success();
}

StringAttr CreateSiFiveMetadataPass::getOrAddInnerSym(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>("inner_sym");
  if (attr)
    return attr;
  auto module = op->getParentOfType<FModuleOp>();
  auto name = getModuleNamespace(module).newName("gct_module_replacement_sym");
  attr = StringAttr::get(op->getContext(), name);
  op->setAttr("inner_sym", attr);
  return attr;
}

hw::InnerRefAttr CreateSiFiveMetadataPass::getInnerRefTo(Operation *op) {
  return hw::InnerRefAttr::get(
      SymbolTable::getSymbolName(op->getParentOfType<FModuleOp>()),
      getOrAddInnerSym(op));
}

void CreateSiFiveMetadataPass::getDependentDialects(
    mlir::DialectRegistry &registry) const {
  // We need this for SV verbatim and HW attributes.
  registry.insert<hw::HWDialect, sv::SVDialect>();
}

void CreateSiFiveMetadataPass::runOnOperation() {
  moduleNamespaces.clear();
  replacements.clear();
  symtbl = nullptr;
  auto circuitOp = getOperation();
  auto *body = circuitOp.getBody();
  SymbolTable currentSymtbl(circuitOp);
  symtbl = &currentSymtbl;

  // Find the device under test and create a set of all modules underneath it.
  auto it = llvm::find_if(*body, [&](Operation &op) -> bool {
    return AnnotationSet(&op).hasAnnotation(dutAnnoClass);
  });
  if (it != body->end()) {
    dutMod = dyn_cast<FModuleOp>(*it);
    auto instanceGraph = getAnalysis<InstanceGraph>();
    auto *node = instanceGraph.lookup(&(*it));
    llvm::for_each(llvm::depth_first(node), [&](InstanceGraphNode *node) {
      dutModuleSet.insert(node->getModule());
    });
  }

  // collect root module replacement annotations
  AnnotationSet::removeAnnotations(circuitOp, [&](Annotation anno) {
    if (anno.isClass(moduleReplacementAnnoClass)) {
      ModuleReplacement replacement;
      auto id = anno.getMember<IntegerAttr>("id");
      replacement.circuitPackage =
          anno.getMember<StringAttr>("circuitPackage").getValue();
      LLVM_DEBUG(llvm::dbgs() << "replacement id `" << anno.getDict() << "`\n");
      replacements.insert({id, std::move(replacement)});
      return true;
    }
    return false;
  });
  // find replaced modules/nlas and add them to their respective
  // ModuleReplacement struct
  circuitOp.walk([&](Operation *op) {
    AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
      auto id = anno.getMember<IntegerAttr>("id");
      if (!id)
        return false;

      auto it = replacements.find(id);
      if (it == replacements.end())
        return false;

      // if anno contains an nla then add the nla otherwise push the op
      auto nla = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal");
      if (!nla) {
        it->second.ops.push_back(op);
      } else {
        auto anchor =
            dyn_cast_or_null<NonLocalAnchor>(symtbl->lookup(nla.getAttr()));
        it->second.nlas.insert(anchor);
      }
      return true;
    });
  });
  if (moduleSwappingConfigOutputDir.empty())
    moduleSwappingConfigOutputDir = ".";

  if (failed(emitRetimeModulesMetadata()) ||
      failed(emitSitestBlackboxMetadata()) || failed(emitMemoryMetadata()) ||
      failed(emitVerilatorConfigFiles()) || failed(emitVCSConfigFiles()))
    return signalPassFailure();

  // This pass does not modify the hierarchy.
  markAnalysesPreserved<InstanceGraph>();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createCreateSiFiveMetadataPass(
    bool replSeqMem, StringRef replSeqMemCircuit, StringRef replSeqMemFile,
    StringRef moduleSwappingConfigOutputDir) {
  return std::make_unique<CreateSiFiveMetadataPass>(
      replSeqMem, replSeqMemCircuit, replSeqMemFile,
      moduleSwappingConfigOutputDir);
}
