//===- GrandCentralModuleSwapping.cpp ---------------------------*- C++ -*-===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the EmitOMIR pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gct-module-swapping"

static constexpr const char *moduleReplacementAnnoClass =
    "sifive.enterprise.grandcentral.ModuleReplacementAnnotation";

using namespace circt;
using namespace firrtl;

namespace {
struct ModuleReplacement {
  std::string circuitPackage;
  DenseSet<NonLocalAnchor> nlas;
  SmallVector<Operation *> ops;
};

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
} // namespace

class EmitModuleSwappingConfigPass
    : public EmitModuleSwappingConfigBase<EmitModuleSwappingConfigPass> {
public:
  using EmitModuleSwappingConfigBase::outputDirectory;

private:
  void runOnOperation() override;

  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;

  /// Get the cached namespace for a module.
  ModuleNamespace &getModuleNamespace(FModuleLike module) {
    auto it = moduleNamespaces.find(module);
    if (it != moduleNamespaces.end())
      return it->second;
    return moduleNamespaces.insert({module, ModuleNamespace(module)})
        .first->second;
  }

  /// Returns an operation's `inner_sym`, adding one if necessary.
  StringAttr getOrAddInnerSym(Operation *op);

  /// Obtain an inner reference to an operation, possibly adding an `inner_sym`
  /// to that operation.
  hw::InnerRefAttr getInnerRefTo(Operation *op);

  /// writes VCS-specific runtime and compiletime config files for module
  /// swapping
  void writeVCSConfigFiles();

  /// writes file lists for replaced modules
  void writeVerilatorConfigFiles();

  /// mapping of annotation id to ModuleReplacement
  SmallDenseMap<IntegerAttr, ModuleReplacement> replacements;

  SymbolTable *symtbl;
};

void EmitModuleSwappingConfigPass::runOnOperation() {
  moduleNamespaces.clear();
  replacements.clear();
  symtbl = nullptr;

  CircuitOp circuitOp = getOperation();
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

  SymbolTable currentSymtbl(circuitOp);
  symtbl = &currentSymtbl;
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

  if (replacements.empty()) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "skipping module swapping because no replacements were found");
    markAllAnalysesPreserved();
    return;
  }

  if (outputDirectory.empty())
    outputDirectory = ".";

  // TODO: add SimulatorAnnotation toggle to pick output format
  writeVerilatorConfigFiles();
  writeVCSConfigFiles();
}

void EmitModuleSwappingConfigPass::writeVerilatorConfigFiles() {
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
    auto subcircuitDir =
        (Twine(outputDirectory) + "/" + replacement.circuitPackage).str();

    // TODO: emit file list for modules in the replacement circuit when we are
    // able to read in multiple circuits
    originalModules.verbatimOp(context, builder,
                               subcircuitDir + "/original_modules.F");
  }
}

void EmitModuleSwappingConfigPass::writeVCSConfigFiles() {
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
        return;
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

    auto subcircuitDir =
        (Twine(outputDirectory) + "/" + replacement.circuitPackage).str();
    runtimeConfig.verbatimOp(context, builder,
                             subcircuitDir + "/runtime_config.txt");
    compileConfig.verbatimOp(context, builder,
                             subcircuitDir + "/compile_config.txt");
  }
}

StringAttr EmitModuleSwappingConfigPass::getOrAddInnerSym(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>("inner_sym");
  if (attr)
    return attr;
  auto module = op->getParentOfType<FModuleOp>();
  auto name = getModuleNamespace(module).newName("gct_module_replacement_sym");
  attr = StringAttr::get(op->getContext(), name);
  op->setAttr("inner_sym", attr);
  return attr;
}

hw::InnerRefAttr EmitModuleSwappingConfigPass::getInnerRefTo(Operation *op) {
  return hw::InnerRefAttr::get(
      SymbolTable::getSymbolName(op->getParentOfType<FModuleOp>()),
      getOrAddInnerSym(op));
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass>
circt::firrtl::createEmitModuleSwappingConfigPass(StringRef outputDirectory) {
  auto pass = std::make_unique<EmitModuleSwappingConfigPass>();
  if (!outputDirectory.empty())
    pass->outputDirectory = outputDirectory.str();
  return pass;
}
