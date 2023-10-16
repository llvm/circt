//===- Firtool.h - Definitions for the firtool pipeline setup ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This library parses options for firtool and sets up its pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_FIRTOOL_FIRTOOL_H
#define CIRCT_FIRTOOL_FIRTOOL_H

#include "circt/Conversion/Passes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Passes.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace circt {
namespace firtool {

enum class BuildMode { Debug, Release };

enum class RandomKind { None, Mem, Reg, All };

// Template parameters unified wrapper for `llvm::cl::opt`
template <class DataType, bool ExternalStorage = false>
class Opt : public llvm::cl::opt<DataType, ExternalStorage> {
private:
  using Super =
      llvm::cl::opt<DataType, ExternalStorage, llvm::cl::parser<DataType>>;

public:
  using typename Super::opt;

  bool isValueSet() const { return Super::getNumOccurrences() != 0; }
};

// Template parameters unified wrapper for `llvm::cl::opt_storage`
//
// `llvm::cl::opt` will be auto registered for Command Line, but in C-API we
// may construct these options multiple times, `opt` doesn't allow to be
// registered more than once, and since C-API doesn't need Command Line stuff,
// we just use the `opt_storage` for it.
template <class DataType, bool ExternalStorage = false>
class OptStorage : public llvm::cl::opt_storage<DataType, ExternalStorage,
                                                std::is_class_v<DataType>> {
private:
  using Super = llvm::cl::opt_storage<DataType, ExternalStorage,
                                      std::is_class_v<DataType>>;

private:
  template <class T>
  void applyMod(const T &mod) {
    // Ignore other command line modifiers, we only care `init` and `location`
    // for storage.
  }

  template <class T>
  void applyMod(const llvm::cl::initializer<T> &mod) {
    Super::setValue(mod.Init, true);
  }

  template <class T>
  void applyMod(const llvm::cl::list_initializer<T> &mod) {
    Super::setValue(mod.Inits, true);
  }

  template <class T>
  void applyMod(const llvm::cl::LocationClass<T> &mod) {
    // HACK: `setLocation` is the only way to set location without modifying
    //       LLVM, it requires a `Option` to just report an error
    static auto mockOption = llvm::cl::opt<bool>(
        "this-is-a-mock-option-to-workaround-opt-storage-set-location",
        llvm::cl::ReallyHidden);
    Super::setLocation(mockOption, mod.Loc);
  }

  template <class T, class... Mods>
  void applyMod(const T &mod, const Mods &...mods) {
    applyMod(mod);
    applyMod(mods...);
  }

public:
  template <class... Mods>
  explicit OptStorage(const Mods &...mods) {
    applyMod(mods...);
  }

  bool isValueSet() const { return isValueSet_; }

  template <class T>
  void setValue(const T &V, bool initial = false) {
    Super::setValue(V, initial);
    isValueSet_ = true;
  }

private:
  bool isValueSet_{false};
};

// Remember to sync changes to C-API
template <template <class, bool = false> class O>
struct FirtoolOptions {
  llvm::cl::OptionCategory &category;

  O<std::string> outputFilename{
      "o", llvm::cl::desc("Output filename, or directory for split output"),
      llvm::cl::value_desc("filename"), llvm::cl::init("-"),
      llvm::cl::cat(category)};

  O<bool> disableAnnotationsUnknown{
      "disable-annotation-unknown",
      llvm::cl::desc("Ignore unknown annotations when parsing"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<bool> disableAnnotationsClassless{
      "disable-annotation-classless",
      llvm::cl::desc("Ignore annotations without a class when parsing"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<bool> lowerAnnotationsNoRefTypePorts{
      "lower-annotations-no-ref-type-ports",
      llvm::cl::desc(
          "Create real ports instead of ref type ports when resolving "
          "wiring problems inside the LowerAnnotations pass"),
      llvm::cl::init(false), llvm::cl::Hidden, llvm::cl::cat(category)};

  O<circt::firrtl::PreserveAggregate::PreserveMode> preserveAggregate{
      "preserve-aggregate", llvm::cl::desc("Specify input file format:"),
      llvm::cl::values(clEnumValN(circt::firrtl::PreserveAggregate::None,
                                  "none", "Preserve no aggregate"),
                       clEnumValN(circt::firrtl::PreserveAggregate::OneDimVec,
                                  "1d-vec",
                                  "Preserve only 1d vectors of ground type"),
                       clEnumValN(circt::firrtl::PreserveAggregate::Vec, "vec",
                                  "Preserve only vectors"),
                       clEnumValN(circt::firrtl::PreserveAggregate::All, "all",
                                  "Preserve vectors and bundles")),
      llvm::cl::init(circt::firrtl::PreserveAggregate::None),
      llvm::cl::cat(category)};

  O<firrtl::PreserveValues::PreserveMode> preserveMode{
      "preserve-values",
      llvm::cl::desc("Specify the values which can be optimized away"),
      llvm::cl::values(
          clEnumValN(firrtl::PreserveValues::Strip, "strip",
                     "Strip all names. No name is preserved"),
          clEnumValN(firrtl::PreserveValues::None, "none",
                     "Names could be preserved by best-effort unlike `strip`"),
          clEnumValN(firrtl::PreserveValues::Named, "named",
                     "Preserve values with meaningful names"),
          clEnumValN(firrtl::PreserveValues::All, "all",
                     "Preserve all values")),
      llvm::cl::init(firrtl::PreserveValues::None), llvm::cl::cat(category)};

  O<BuildMode> buildMode{
      "O", llvm::cl::desc("Controls how much optimization should be performed"),
      llvm::cl::values(clEnumValN(BuildMode::Debug, "debug",
                                  "Compile with only necessary optimizations"),
                       clEnumValN(BuildMode::Release, "release",
                                  "Compile with optimizations")),
      llvm::cl::cat(category)};

  O<bool> disableOptimization{"disable-opt",
                              llvm::cl::desc("Disable optimizations"),
                              llvm::cl::cat(category)};

  O<bool> exportChiselInterface{
      "export-chisel-interface",
      llvm::cl::desc("Generate a Scala Chisel interface to the top level "
                     "module of the firrtl circuit"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<std::string> chiselInterfaceOutDirectory{
      "chisel-interface-out-dir",
      llvm::cl::desc(
          "The output directory for generated Chisel interface files"),
      llvm::cl::init(""), llvm::cl::cat(category)};

  O<bool> vbToBV{
      "vb-to-bv",
      llvm::cl::desc("Transform vectors of bundles to bundles of vectors"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<bool> dedup{"dedup",
                llvm::cl::desc("Deduplicate structurally identical modules"),
                llvm::cl::init(false), llvm::cl::cat(category)};

  O<bool> noDedup{
      "no-dedup",
      llvm::cl::desc("Disable deduplication of structurally identical modules"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<firrtl::CompanionMode> companionMode{
      "grand-central-companion-mode",
      llvm::cl::desc("Specifies the handling of Grand Central companions"),
      ::llvm::cl::values(
          clEnumValN(firrtl::CompanionMode::Bind, "bind",
                     "Lower companion instances to SystemVerilog binds"),
          clEnumValN(firrtl::CompanionMode::Instantiate, "instantiate",
                     "Instantiate companions in the design"),
          clEnumValN(firrtl::CompanionMode::Drop, "drop",
                     "Remove companions from the design")),
      llvm::cl::init(firrtl::CompanionMode::Bind),
      llvm::cl::Hidden,
      llvm::cl::cat(category)};

  O<bool> disableAggressiveMergeConnections{
      "disable-aggressive-merge-connections",
      llvm::cl::desc(
          "Disable aggressive merge connections (i.e. merge all field-level "
          "connections into bulk connections)"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<bool> disableHoistingHWPassthrough{
      "disable-hoisting-hw-passthrough",
      llvm::cl::desc("Disable hoisting HW passthrough signals"),
      llvm::cl::init(true), llvm::cl::Hidden, llvm::cl::cat(category)};

  O<bool> emitOMIR{"emit-omir",
                   llvm::cl::desc("Emit OMIR annotations to a JSON file"),
                   llvm::cl::init(true), llvm::cl::cat(category)};

  O<std::string> omirOutFile{"output-omir",
                             llvm::cl::desc("File name for the output omir"),
                             llvm::cl::init(""), llvm::cl::cat(category)};

  O<bool> lowerMemories{
      "lower-memories",
      llvm::cl::desc("Lower memories to have memories with masks as an "
                     "array with one memory per ground type"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<std::string> blackBoxRootPath{
      "blackbox-path",
      llvm::cl::desc(
          "Optional path to use as the root of black box annotations"),
      llvm::cl::value_desc("path"), llvm::cl::init(""),
      llvm::cl::cat(category)};

  O<bool> replSeqMem{
      "repl-seq-mem",
      llvm::cl::desc("Replace the seq mem for macro replacement and emit "
                     "relevant metadata"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<std::string> replSeqMemFile{
      "repl-seq-mem-file", llvm::cl::desc("File name for seq mem metadata"),
      llvm::cl::init(""), llvm::cl::cat(category)};

  O<bool> extractTestCode{"extract-test-code",
                          llvm::cl::desc("Run the extract test code pass"),
                          llvm::cl::init(false), llvm::cl::cat(category)};

  O<bool> ignoreReadEnableMem{
      "ignore-read-enable-mem",
      llvm::cl::desc("Ignore the read enable signal, instead of "
                     "assigning X on read disable"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<RandomKind> disableRandom{
      llvm::cl::desc(
          "Disable random initialization code (may break semantics!)"),
      llvm::cl::values(
          clEnumValN(RandomKind::Mem, "disable-mem-randomization",
                     "Disable emission of memory randomization code"),
          clEnumValN(RandomKind::Reg, "disable-reg-randomization",
                     "Disable emission of register randomization code"),
          clEnumValN(RandomKind::All, "disable-all-randomization",
                     "Disable emission of all randomization code")),
      llvm::cl::init(RandomKind::None), llvm::cl::cat(category)};

  O<std::string> outputAnnotationFilename{
      "output-annotation-file",
      llvm::cl::desc("Optional output annotation file"),
      llvm::cl::CommaSeparated, llvm::cl::value_desc("filename"),
      llvm::cl::cat(category)};

  O<bool> enableAnnotationWarning{
      "warn-on-unprocessed-annotations",
      llvm::cl::desc(
          "Warn about annotations that were not removed by lower-to-hw"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<bool> addMuxPragmas{
      "add-mux-pragmas",
      llvm::cl::desc("Annotate mux pragmas for memory array access"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<bool> emitChiselAssertsAsSVA{
      "emit-chisel-asserts-as-sva",
      llvm::cl::desc("Convert all chisel asserts into SVA"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<bool> emitSeparateAlwaysBlocks{
      "emit-separate-always-blocks",
      llvm::cl::desc(
          "Prevent always blocks from being merged and emit constructs into "
          "separate always blocks whenever possible"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<bool> etcDisableInstanceExtraction{
      "etc-disable-instance-extraction",
      llvm::cl::desc("Disable extracting instances only that feed test code"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<bool> etcDisableRegisterExtraction{
      "etc-disable-register-extraction",
      llvm::cl::desc("Disable extracting registers that only feed test code"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<bool> etcDisableModuleInlining{
      "etc-disable-module-inlining",
      llvm::cl::desc("Disable inlining modules that only feed test code"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<bool> addVivadoRAMAddressConflictSynthesisBugWorkaround{
      "add-vivado-ram-address-conflict-synthesis-bug-workaround",
      llvm::cl::desc(
          "Add a vivado specific SV attribute (* ram_style = "
          "\"distributed\" *) to unpacked array registers as a workaronud "
          "for a vivado synthesis bug that incorrectly modifies "
          "address conflict behavivor of combinational memories"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  //===----------------------------------------------------------------------===
  // External Clock Gate Options
  //===----------------------------------------------------------------------===

  seq::ExternalizeClockGateOptions clockGateOpts;

  O<std::string, true> ckgModuleName{
      "ckg-name", llvm::cl::desc("Clock gate module name"),
      llvm::cl::location(clockGateOpts.moduleName),
      llvm::cl::init("EICG_wrapper"), llvm::cl::cat(category)};

  O<std::string, true> ckgInputName{
      "ckg-input", llvm::cl::desc("Clock gate input port name"),
      llvm::cl::location(clockGateOpts.inputName), llvm::cl::init("in"),
      llvm::cl::cat(category)};

  O<std::string, true> ckgOutputName{
      "ckg-output", llvm::cl::desc("Clock gate output port name"),
      llvm::cl::location(clockGateOpts.outputName), llvm::cl::init("out"),
      llvm::cl::cat(category)};

  O<std::string, true> ckgEnableName{
      "ckg-enable", llvm::cl::desc("Clock gate enable port name"),
      llvm::cl::location(clockGateOpts.enableName), llvm::cl::init("en"),
      llvm::cl::cat(category)};

  O<std::string, true> ckgTestEnableName{
      "ckg-test-enable",
      llvm::cl::desc("Clock gate test enable port name (optional)"),
      llvm::cl::location(clockGateOpts.testEnableName),
      llvm::cl::init("test_en"), llvm::cl::cat(category)};

  O<bool> exportModuleHierarchy{
      "export-module-hierarchy",
      llvm::cl::desc("Export module and instance hierarchy as JSON"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  O<bool> stripFirDebugInfo{
      "strip-fir-debug-info",
      llvm::cl::desc(
          "Disable source fir locator information in output Verilog"),
      llvm::cl::init(true), llvm::cl::cat(category)};

  O<bool> stripDebugInfo{
      "strip-debug-info",
      llvm::cl::desc("Disable source locator information in output Verilog"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  inline bool isRandomEnabled(RandomKind kind) const {
    return disableRandom != RandomKind::All && disableRandom != kind;
  }

  inline firrtl::PreserveValues::PreserveMode getPreserveMode() const {
    if (!buildMode.isValueSet())
      return preserveMode;
    switch (buildMode) {
    case BuildMode::Debug:
      return firrtl::PreserveValues::Named;
    case BuildMode::Release:
      return firrtl::PreserveValues::None;
    }
    llvm_unreachable("unknown build mode");
  }

  inline FirtoolOptions(llvm::cl::OptionCategory &category)
      : category(category) {}
};

template <template <class, bool> class O>
LogicalResult populatePreprocessTransforms(mlir::PassManager &pm,
                                           const FirtoolOptions<O> &opt) {
  // Legalize away "open" aggregates to hw-only versions.
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerOpenAggsPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createResolvePathsPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerFIRRTLAnnotationsPass(
      opt.disableAnnotationsUnknown, opt.disableAnnotationsClassless,
      opt.lowerAnnotationsNoRefTypePorts));

  return success();
}

template <template <class, bool> class O>
LogicalResult
populateCHIRRTLToLowFIRRTL(mlir::PassManager &pm, const FirtoolOptions<O> &opt,
                           ModuleOp module, StringRef inputFilename) {
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerIntrinsicsPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInjectDUTHierarchyPass());

  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      firrtl::createDropNamesPass(opt.getPreserveMode()));

  if (!opt.disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        mlir::createCSEPass());

  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      firrtl::createLowerCHIRRTLPass());

  // Run LowerMatches before InferWidths, as the latter does not support the
  // match statement, but it does support what they lower to.
  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      firrtl::createLowerMatchesPass());

  // Width inference creates canonicalization opportunities.
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInferWidthsPass());

  pm.nest<firrtl::CircuitOp>().addPass(
      firrtl::createMemToRegOfVecPass(opt.replSeqMem, opt.ignoreReadEnableMem));

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInferResetsPass());

  if (opt.exportChiselInterface) {
    if (opt.chiselInterfaceOutDirectory.empty()) {
      pm.nest<firrtl::CircuitOp>().addPass(createExportChiselInterfacePass());
    } else {
      pm.nest<firrtl::CircuitOp>().addPass(createExportSplitChiselInterfacePass(
          opt.chiselInterfaceOutDirectory));
    }
  }

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createDropConstPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createHoistPassthroughPass(
      /*hoistHWDrivers=*/!opt.disableOptimization &&
      !opt.disableHoistingHWPassthrough));
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createProbeDCEPass());

  if (opt.dedup)
    emitWarning(UnknownLoc::get(pm.getContext()),
                "option -dedup is deprecated since firtool 1.57.0, has no "
                "effect (deduplication is always enabled), and will be removed "
                "in firtool 1.58.0");

  if (!opt.noDedup)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createDedupPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createWireDFTPass());

  if (opt.vbToBV) {
    pm.addNestedPass<firrtl::CircuitOp>(firrtl::createLowerFIRRTLTypesPass(
        firrtl::PreserveAggregate::All, firrtl::PreserveAggregate::All));
    pm.addNestedPass<firrtl::CircuitOp>(firrtl::createVBToBVPass());
  }

  if (!opt.lowerMemories)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createFlattenMemoryPass());

  // The input mlir file could be firrtl dialect so we might need to clean
  // things up.
  pm.addNestedPass<firrtl::CircuitOp>(firrtl::createLowerFIRRTLTypesPass(
      opt.preserveAggregate, firrtl::PreserveAggregate::None));

  pm.nest<firrtl::CircuitOp>().nestAny().addPass(
      firrtl::createExpandWhensPass());

  auto &modulePM = pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>();
  modulePM.addPass(firrtl::createSFCCompatPass());
  modulePM.addPass(firrtl::createGroupSinkPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerGroupsPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInlinerPass());

  // Preset the random initialization parameters for each module. The current
  // implementation assumes it can run at a time where every register is
  // currently in the final module it will be emitted in, all registers have
  // been created, and no registers have yet been removed.
  if (opt.isRandomEnabled(RandomKind::Reg))
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createRandomizeRegisterInitPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createCheckCombLoopsPass());

  // If we parsed a FIRRTL file and have optimizations enabled, clean it up.
  if (!opt.disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createSimpleCanonicalizerPass());

  // Run the infer-rw pass, which merges read and write ports of a memory with
  // mutually exclusive enables.
  if (!opt.disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createInferReadWritePass());

  if (opt.replSeqMem)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerMemoryPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createPrefixModulesPass());

  if (!opt.disableOptimization) {
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createIMConstPropPass());

    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createHoistPassthroughPass(
        /*hoistHWDrivers=*/!opt.disableOptimization &&
        !opt.disableHoistingHWPassthrough));
    // Cleanup after hoisting passthroughs, for separation-of-concerns.
    pm.addPass(firrtl::createIMDeadCodeElimPass());
  }

  pm.addNestedPass<firrtl::CircuitOp>(firrtl::createAddSeqMemPortsPass());

  pm.addPass(firrtl::createCreateSiFiveMetadataPass(opt.replSeqMem,
                                                    opt.replSeqMemFile));

  pm.addNestedPass<firrtl::CircuitOp>(firrtl::createExtractInstancesPass());

  // Run passes to resolve Grand Central features.  This should run before
  // BlackBoxReader because Grand Central needs to inform BlackBoxReader where
  // certain black boxes should be placed.  Note: all Grand Central Taps related
  // collateral is resolved entirely by LowerAnnotations.
  pm.addNestedPass<firrtl::CircuitOp>(
      firrtl::createGrandCentralPass(opt.companionMode));

  // Read black box source files into the IR.
  StringRef blackBoxRoot = opt.blackBoxRootPath.empty()
                               ? llvm::sys::path::parent_path(inputFilename)
                               : opt.blackBoxRootPath;
  pm.nest<firrtl::CircuitOp>().addPass(
      firrtl::createBlackBoxReaderPass(blackBoxRoot));

  // Run SymbolDCE as late as possible, but before InnerSymbolDCE. This is for
  // hierpathop's and just for general cleanup.
  pm.addNestedPass<firrtl::CircuitOp>(mlir::createSymbolDCEPass());

  // Run InnerSymbolDCE as late as possible, but before IMDCE.
  pm.addPass(firrtl::createInnerSymbolDCEPass());

  // The above passes, IMConstProp in particular, introduce additional
  // canonicalization opportunities that we should pick up here before we
  // proceed to output-specific pipelines.
  if (!opt.disableOptimization) {
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createSimpleCanonicalizerPass());
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        circt::firrtl::createRegisterOptimizerPass());
    // Re-run IMConstProp to propagate constants produced by register
    // optimizations.
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createIMConstPropPass());
    pm.addPass(firrtl::createIMDeadCodeElimPass());
  }

  if (opt.emitOMIR)
    pm.nest<firrtl::CircuitOp>().addPass(
        firrtl::createEmitOMIRPass(opt.omirOutFile));

  // Always run this, required for legalization.
  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      firrtl::createMergeConnectionsPass(
          !opt.disableAggressiveMergeConnections.getValue()));

  if (!opt.disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createVectorizationPass());

  return success();
}

template <template <class, bool> class O>
LogicalResult populateLowFIRRTLToHW(mlir::PassManager &pm,
                                    const FirtoolOptions<O> &opt) {
  // Remove TraceAnnotations and write their updated paths to an output
  // annotation file.
  if (opt.outputAnnotationFilename.empty())
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createResolveTracesPass());
  else
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createResolveTracesPass(
        opt.outputAnnotationFilename.getValue()));

  // Lower the ref.resolve and ref.send ops and remove the RefType ports.
  // LowerToHW cannot handle RefType so, this pass must be run to remove all
  // RefType ports and ops.
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerXMRPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerClassesPass());

  pm.addPass(createLowerFIRRTLToHWPass(opt.enableAnnotationWarning.getValue(),
                                       opt.emitChiselAssertsAsSVA.getValue(),
                                       !opt.isRandomEnabled(RandomKind::Mem),
                                       !opt.isRandomEnabled(RandomKind::Reg)));

  if (!opt.disableOptimization) {
    auto &modulePM = pm.nest<hw::HWModuleOp>();
    modulePM.addPass(mlir::createCSEPass());
    modulePM.addPass(createSimpleCanonicalizerPass());
  }

  // Check inner symbols and inner refs.
  pm.addPass(hw::createVerifyInnerRefNamespacePass());

  return success();
}

template <template <class, bool> class O>
LogicalResult populateHWToSV(mlir::PassManager &pm,
                             const FirtoolOptions<O> &opt) {
  if (opt.extractTestCode)
    pm.addPass(sv::createSVExtractTestCodePass(opt.etcDisableInstanceExtraction,
                                               opt.etcDisableRegisterExtraction,
                                               opt.etcDisableModuleInlining));

  pm.addPass(seq::createExternalizeClockGatePass(opt.clockGateOpts));
  pm.addPass(circt::createLowerSeqToSVPass(
      {/*disableRandomization=*/!opt.isRandomEnabled(RandomKind::Reg),
       /*emitSeparateAlwaysBlocks=*/
       opt.emitSeparateAlwaysBlocks}));
  pm.addNestedPass<hw::HWModuleOp>(createLowerVerifToSVPass());
  pm.addPass(sv::createHWMemSimImplPass(
      opt.replSeqMem, opt.ignoreReadEnableMem, opt.addMuxPragmas,
      !opt.isRandomEnabled(RandomKind::Mem),
      !opt.isRandomEnabled(RandomKind::Reg),
      opt.addVivadoRAMAddressConflictSynthesisBugWorkaround));

  // If enabled, run the optimizer.
  if (!opt.disableOptimization) {
    auto &modulePM = pm.nest<hw::HWModuleOp>();
    modulePM.addPass(mlir::createCSEPass());
    modulePM.addPass(createSimpleCanonicalizerPass());
    modulePM.addPass(mlir::createCSEPass());
    modulePM.addPass(sv::createHWCleanupPass(
        /*mergeAlwaysBlocks=*/!opt.emitSeparateAlwaysBlocks));
  }

  // Check inner symbols and inner refs.
  pm.addPass(hw::createVerifyInnerRefNamespacePass());

  return success();
}

namespace detail {
template <template <class, bool> class O>
LogicalResult
populatePrepareForExportVerilog(mlir::PassManager &pm,
                                const firtool::FirtoolOptions<O> &opt) {

  // Legalize unsupported operations within the modules.
  pm.nest<hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());

  // Tidy up the IR to improve verilog emission quality.
  if (!opt.disableOptimization)
    pm.nest<hw::HWModuleOp>().addPass(sv::createPrettifyVerilogPass());

  if (opt.stripFirDebugInfo)
    pm.addPass(circt::createStripDebugInfoWithPredPass([](mlir::Location loc) {
      if (auto fileLoc = loc.dyn_cast<FileLineColLoc>())
        return fileLoc.getFilename().getValue().endswith(".fir");
      return false;
    }));

  if (opt.stripDebugInfo)
    pm.addPass(circt::createStripDebugInfoWithPredPass(
        [](mlir::Location loc) { return true; }));

  // Emit module and testbench hierarchy JSON files.
  if (opt.exportModuleHierarchy)
    pm.addPass(sv::createHWExportModuleHierarchyPass(opt.outputFilename));

  // Check inner symbols and inner refs.
  pm.addPass(hw::createVerifyInnerRefNamespacePass());

  return success();
}
} // namespace detail

template <template <class, bool> class O>
LogicalResult populateExportVerilog(mlir::PassManager &pm,
                                    const FirtoolOptions<O> &opt,
                                    std::unique_ptr<llvm::raw_ostream> os) {
  pm.addPass(createExportVerilogPass(std::move(os)));

  return success();
}

template <template <class, bool> class O>
LogicalResult populateExportVerilog(mlir::PassManager &pm,
                                    const FirtoolOptions<O> &opt,
                                    llvm::raw_ostream &os) {
  if (failed(detail::populatePrepareForExportVerilog(pm, opt)))
    return failure();

  pm.addPass(createExportVerilogPass(os));
  return success();
}

template <template <class, bool> class O>
LogicalResult populateExportSplitVerilog(mlir::PassManager &pm,
                                         const FirtoolOptions<O> &opt,
                                         llvm::StringRef directory) {
  if (failed(detail::populatePrepareForExportVerilog(pm, opt)))
    return failure();

  pm.addPass(createExportSplitVerilogPass(directory));
  return success();
}

template <template <class, bool> class O>
LogicalResult populateFinalizeIR(mlir::PassManager &pm,
                                 const FirtoolOptions<O> &opt) {
  pm.addPass(firrtl::createFinalizeIRPass());
  pm.addPass(om::createFreezePathsPass());

  return success();
}

} // namespace firtool
} // namespace circt

#endif // CIRCT_FIRTOOL_FIRTOOL_H
