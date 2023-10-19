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

#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace circt {
namespace firtool {

enum class RandomKind { None, Mem, Reg, All };

struct FirtoolGeneralOptions {
  bool disableOptimization{false};
  bool replSeqMem{false};
  std::string replSeqMemFile{""};
  bool ignoreReadEnableMem{false};
  RandomKind disableRandom{RandomKind::None};

  bool isRandomEnabled(RandomKind kind) const {
    return disableRandom != RandomKind::All && disableRandom != kind;
  }
};

struct FirtoolPreprocessTransformsOptions {
  FirtoolGeneralOptions *general;

  bool disableAnnotationsUnknown{false};
  bool disableAnnotationsClassless{false};
  bool lowerAnnotationsNoRefTypePorts{false};

  FirtoolPreprocessTransformsOptions(FirtoolGeneralOptions *g) : general{g} {}
};

struct FirtoolCHIRRTLToLowFIRRTLOptions {
  FirtoolGeneralOptions *general;

  firrtl::PreserveValues::PreserveMode preserveValues{
      firrtl::PreserveValues::None};
  circt::firrtl::PreserveAggregate::PreserveMode preserveAggregate{
      circt::firrtl::PreserveAggregate::None};
  bool exportChiselInterface{false};
  std::string chiselInterfaceOutDirectory{""};
  bool disableHoistingHWPassthrough{true};
  bool dedup{false};
  bool noDedup{false};
  bool vbToBV{false};
  bool lowerMemories{false};
  firrtl::CompanionMode companionMode{firrtl::CompanionMode::Bind};
  std::string blackBoxRootPath{""};
  bool emitOMIR{true};
  std::string omirOutFile{""};
  bool disableAggressiveMergeConnections{false};

  FirtoolCHIRRTLToLowFIRRTLOptions(FirtoolGeneralOptions *g) : general{g} {}
};

struct FirtoolLowFIRRTLToHWOptions {
  FirtoolGeneralOptions *general;

  std::string outputAnnotationFilename{""};
  bool enableAnnotationWarning{false};
  bool emitChiselAssertsAsSVA{false};

  FirtoolLowFIRRTLToHWOptions(FirtoolGeneralOptions *g) : general{g} {}
};

struct FirtoolHWToSVOptions {
  FirtoolGeneralOptions *general;

  bool extractTestCode{false};
  bool etcDisableInstanceExtraction{false};
  bool etcDisableRegisterExtraction{false};
  bool etcDisableModuleInlining{false};
  seq::ExternalizeClockGateOptions ckg;
  bool emitSeparateAlwaysBlocks{false};
  bool addMuxPragmas{false};
  bool addVivadoRAMAddressConflictSynthesisBugWorkaround{false};

  FirtoolHWToSVOptions(FirtoolGeneralOptions *g) : general{g} {}
};

struct FirtoolExportVerilogOptions {
  FirtoolGeneralOptions *general;

  bool stripFirDebugInfo{true};
  bool stripDebugInfo{false};
  bool exportModuleHierarchy{false};
  std::string outputPath{"-"};

  FirtoolExportVerilogOptions(FirtoolGeneralOptions *g) : general{g} {}
};

struct FirtoolFinalizeIROptions {
  FirtoolGeneralOptions *general;

  FirtoolFinalizeIROptions(FirtoolGeneralOptions *g) : general{g} {}
};

// Remember to sync changes to C-API
struct FirtoolOptions {
  llvm::cl::OptionCategory &category;

  // Build mode options.
  enum BuildMode { BuildModeDebug, BuildModeRelease };
  llvm::cl::opt<BuildMode> buildMode{
      "O", llvm::cl::desc("Controls how much optimization should be performed"),
      llvm::cl::values(clEnumValN(BuildModeDebug, "debug",
                                  "Compile with only necessary optimizations"),
                       clEnumValN(BuildModeRelease, "release",
                                  "Compile with optimizations")),
      llvm::cl::cat(category)};

  llvm::cl::opt<std::string> outputFilename{
      "o", llvm::cl::desc("Output filename, or directory for split output"),
      llvm::cl::value_desc("filename"), llvm::cl::init("-"),
      llvm::cl::cat(category)};

  // ========== General ==========

  FirtoolGeneralOptions generalOpts;

  llvm::cl::opt<bool, true> disableOptimization{
      "disable-opt", llvm::cl::desc("Disable optimizations"),
      llvm::cl::location(generalOpts.disableOptimization),
      llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> replSeqMem{
      "repl-seq-mem",
      llvm::cl::desc("Replace the seq mem for macro replacement and emit "
                     "relevant metadata"),
      llvm::cl::location(generalOpts.replSeqMem), llvm::cl::init(false),
      llvm::cl::cat(category)};

  llvm::cl::opt<std::string, true> replSeqMemFile{
      "repl-seq-mem-file", llvm::cl::desc("File name for seq mem metadata"),
      llvm::cl::location(generalOpts.replSeqMemFile), llvm::cl::init(""),
      llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> ignoreReadEnableMem{
      "ignore-read-enable-mem",
      llvm::cl::desc("Ignore the read enable signal, instead of "
                     "assigning X on read disable"),
      llvm::cl::location(generalOpts.ignoreReadEnableMem),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<RandomKind, true> disableRandom{
      llvm::cl::desc(
          "Disable random initialization code (may break semantics!)"),
      llvm::cl::values(
          clEnumValN(RandomKind::Mem, "disable-mem-randomization",
                     "Disable emission of memory randomization code"),
          clEnumValN(RandomKind::Reg, "disable-reg-randomization",
                     "Disable emission of register randomization code"),
          clEnumValN(RandomKind::All, "disable-all-randomization",
                     "Disable emission of all randomization code")),
      llvm::cl::location(generalOpts.disableRandom),
      llvm::cl::init(RandomKind::None), llvm::cl::cat(category)};

  // ========== PreprocessTransforms ==========

  FirtoolPreprocessTransformsOptions preprocessTransformsOpts{&generalOpts};

  llvm::cl::opt<bool, true> disableAnnotationsUnknown{
      "disable-annotation-unknown",
      llvm::cl::desc("Ignore unknown annotations when parsing"),
      llvm::cl::location(preprocessTransformsOpts.disableAnnotationsUnknown),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> disableAnnotationsClassless{
      "disable-annotation-classless",
      llvm::cl::desc("Ignore annotations without a class when parsing"),
      llvm::cl::location(preprocessTransformsOpts.disableAnnotationsClassless),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> lowerAnnotationsNoRefTypePorts{
      "lower-annotations-no-ref-type-ports",
      llvm::cl::desc(
          "Create real ports instead of ref type ports when resolving "
          "wiring problems inside the LowerAnnotations pass"),
      llvm::cl::location(
          preprocessTransformsOpts.lowerAnnotationsNoRefTypePorts),
      llvm::cl::init(false),
      llvm::cl::Hidden,
      llvm::cl::cat(category)};

  // ========== CHIRRTLToLowFIRRTL ==========

  mutable FirtoolCHIRRTLToLowFIRRTLOptions chirrtlToLowFIRRTLOpts{&generalOpts};

  llvm::cl::opt<firrtl::PreserveValues::PreserveMode, true> preserveValues{
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
      llvm::cl::location(chirrtlToLowFIRRTLOpts.preserveValues),
      llvm::cl::init(firrtl::PreserveValues::None),
      llvm::cl::cat(category)};

  llvm::cl::opt<circt::firrtl::PreserveAggregate::PreserveMode, true>
      preserveAggregate{
          "preserve-aggregate",
          llvm::cl::desc("Specify input file format:"),
          llvm::cl::values(
              clEnumValN(circt::firrtl::PreserveAggregate::None, "none",
                         "Preserve no aggregate"),
              clEnumValN(circt::firrtl::PreserveAggregate::OneDimVec, "1d-vec",
                         "Preserve only 1d vectors of ground type"),
              clEnumValN(circt::firrtl::PreserveAggregate::Vec, "vec",
                         "Preserve only vectors"),
              clEnumValN(circt::firrtl::PreserveAggregate::All, "all",
                         "Preserve vectors and bundles")),
          llvm::cl::location(chirrtlToLowFIRRTLOpts.preserveAggregate),
          llvm::cl::init(circt::firrtl::PreserveAggregate::None),
          llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> exportChiselInterface{
      "export-chisel-interface",
      llvm::cl::desc("Generate a Scala Chisel interface to the top level "
                     "module of the firrtl circuit"),
      llvm::cl::location(chirrtlToLowFIRRTLOpts.exportChiselInterface),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<std::string, true> chiselInterfaceOutDirectory{
      "chisel-interface-out-dir",
      llvm::cl::desc(
          "The output directory for generated Chisel interface files"),
      llvm::cl::location(chirrtlToLowFIRRTLOpts.chiselInterfaceOutDirectory),
      llvm::cl::init(""), llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> disableHoistingHWPassthrough{
      "disable-hoisting-hw-passthrough",
      llvm::cl::desc("Disable hoisting HW passthrough signals"),
      llvm::cl::location(chirrtlToLowFIRRTLOpts.disableHoistingHWPassthrough),
      llvm::cl::init(true),
      llvm::cl::Hidden,
      llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> dedup{
      "dedup", llvm::cl::desc("Deduplicate structurally identical modules"),
      llvm::cl::location(chirrtlToLowFIRRTLOpts.dedup), llvm::cl::init(false),
      llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> noDedup{
      "no-dedup",
      llvm::cl::desc("Disable deduplication of structurally identical modules"),
      llvm::cl::location(chirrtlToLowFIRRTLOpts.noDedup), llvm::cl::init(false),
      llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> vbToBV{
      "vb-to-bv",
      llvm::cl::desc("Transform vectors of bundles to bundles of vectors"),
      llvm::cl::location(chirrtlToLowFIRRTLOpts.vbToBV), llvm::cl::init(false),
      llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> lowerMemories{
      "lower-memories",
      llvm::cl::desc("Lower memories to have memories with masks as an "
                     "array with one memory per ground type"),
      llvm::cl::location(chirrtlToLowFIRRTLOpts.lowerMemories),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<firrtl::CompanionMode, true> companionMode{
      "grand-central-companion-mode",
      llvm::cl::desc("Specifies the handling of Grand Central companions"),
      ::llvm::cl::values(
          clEnumValN(firrtl::CompanionMode::Bind, "bind",
                     "Lower companion instances to SystemVerilog binds"),
          clEnumValN(firrtl::CompanionMode::Instantiate, "instantiate",
                     "Instantiate companions in the design"),
          clEnumValN(firrtl::CompanionMode::Drop, "drop",
                     "Remove companions from the design")),
      llvm::cl::location(chirrtlToLowFIRRTLOpts.companionMode),
      llvm::cl::init(firrtl::CompanionMode::Bind),
      llvm::cl::Hidden,
      llvm::cl::cat(category)};

  llvm::cl::opt<std::string, true> blackBoxRootPath{
      "blackbox-path",
      llvm::cl::desc(
          "Optional path to use as the root of black box annotations"),
      llvm::cl::value_desc("path"),
      llvm::cl::location(chirrtlToLowFIRRTLOpts.blackBoxRootPath),
      llvm::cl::init(""),
      llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> emitOMIR{
      "emit-omir", llvm::cl::desc("Emit OMIR annotations to a JSON file"),
      llvm::cl::location(chirrtlToLowFIRRTLOpts.emitOMIR), llvm::cl::init(true),
      llvm::cl::cat(category)};

  llvm::cl::opt<std::string, true> omirOutFile{
      "output-omir", llvm::cl::desc("File name for the output omir"),
      llvm::cl::location(chirrtlToLowFIRRTLOpts.omirOutFile),
      llvm::cl::init(""), llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> disableAggressiveMergeConnections{
      "disable-aggressive-merge-connections",
      llvm::cl::desc(
          "Disable aggressive merge connections (i.e. merge all field-level "
          "connections into bulk connections)"),
      llvm::cl::location(
          chirrtlToLowFIRRTLOpts.disableAggressiveMergeConnections),
      llvm::cl::init(false), llvm::cl::cat(category)};

  // ========== LowFIRRTLToHW ==========

  FirtoolLowFIRRTLToHWOptions lowFIRRTLToHWOpts{&generalOpts};

  llvm::cl::opt<std::string, true> outputAnnotationFilename{
      "output-annotation-file",
      llvm::cl::desc("Optional output annotation file"),
      llvm::cl::CommaSeparated,
      llvm::cl::value_desc("filename"),
      llvm::cl::location(lowFIRRTLToHWOpts.outputAnnotationFilename),
      llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> enableAnnotationWarning{
      "warn-on-unprocessed-annotations",
      llvm::cl::desc(
          "Warn about annotations that were not removed by lower-to-hw"),
      llvm::cl::location(lowFIRRTLToHWOpts.enableAnnotationWarning),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> emitChiselAssertsAsSVA{
      "emit-chisel-asserts-as-sva",
      llvm::cl::desc("Convert all chisel asserts into SVA"),
      llvm::cl::location(lowFIRRTLToHWOpts.emitChiselAssertsAsSVA),
      llvm::cl::init(false), llvm::cl::cat(category)};

  // ========== HWToSV ==========

  FirtoolHWToSVOptions hwToSVOpts{&generalOpts};

  llvm::cl::opt<bool, true> extractTestCode{
      "extract-test-code", llvm::cl::desc("Run the extract test code pass"),
      llvm::cl::location(hwToSVOpts.extractTestCode), llvm::cl::init(false),
      llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> etcDisableInstanceExtraction{
      "etc-disable-instance-extraction",
      llvm::cl::desc("Disable extracting instances only that feed test code"),
      llvm::cl::location(hwToSVOpts.etcDisableInstanceExtraction),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> etcDisableRegisterExtraction{
      "etc-disable-register-extraction",
      llvm::cl::desc("Disable extracting registers that only feed test code"),
      llvm::cl::location(hwToSVOpts.etcDisableRegisterExtraction),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> etcDisableModuleInlining{
      "etc-disable-module-inlining",
      llvm::cl::desc("Disable inlining modules that only feed test code"),
      llvm::cl::location(hwToSVOpts.etcDisableModuleInlining),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<std::string, true> ckgModuleName{
      "ckg-name", llvm::cl::desc("Clock gate module name"),
      llvm::cl::location(hwToSVOpts.ckg.moduleName),
      llvm::cl::init("EICG_wrapper"), llvm::cl::cat(category)};

  llvm::cl::opt<std::string, true> ckgInputName{
      "ckg-input", llvm::cl::desc("Clock gate input port name"),
      llvm::cl::location(hwToSVOpts.ckg.inputName), llvm::cl::init("in"),
      llvm::cl::cat(category)};

  llvm::cl::opt<std::string, true> ckgOutputName{
      "ckg-output", llvm::cl::desc("Clock gate output port name"),
      llvm::cl::location(hwToSVOpts.ckg.outputName), llvm::cl::init("out"),
      llvm::cl::cat(category)};

  llvm::cl::opt<std::string, true> ckgEnableName{
      "ckg-enable", llvm::cl::desc("Clock gate enable port name"),
      llvm::cl::location(hwToSVOpts.ckg.enableName), llvm::cl::init("en"),
      llvm::cl::cat(category)};

  llvm::cl::opt<std::string, true> ckgTestEnableName{
      "ckg-test-enable",
      llvm::cl::desc("Clock gate test enable port name (optional)"),
      llvm::cl::location(hwToSVOpts.ckg.testEnableName),
      llvm::cl::init("test_en"), llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> emitSeparateAlwaysBlocks{
      "emit-separate-always-blocks",
      llvm::cl::desc(
          "Prevent always blocks from being merged and emit constructs into "
          "separate always blocks whenever possible"),
      llvm::cl::location(hwToSVOpts.emitSeparateAlwaysBlocks),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> addMuxPragmas{
      "add-mux-pragmas",
      llvm::cl::desc("Annotate mux pragmas for memory array access"),
      llvm::cl::location(hwToSVOpts.addMuxPragmas), llvm::cl::init(false),
      llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> addVivadoRAMAddressConflictSynthesisBugWorkaround{
      "add-vivado-ram-address-conflict-synthesis-bug-workaround",
      llvm::cl::desc(
          "Add a vivado specific SV attribute (* ram_style = "
          "\"distributed\" *) to unpacked array registers as a workaronud "
          "for a vivado synthesis bug that incorrectly modifies "
          "address conflict behavivor of combinational memories"),
      llvm::cl::location(
          hwToSVOpts.addVivadoRAMAddressConflictSynthesisBugWorkaround),
      llvm::cl::init(false), llvm::cl::cat(category)};

  // ========== ExportVerilog ==========

  mutable FirtoolExportVerilogOptions exportVerilogOpts{&generalOpts};

  llvm::cl::opt<bool, true> stripFirDebugInfo{
      "strip-fir-debug-info",
      llvm::cl::desc(
          "Disable source fir locator information in output Verilog"),
      llvm::cl::location(exportVerilogOpts.stripFirDebugInfo),
      llvm::cl::init(true), llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> stripDebugInfo{
      "strip-debug-info",
      llvm::cl::desc("Disable source locator information in output Verilog"),
      llvm::cl::location(exportVerilogOpts.stripDebugInfo),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool, true> exportModuleHierarchy{
      "export-module-hierarchy",
      llvm::cl::desc("Export module and instance hierarchy as JSON"),
      llvm::cl::location(exportVerilogOpts.exportModuleHierarchy),
      llvm::cl::init(false), llvm::cl::cat(category)};

  // ========== ExportVerilog ==========

  FirtoolFinalizeIROptions finalizeIROpts{&generalOpts};

  //

  const FirtoolPreprocessTransformsOptions &
  getPreprocessTransformsOptions() const {
    return preprocessTransformsOpts;
  }

  const FirtoolCHIRRTLToLowFIRRTLOptions &getCHIRRTLToLowFIRRTLOptions() const {
    if (buildMode.getNumOccurrences()) {
      switch (buildMode) {
      case BuildModeDebug:
        chirrtlToLowFIRRTLOpts.preserveValues = firrtl::PreserveValues::Named;
        break;
      case BuildModeRelease:
        chirrtlToLowFIRRTLOpts.preserveValues = firrtl::PreserveValues::None;
        break;
      default:
        llvm_unreachable("unknown build mode");
      }
    }
    return chirrtlToLowFIRRTLOpts;
  }

  const FirtoolLowFIRRTLToHWOptions &getLowFIRRTLToHWOptions() const {
    return lowFIRRTLToHWOpts;
  }

  const FirtoolHWToSVOptions &getHWToSVOptions() const { return hwToSVOpts; }

  const FirtoolExportVerilogOptions &getExportVerilogOptions() const {
    exportVerilogOpts.outputPath = outputFilename;
    return exportVerilogOpts;
  }

  const FirtoolFinalizeIROptions &getFinalizeIROptions() const {
    return finalizeIROpts;
  }

  FirtoolOptions(llvm::cl::OptionCategory &category) : category(category) {}
};

LogicalResult
populatePreprocessTransforms(mlir::PassManager &pm,
                             const FirtoolPreprocessTransformsOptions &opt);

LogicalResult
populateCHIRRTLToLowFIRRTL(mlir::PassManager &pm,
                           const FirtoolCHIRRTLToLowFIRRTLOptions &opt);

LogicalResult populateLowFIRRTLToHW(mlir::PassManager &pm,
                                    const FirtoolLowFIRRTLToHWOptions &opt);

LogicalResult populateHWToSV(mlir::PassManager &pm,
                             const FirtoolHWToSVOptions &opt);

LogicalResult populateExportVerilog(mlir::PassManager &pm,
                                    const FirtoolExportVerilogOptions &opt,
                                    std::unique_ptr<llvm::raw_ostream> os);

LogicalResult populateExportVerilog(mlir::PassManager &pm,
                                    const FirtoolExportVerilogOptions &opt,
                                    llvm::raw_ostream &os);

LogicalResult
populateExportSplitVerilog(mlir::PassManager &pm,
                           const FirtoolExportVerilogOptions &opt);

LogicalResult populateFinalizeIR(mlir::PassManager &pm,
                                 const FirtoolFinalizeIROptions &opt);

} // namespace firtool
} // namespace circt

#endif // CIRCT_FIRTOOL_FIRTOOL_H
