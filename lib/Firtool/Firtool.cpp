//===- Firtool.cpp - Definitions for the firtool pipeline setup -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Firtool/Firtool.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/Passes.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace circt;

LogicalResult firtool::populatePreprocessTransforms(
    mlir::PassManager &pm, const FirtoolPreprocessTransformsOptions &opt) {
  // Legalize away "open" aggregates to hw-only versions.
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerOpenAggsPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createResolvePathsPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerFIRRTLAnnotationsPass(
      opt.disableAnnotationsUnknown, opt.disableAnnotationsClassless,
      opt.lowerAnnotationsNoRefTypePorts));

  return success();
}

LogicalResult firtool::populateCHIRRTLToLowFIRRTL(
    mlir::PassManager &pm, const FirtoolCHIRRTLToLowFIRRTLOptions &opt) {
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerIntrinsicsPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createInjectDUTHierarchyPass());

  pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
      firrtl::createDropNamesPass(opt.preserveValues));

  if (!opt.general->disableOptimization)
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

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createMemToRegOfVecPass(
      opt.general->replSeqMem, opt.general->ignoreReadEnableMem));

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
      /*hoistHWDrivers=*/!opt.general->disableOptimization &&
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
  if (opt.general->isRandomEnabled(RandomKind::Reg))
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createRandomizeRegisterInitPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createCheckCombLoopsPass());

  // If we parsed a FIRRTL file and have optimizations enabled, clean it up.
  if (!opt.general->disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        createSimpleCanonicalizerPass());

  // Run the infer-rw pass, which merges read and write ports of a memory with
  // mutually exclusive enables.
  if (!opt.general->disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createInferReadWritePass());

  if (opt.general->replSeqMem)
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerMemoryPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createPrefixModulesPass());

  if (!opt.general->disableOptimization) {
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createIMConstPropPass());

    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createHoistPassthroughPass(
        /*hoistHWDrivers=*/!opt.general->disableOptimization &&
        !opt.disableHoistingHWPassthrough));
    // Cleanup after hoisting passthroughs, for separation-of-concerns.
    pm.addPass(firrtl::createIMDeadCodeElimPass());
  }

  pm.addNestedPass<firrtl::CircuitOp>(firrtl::createAddSeqMemPortsPass());

  pm.addPass(firrtl::createCreateSiFiveMetadataPass(
      opt.general->replSeqMem, opt.general->replSeqMemFile));

  pm.addNestedPass<firrtl::CircuitOp>(firrtl::createExtractInstancesPass());

  // Run passes to resolve Grand Central features.  This should run before
  // BlackBoxReader because Grand Central needs to inform BlackBoxReader where
  // certain black boxes should be placed.  Note: all Grand Central Taps related
  // collateral is resolved entirely by LowerAnnotations.
  pm.addNestedPass<firrtl::CircuitOp>(
      firrtl::createGrandCentralPass(opt.companionMode));

  // Read black box source files into the IR.
  pm.nest<firrtl::CircuitOp>().addPass(
      firrtl::createBlackBoxReaderPass(opt.blackBoxRootPath));

  // Run SymbolDCE as late as possible, but before InnerSymbolDCE. This is for
  // hierpathop's and just for general cleanup.
  pm.addNestedPass<firrtl::CircuitOp>(mlir::createSymbolDCEPass());

  // Run InnerSymbolDCE as late as possible, but before IMDCE.
  pm.addPass(firrtl::createInnerSymbolDCEPass());

  // The above passes, IMConstProp in particular, introduce additional
  // canonicalization opportunities that we should pick up here before we
  // proceed to output-specific pipelines.
  if (!opt.general->disableOptimization) {
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
          !opt.disableAggressiveMergeConnections));

  if (!opt.general->disableOptimization)
    pm.nest<firrtl::CircuitOp>().nest<firrtl::FModuleOp>().addPass(
        firrtl::createVectorizationPass());

  return success();
}

LogicalResult
firtool::populateLowFIRRTLToHW(mlir::PassManager &pm,
                               const FirtoolLowFIRRTLToHWOptions &opt) {
  // Remove TraceAnnotations and write their updated paths to an output
  // annotation file.
  if (opt.outputAnnotationFilename.empty())
    pm.nest<firrtl::CircuitOp>().addPass(firrtl::createResolveTracesPass());
  else
    pm.nest<firrtl::CircuitOp>().addPass(
        firrtl::createResolveTracesPass(opt.outputAnnotationFilename));

  // Lower the ref.resolve and ref.send ops and remove the RefType ports.
  // LowerToHW cannot handle RefType so, this pass must be run to remove all
  // RefType ports and ops.
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerXMRPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerClassesPass());

  pm.addPass(createLowerFIRRTLToHWPass(
      opt.enableAnnotationWarning, opt.emitChiselAssertsAsSVA,
      !opt.general->isRandomEnabled(RandomKind::Mem),
      !opt.general->isRandomEnabled(RandomKind::Reg)));

  if (!opt.general->disableOptimization) {
    auto &modulePM = pm.nest<hw::HWModuleOp>();
    modulePM.addPass(mlir::createCSEPass());
    modulePM.addPass(createSimpleCanonicalizerPass());
  }

  // Check inner symbols and inner refs.
  pm.addPass(hw::createVerifyInnerRefNamespacePass());

  return success();
}

LogicalResult firtool::populateHWToSV(mlir::PassManager &pm,
                                      const FirtoolHWToSVOptions &opt) {
  if (opt.extractTestCode)
    pm.addPass(sv::createSVExtractTestCodePass(opt.etcDisableInstanceExtraction,
                                               opt.etcDisableRegisterExtraction,
                                               opt.etcDisableModuleInlining));

  pm.addPass(seq::createExternalizeClockGatePass(opt.ckg));
  pm.addPass(circt::createLowerSeqToSVPass(
      {/*disableRandomization=*/!opt.general->isRandomEnabled(RandomKind::Reg),
       /*emitSeparateAlwaysBlocks=*/
       opt.emitSeparateAlwaysBlocks}));
  pm.addNestedPass<hw::HWModuleOp>(createLowerVerifToSVPass());
  pm.addPass(sv::createHWMemSimImplPass(
      opt.general->replSeqMem, opt.general->ignoreReadEnableMem,
      opt.addMuxPragmas, !opt.general->isRandomEnabled(RandomKind::Mem),
      !opt.general->isRandomEnabled(RandomKind::Reg),
      opt.addVivadoRAMAddressConflictSynthesisBugWorkaround));

  // If enabled, run the optimizer.
  if (!opt.general->disableOptimization) {
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
LogicalResult populatePrepareForExportVerilog(
    mlir::PassManager &pm, const firtool::FirtoolExportVerilogOptions &opt) {

  // Legalize unsupported operations within the modules.
  pm.nest<hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());

  // Tidy up the IR to improve verilog emission quality.
  if (!opt.general->disableOptimization)
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
    pm.addPass(sv::createHWExportModuleHierarchyPass(opt.outputPath));

  // Check inner symbols and inner refs.
  pm.addPass(hw::createVerifyInnerRefNamespacePass());

  return success();
}
} // namespace detail

LogicalResult
firtool::populateExportVerilog(mlir::PassManager &pm,
                               const FirtoolExportVerilogOptions &opt,
                               std::unique_ptr<llvm::raw_ostream> os) {
  pm.addPass(createExportVerilogPass(std::move(os)));

  return success();
}

LogicalResult
firtool::populateExportVerilog(mlir::PassManager &pm,
                               const FirtoolExportVerilogOptions &opt,
                               llvm::raw_ostream &os) {
  if (failed(::detail::populatePrepareForExportVerilog(pm, opt)))
    return failure();

  pm.addPass(createExportVerilogPass(os));
  return success();
}

LogicalResult
firtool::populateExportSplitVerilog(mlir::PassManager &pm,
                                    const FirtoolExportVerilogOptions &opt) {
  if (failed(::detail::populatePrepareForExportVerilog(pm, opt)))
    return failure();

  pm.addPass(createExportSplitVerilogPass(opt.outputPath));
  return success();
}

LogicalResult firtool::populateFinalizeIR(mlir::PassManager &pm,
                                          const FirtoolFinalizeIROptions &opt) {
  pm.addPass(firrtl::createFinalizeIRPass());
  pm.addPass(om::createFreezePathsPass());

  return success();
}
