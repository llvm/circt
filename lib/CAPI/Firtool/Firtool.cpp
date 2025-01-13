//===- Firtool.cpp - C Interface to Firtool-lib ---------------------------===//
//
//  Implements a C Interface for Firtool-lib.
//
//===----------------------------------------------------------------------===//

#include "circt-c/Firtool/Firtool.h"
#include "circt/Firtool/Firtool.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;

DEFINE_C_API_PTR_METHODS(CirctFirtoolFirtoolOptions,
                         circt::firtool::FirtoolOptions)

//===----------------------------------------------------------------------===//
// Option API.
//===----------------------------------------------------------------------===//

CirctFirtoolFirtoolOptions circtFirtoolOptionsCreateDefault() {
  auto *options = new firtool::FirtoolOptions();
  return wrap(options);
}

void circtFirtoolOptionsDestroy(CirctFirtoolFirtoolOptions options) {
  delete unwrap(options);
}

void circtFirtoolOptionsSetOutputFilename(CirctFirtoolFirtoolOptions options,
                                          MlirStringRef filename) {
  unwrap(options)->setOutputFilename(unwrap(filename));
}

void circtFirtoolOptionsSetDisableUnknownAnnotations(
    CirctFirtoolFirtoolOptions options, bool disable) {
  unwrap(options)->setDisableUnknownAnnotations(disable);
}

void circtFirtoolOptionsSetDisableAnnotationsClassless(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setDisableAnnotationsClassless(value);
}

void circtFirtoolOptionsSetLowerAnnotationsNoRefTypePorts(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setLowerAnnotationsNoRefTypePorts(value);
}

void circtFirtoolOptionsSetAllowAddingPortsOnPublic(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setAllowAddingPortsOnPublic(value);
}

void circtFirtoolOptionsSetPreserveAggregate(
    CirctFirtoolFirtoolOptions options,
    CirctFirtoolPreserveAggregateMode value) {
  firrtl::PreserveAggregate::PreserveMode converted;

  switch (value) {
  case CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_NONE:
    converted = firrtl::PreserveAggregate::PreserveMode::None;
    break;
  case CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_ONE_DIM_VEC:
    converted = firrtl::PreserveAggregate::PreserveMode::OneDimVec;
    break;
  case CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_VEC:
    converted = firrtl::PreserveAggregate::PreserveMode::Vec;
    break;
  case CIRCT_FIRTOOL_PRESERVE_AGGREGATE_MODE_ALL:
    converted = firrtl::PreserveAggregate::PreserveMode::All;
    break;
  }

  unwrap(options)->setPreserveAggregate(converted);
}

void circtFirtoolOptionsSetPreserveValues(
    CirctFirtoolFirtoolOptions options, CirctFirtoolPreserveValuesMode value) {
  firrtl::PreserveValues::PreserveMode converted;

  switch (value) {
  case CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_STRIP:
    converted = firrtl::PreserveValues::PreserveMode::Strip;
    break;
  case CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_NONE:
    converted = firrtl::PreserveValues::PreserveMode::None;
    break;
  case CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_NAMED:
    converted = firrtl::PreserveValues::PreserveMode::Named;
    break;
  case CIRCT_FIRTOOL_PRESERVE_VALUES_MODE_ALL:
    converted = firrtl::PreserveValues::PreserveMode::All;
    break;
  }

  unwrap(options)->setPreserveValues(converted);
}

void circtFirtoolOptionsSetEnableDebugInfo(CirctFirtoolFirtoolOptions options,
                                           bool value) {
  unwrap(options)->setEnableDebugInfo(value);
}

void circtFirtoolOptionsSetBuildMode(CirctFirtoolFirtoolOptions options,
                                     CirctFirtoolBuildMode value) {
  firtool::FirtoolOptions::BuildMode converted;

  switch (value) {
  case CIRCT_FIRTOOL_BUILD_MODE_DEFAULT:
    converted = firtool::FirtoolOptions::BuildMode::BuildModeDefault;
    break;
  case CIRCT_FIRTOOL_BUILD_MODE_DEBUG:
    converted = firtool::FirtoolOptions::BuildMode::BuildModeDebug;
    break;
  case CIRCT_FIRTOOL_BUILD_MODE_RELEASE:
    converted = firtool::FirtoolOptions::BuildMode::BuildModeRelease;
    break;
  }

  unwrap(options)->setBuildMode(converted);
}

void circtFirtoolOptionsSetDisableLayerSink(CirctFirtoolFirtoolOptions options,
                                            bool value) {
  unwrap(options)->setDisableLayerSink(value);
}

void circtFirtoolOptionsSetDisableOptimization(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setDisableOptimization(value);
}

void circtFirtoolOptionsSetExportChiselInterface(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setExportChiselInterface(value);
}

void circtFirtoolOptionsSetChiselInterfaceOutDirectory(
    CirctFirtoolFirtoolOptions options, MlirStringRef value) {
  unwrap(options)->setChiselInterfaceOutDirectory(unwrap(value));
}

void circtFirtoolOptionsSetVbToBv(CirctFirtoolFirtoolOptions options,
                                  bool value) {
  unwrap(options)->setVbToBV(value);
}

void circtFirtoolOptionsSetNoDedup(CirctFirtoolFirtoolOptions options,
                                   bool value) {
  unwrap(options)->setNoDedup(value);
}

void circtFirtoolOptionsSetCompanionMode(CirctFirtoolFirtoolOptions options,
                                         CirctFirtoolCompanionMode value) {
  firrtl::CompanionMode converted;

  switch (value) {
  case CIRCT_FIRTOOL_COMPANION_MODE_BIND:
    converted = firrtl::CompanionMode::Bind;
    break;
  case CIRCT_FIRTOOL_COMPANION_MODE_INSTANTIATE:
    converted = firrtl::CompanionMode::Instantiate;
    break;
  case CIRCT_FIRTOOL_COMPANION_MODE_DROP:
    converted = firrtl::CompanionMode::Drop;
    break;
  }

  unwrap(options)->setCompanionMode(converted);
}

void circtFirtoolOptionsSetDisableAggressiveMergeConnections(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setDisableAggressiveMergeConnections(value);
}

void circtFirtoolOptionsSetLowerMemories(CirctFirtoolFirtoolOptions options,
                                         bool value) {
  unwrap(options)->setLowerMemories(value);
}

void circtFirtoolOptionsSetBlackBoxRootPath(CirctFirtoolFirtoolOptions options,
                                            MlirStringRef value) {
  unwrap(options)->setBlackBoxRootPath(unwrap(value));
}

void circtFirtoolOptionsSetReplSeqMem(CirctFirtoolFirtoolOptions options,
                                      bool value) {
  unwrap(options)->setReplSeqMem(value);
}

void circtFirtoolOptionsSetReplSeqMemFile(CirctFirtoolFirtoolOptions options,
                                          MlirStringRef value) {
  unwrap(options)->setReplSeqMemFile(unwrap(value));
}

void circtFirtoolOptionsSetExtractTestCode(CirctFirtoolFirtoolOptions options,
                                           bool value) {
  unwrap(options)->setExtractTestCode(value);
}

void circtFirtoolOptionsSetIgnoreReadEnableMem(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setIgnoreReadEnableMem(value);
}

void circtFirtoolOptionsSetDisableRandom(CirctFirtoolFirtoolOptions options,
                                         CirctFirtoolRandomKind value) {
  firtool::FirtoolOptions::RandomKind converted;

  switch (value) {
  case CIRCT_FIRTOOL_RANDOM_KIND_NONE:
    converted = firtool::FirtoolOptions::RandomKind::None;
    break;
  case CIRCT_FIRTOOL_RANDOM_KIND_MEM:
    converted = firtool::FirtoolOptions::RandomKind::Mem;
    break;
  case CIRCT_FIRTOOL_RANDOM_KIND_REG:
    converted = firtool::FirtoolOptions::RandomKind::Reg;
    break;
  case CIRCT_FIRTOOL_RANDOM_KIND_ALL:
    converted = firtool::FirtoolOptions::RandomKind::All;
    break;
  }

  unwrap(options)->setDisableRandom(converted);
}

void circtFirtoolOptionsSetOutputAnnotationFilename(
    CirctFirtoolFirtoolOptions options, MlirStringRef value) {
  unwrap(options)->setOutputAnnotationFilename(unwrap(value));
}

void circtFirtoolOptionsSetEnableAnnotationWarning(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setEnableAnnotationWarning(value);
}

void circtFirtoolOptionsSetAddMuxPragmas(CirctFirtoolFirtoolOptions options,
                                         bool value) {
  unwrap(options)->setAddMuxPragmas(value);
}

void circtFirtoolOptionsSetVerificationFlavor(
    CirctFirtoolFirtoolOptions options, firrtl::VerificationFlavor value) {
  unwrap(options)->setVerificationFlavor(value);
}

void circtFirtoolOptionsSetEmitSeparateAlwaysBlocks(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setEmitSeparateAlwaysBlocks(value);
}

void circtFirtoolOptionsSetEtcDisableInstanceExtraction(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setEtcDisableInstanceExtraction(value);
}

void circtFirtoolOptionsSetEtcDisableRegisterExtraction(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setEtcDisableRegisterExtraction(value);
}

void circtFirtoolOptionsSetEtcDisableModuleInlining(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setEtcDisableModuleInlining(value);
}

void circtFirtoolOptionsSetAddVivadoRAMAddressConflictSynthesisBugWorkaround(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setAddVivadoRAMAddressConflictSynthesisBugWorkaround(value);
}

void circtFirtoolOptionsSetCkgModuleName(CirctFirtoolFirtoolOptions options,
                                         MlirStringRef value) {
  unwrap(options)->setCkgModuleName(unwrap(value));
}

void circtFirtoolOptionsSetCkgInputName(CirctFirtoolFirtoolOptions options,
                                        MlirStringRef value) {
  unwrap(options)->setCkgInputName(unwrap(value));
}

void circtFirtoolOptionsSetCkgOutputName(CirctFirtoolFirtoolOptions options,
                                         MlirStringRef value) {
  unwrap(options)->setCkgOutputName(unwrap(value));
}

void circtFirtoolOptionsSetCkgEnableName(CirctFirtoolFirtoolOptions options,
                                         MlirStringRef value) {
  unwrap(options)->setCkgEnableName(unwrap(value));
}

void circtFirtoolOptionsSetCkgTestEnableName(CirctFirtoolFirtoolOptions options,
                                             MlirStringRef value) {
  unwrap(options)->setCkgTestEnableName(unwrap(value));
}

void circtFirtoolOptionsSetExportModuleHierarchy(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setExportModuleHierarchy(value);
}

void circtFirtoolOptionsSetStripFirDebugInfo(CirctFirtoolFirtoolOptions options,
                                             bool value) {
  unwrap(options)->setStripFirDebugInfo(value);
}

void circtFirtoolOptionsSetStripDebugInfo(CirctFirtoolFirtoolOptions options,
                                          bool value) {
  unwrap(options)->setStripDebugInfo(value);
}

void circtFirtoolOptionsSetDisableCSEinClasses(
    CirctFirtoolFirtoolOptions options, bool value) {

  unwrap(options)->setDisableCSEinClasses(value);
}

void circtFirtoolOptionsSetSelectDefaultInstanceChoice(
    CirctFirtoolFirtoolOptions options, bool value) {
  unwrap(options)->setSelectDefaultInstanceChoice(value);
}

//===----------------------------------------------------------------------===//
// Populate API.
//===----------------------------------------------------------------------===//

MlirLogicalResult
circtFirtoolPopulatePreprocessTransforms(MlirPassManager pm,
                                         CirctFirtoolFirtoolOptions options) {
  return wrap(
      firtool::populatePreprocessTransforms(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult
circtFirtoolPopulateCHIRRTLToLowFIRRTL(MlirPassManager pm,
                                       CirctFirtoolFirtoolOptions options,
                                       MlirStringRef inputFilename) {
  return wrap(firtool::populateCHIRRTLToLowFIRRTL(*unwrap(pm), *unwrap(options),
                                                  unwrap(inputFilename)));
}

MlirLogicalResult
circtFirtoolPopulateLowFIRRTLToHW(MlirPassManager pm,
                                  CirctFirtoolFirtoolOptions options) {
  return wrap(firtool::populateLowFIRRTLToHW(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult
circtFirtoolPopulateHWToSV(MlirPassManager pm,
                           CirctFirtoolFirtoolOptions options) {
  return wrap(firtool::populateHWToSV(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult
circtFirtoolPopulateExportVerilog(MlirPassManager pm,
                                  CirctFirtoolFirtoolOptions options,
                                  MlirStringCallback callback, void *userData) {
  auto stream =
      std::make_unique<mlir::detail::CallbackOstream>(callback, userData);
  return wrap(firtool::populateExportVerilog(*unwrap(pm), *unwrap(options),
                                             std::move(stream)));
}

MlirLogicalResult
circtFirtoolPopulateExportSplitVerilog(MlirPassManager pm,
                                       CirctFirtoolFirtoolOptions options,
                                       MlirStringRef directory) {
  return wrap(firtool::populateExportSplitVerilog(*unwrap(pm), *unwrap(options),
                                                  unwrap(directory)));
}

MlirLogicalResult
circtFirtoolPopulateFinalizeIR(MlirPassManager pm,
                               CirctFirtoolFirtoolOptions options) {
  return wrap(firtool::populateFinalizeIR(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult
circtFirtoolpopulateHWToBTOR2(MlirPassManager pm,
                              CirctFirtoolFirtoolOptions options,
                              MlirStringCallback callback, void *userData) {
  auto stream =
      std::make_unique<mlir::detail::CallbackOstream>(callback, userData);
  return wrap(firtool::populateHWToBTOR2(*unwrap(pm), *unwrap(options),
                                         *std::move(stream)));
}
