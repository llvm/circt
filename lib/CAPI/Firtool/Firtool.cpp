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

//===----------------------------------------------------------------------===//
// Option API.
//===----------------------------------------------------------------------===//

DEFINE_C_API_PTR_METHODS(FirtoolGeneralOptions, firtool::FirtoolGeneralOptions)
DEFINE_C_API_PTR_METHODS(FirtoolPreprocessTransformsOptions,
                         firtool::FirtoolPreprocessTransformsOptions)
DEFINE_C_API_PTR_METHODS(FirtoolCHIRRTLToLowFIRRTLOptions,
                         firtool::FirtoolCHIRRTLToLowFIRRTLOptions)
DEFINE_C_API_PTR_METHODS(FirtoolLowFIRRTLToHWOptions,
                         firtool::FirtoolLowFIRRTLToHWOptions)
DEFINE_C_API_PTR_METHODS(FirtoolHWToSVOptions, firtool::FirtoolHWToSVOptions)
DEFINE_C_API_PTR_METHODS(FirtoolExportVerilogOptions,
                         firtool::FirtoolExportVerilogOptions)
DEFINE_C_API_PTR_METHODS(FirtoolFinalizeIROptions,
                         firtool::FirtoolFinalizeIROptions)

#define DEFINE_FIRTOOL_OPTIONS_CREATE_DESTROY(opt)                             \
  Firtool##opt##Options firtool##opt##OptionsCreateDefault(                    \
      FirtoolGeneralOptions general) {                                         \
    auto *options = new firtool::Firtool##opt##Options{unwrap(general)};       \
    return wrap(options);                                                      \
  }                                                                            \
  void firtool##opt##OptionsDestroy(Firtool##opt##Options options) {           \
    delete unwrap(options);                                                    \
  }                                                                            \
  void firtool##opt##OptionsSetGeneral(Firtool##opt##Options options,          \
                                       FirtoolGeneralOptions general) {        \
    unwrap(options)->general = unwrap(general);                                \
  }                                                                            \
  FirtoolGeneralOptions firtool##opt##OptionsGetGeneral(                       \
      Firtool##opt##Options options) {                                         \
    return wrap(unwrap(options)->general);                                     \
  }

#define DEFINE_FIRTOOL_OPTION_STRING(opt, name, field)                         \
  void firtool##opt##OptionsSet##name(Firtool##opt##Options options,           \
                                      MlirStringRef value) {                   \
    unwrap(options)->field = unwrap(value).str();                              \
  }                                                                            \
  MlirStringRef firtool##opt##OptionsGet##name(                                \
      Firtool##opt##Options options) {                                         \
    return wrap(unwrap(options)->field);                                       \
  }

#define DEFINE_FIRTOOL_OPTION_BOOL(opt, name, field)                           \
  void firtool##opt##OptionsSet##name(Firtool##opt##Options options,           \
                                      bool value) {                            \
    unwrap(options)->field = value;                                            \
  }                                                                            \
  bool firtool##opt##OptionsGet##name(Firtool##opt##Options options) {         \
    return unwrap(options)->field;                                             \
  }

#define DEFINE_FIRTOOL_OPTION_ENUM(opt, name, field, enum_type, c_to_cpp,      \
                                   cpp_to_c)                                   \
  void firtool##opt##OptionsSet##name(Firtool##opt##Options options,           \
                                      enum_type value) {                       \
    unwrap(options)->field = c_to_cpp(value);                                  \
  }                                                                            \
  enum_type firtool##opt##OptionsGet##name(Firtool##opt##Options options) {    \
    return cpp_to_c(unwrap(options)->field);                                   \
  }

// ========== General ==========

FirtoolGeneralOptions firtoolGeneralOptionsCreateDefault() {
  auto *options = new firtool::FirtoolGeneralOptions;
  return wrap(options);
}

void firtoolGeneralOptionsDestroy(FirtoolGeneralOptions options) {
  delete unwrap(options);
}

DEFINE_FIRTOOL_OPTION_BOOL(General, DisableOptimization, disableOptimization)
DEFINE_FIRTOOL_OPTION_BOOL(General, ReplSeqMem, replSeqMem)
DEFINE_FIRTOOL_OPTION_STRING(General, ReplSeqMemFile, replSeqMemFile)
DEFINE_FIRTOOL_OPTION_BOOL(General, IgnoreReadEnableMem, ignoreReadEnableMem)
DEFINE_FIRTOOL_OPTION_ENUM(
    General, DisableRandom, disableRandom, FirtoolRandomKind,
    [](FirtoolRandomKind value) {
      switch (value) {
      case FIRTOOL_RANDOM_KIND_NONE:
        return firtool::RandomKind::None;
      case FIRTOOL_RANDOM_KIND_MEM:
        return firtool::RandomKind::Mem;
      case FIRTOOL_RANDOM_KIND_REG:
        return firtool::RandomKind::Reg;
      case FIRTOOL_RANDOM_KIND_ALL:
        return firtool::RandomKind::All;
      default: // NOLINT(clang-diagnostic-covered-switch-default)
        llvm_unreachable("unknown random kind");
      }
    },
    [](firtool::RandomKind value) {
      switch (value) {
      case firtool::RandomKind::None:
        return FIRTOOL_RANDOM_KIND_NONE;
      case firtool::RandomKind::Mem:
        return FIRTOOL_RANDOM_KIND_MEM;
      case firtool::RandomKind::Reg:
        return FIRTOOL_RANDOM_KIND_REG;
      case firtool::RandomKind::All:
        return FIRTOOL_RANDOM_KIND_ALL;
      default: // NOLINT(clang-diagnostic-covered-switch-default)
        llvm_unreachable("unknown random kind");
      }
    })

// ========== PreprocessTransforms ==========

DEFINE_FIRTOOL_OPTIONS_CREATE_DESTROY(PreprocessTransforms)
DEFINE_FIRTOOL_OPTION_BOOL(PreprocessTransforms, DisableAnnotationsUnknown,
                           disableAnnotationsUnknown)
DEFINE_FIRTOOL_OPTION_BOOL(PreprocessTransforms, DisableAnnotationsClassless,
                           disableAnnotationsClassless)
DEFINE_FIRTOOL_OPTION_BOOL(PreprocessTransforms, LowerAnnotationsNoRefTypePorts,
                           lowerAnnotationsNoRefTypePorts)

// ========== CHIRRTLToLowFIRRTL ==========

DEFINE_FIRTOOL_OPTIONS_CREATE_DESTROY(CHIRRTLToLowFIRRTL)
DEFINE_FIRTOOL_OPTION_ENUM(
    CHIRRTLToLowFIRRTL, PreserveValues, preserveValues,
    FirtoolPreserveValuesMode,
    [](FirtoolPreserveValuesMode value) {
      switch (value) {
      case FIRTOOL_PRESERVE_VALUES_MODE_NONE:
        return firrtl::PreserveValues::None;
      case FIRTOOL_PRESERVE_VALUES_MODE_NAMED:
        return firrtl::PreserveValues::Named;
      case FIRTOOL_PRESERVE_VALUES_MODE_ALL:
        return firrtl::PreserveValues::All;
      default: // NOLINT(clang-diagnostic-covered-switch-default)
        llvm_unreachable("unknown preserve values mode");
      }
    },
    [](firrtl::PreserveValues::PreserveMode value) {
      switch (value) {
      case firrtl::PreserveValues::None:
        return FIRTOOL_PRESERVE_VALUES_MODE_NONE;
      case firrtl::PreserveValues::Named:
        return FIRTOOL_PRESERVE_VALUES_MODE_NAMED;
      case firrtl::PreserveValues::All:
        return FIRTOOL_PRESERVE_VALUES_MODE_ALL;
      default: // NOLINT(clang-diagnostic-covered-switch-default)
        llvm_unreachable("unknown preserve values mode");
      }
    })
DEFINE_FIRTOOL_OPTION_ENUM(
    CHIRRTLToLowFIRRTL, PreserveAggregate, preserveAggregate,
    FirtoolPreserveAggregateMode,
    [](FirtoolPreserveAggregateMode value) {
      switch (value) {
      case FIRTOOL_PRESERVE_AGGREGATE_MODE_NONE:
        return firrtl::PreserveAggregate::None;
      case FIRTOOL_PRESERVE_AGGREGATE_MODE_ONE_DIM_VEC:
        return firrtl::PreserveAggregate::OneDimVec;
      case FIRTOOL_PRESERVE_AGGREGATE_MODE_VEC:
        return firrtl::PreserveAggregate::Vec;
      case FIRTOOL_PRESERVE_AGGREGATE_MODE_ALL:
        return firrtl::PreserveAggregate::All;
      default: // NOLINT(clang-diagnostic-covered-switch-default)
        llvm_unreachable("unknown preserve aggregate mode");
      }
    },
    [](firrtl::PreserveAggregate::PreserveMode value) {
      switch (value) {
      case firrtl::PreserveAggregate::None:
        return FIRTOOL_PRESERVE_AGGREGATE_MODE_NONE;
      case firrtl::PreserveAggregate::OneDimVec:
        return FIRTOOL_PRESERVE_AGGREGATE_MODE_ONE_DIM_VEC;
      case firrtl::PreserveAggregate::Vec:
        return FIRTOOL_PRESERVE_AGGREGATE_MODE_VEC;
      case firrtl::PreserveAggregate::All:
        return FIRTOOL_PRESERVE_AGGREGATE_MODE_ALL;
      default: // NOLINT(clang-diagnostic-covered-switch-default)
        llvm_unreachable("unknown preserve aggregate mode");
      }
    })
DEFINE_FIRTOOL_OPTION_BOOL(CHIRRTLToLowFIRRTL, ExportChiselInterface,
                           exportChiselInterface)
DEFINE_FIRTOOL_OPTION_STRING(CHIRRTLToLowFIRRTL, ChiselInterfaceOutDirectory,
                             chiselInterfaceOutDirectory)
DEFINE_FIRTOOL_OPTION_BOOL(CHIRRTLToLowFIRRTL, DisableHoistingHWPassthrough,
                           disableHoistingHWPassthrough)
DEFINE_FIRTOOL_OPTION_BOOL(CHIRRTLToLowFIRRTL, Dedup, dedup)
DEFINE_FIRTOOL_OPTION_BOOL(CHIRRTLToLowFIRRTL, NoDedup, noDedup)
DEFINE_FIRTOOL_OPTION_BOOL(CHIRRTLToLowFIRRTL, VbToBv, vbToBV)
DEFINE_FIRTOOL_OPTION_BOOL(CHIRRTLToLowFIRRTL, LowerMemories, lowerMemories)
DEFINE_FIRTOOL_OPTION_ENUM(
    CHIRRTLToLowFIRRTL, CompanionMode, companionMode, FirtoolCompanionMode,
    [](FirtoolCompanionMode value) {
      switch (value) {
      case FIRTOOL_COMPANION_MODE_BIND:
        return firrtl::CompanionMode::Bind;
      case FIRTOOL_COMPANION_MODE_INSTANTIATE:
        return firrtl::CompanionMode::Instantiate;
      case FIRTOOL_COMPANION_MODE_DROP:
        return firrtl::CompanionMode::Drop;
      default: // NOLINT(clang-diagnostic-covered-switch-default)
        llvm_unreachable("unknown build mode");
      }
    },
    [](firrtl::CompanionMode value) {
      switch (value) {
      case firrtl::CompanionMode::Bind:
        return FIRTOOL_COMPANION_MODE_BIND;
      case firrtl::CompanionMode::Instantiate:
        return FIRTOOL_COMPANION_MODE_INSTANTIATE;
      case firrtl::CompanionMode::Drop:
        return FIRTOOL_COMPANION_MODE_DROP;
      default: // NOLINT(clang-diagnostic-covered-switch-default)
        llvm_unreachable("unknown build mode");
      }
    })
DEFINE_FIRTOOL_OPTION_STRING(CHIRRTLToLowFIRRTL, BlackBoxRootPath,
                             blackBoxRootPath)
DEFINE_FIRTOOL_OPTION_BOOL(CHIRRTLToLowFIRRTL, EmitOMIR, emitOMIR)
DEFINE_FIRTOOL_OPTION_STRING(CHIRRTLToLowFIRRTL, OMIROutFile, omirOutFile)
DEFINE_FIRTOOL_OPTION_BOOL(CHIRRTLToLowFIRRTL,
                           DisableAggressiveMergeConnections,
                           disableAggressiveMergeConnections)

// ========== LowFIRRTLToHW ==========

DEFINE_FIRTOOL_OPTIONS_CREATE_DESTROY(LowFIRRTLToHW)
DEFINE_FIRTOOL_OPTION_STRING(LowFIRRTLToHW, OutputAnnotationFilename,
                             outputAnnotationFilename)
DEFINE_FIRTOOL_OPTION_BOOL(LowFIRRTLToHW, EnableAnnotationWarning,
                           enableAnnotationWarning)
DEFINE_FIRTOOL_OPTION_BOOL(LowFIRRTLToHW, EmitChiselAssertsAsSVA,
                           emitChiselAssertsAsSVA)

// ========== HWToSV ==========

DEFINE_FIRTOOL_OPTIONS_CREATE_DESTROY(HWToSV)
DEFINE_FIRTOOL_OPTION_BOOL(HWToSV, ExtractTestCode, extractTestCode)
DEFINE_FIRTOOL_OPTION_BOOL(HWToSV, EtcDisableInstanceExtraction,
                           etcDisableInstanceExtraction)
DEFINE_FIRTOOL_OPTION_BOOL(HWToSV, EtcDisableRegisterExtraction,
                           etcDisableRegisterExtraction)
DEFINE_FIRTOOL_OPTION_BOOL(HWToSV, EtcDisableModuleInlining,
                           etcDisableModuleInlining)
DEFINE_FIRTOOL_OPTION_STRING(HWToSV, CkgModuleName, ckg.moduleName)
DEFINE_FIRTOOL_OPTION_STRING(HWToSV, CkgInputName, ckg.inputName)
DEFINE_FIRTOOL_OPTION_STRING(HWToSV, CkgOutputName, ckg.outputName)
DEFINE_FIRTOOL_OPTION_STRING(HWToSV, CkgEnableName, ckg.enableName)
DEFINE_FIRTOOL_OPTION_STRING(HWToSV, CkgTestEnableName, ckg.testEnableName)
DEFINE_FIRTOOL_OPTION_STRING(HWToSV, CkgInstName, ckg.instName)
DEFINE_FIRTOOL_OPTION_BOOL(HWToSV, EmitSeparateAlwaysBlocks,
                           emitSeparateAlwaysBlocks)
DEFINE_FIRTOOL_OPTION_BOOL(HWToSV, AddMuxPragmas, addMuxPragmas)
DEFINE_FIRTOOL_OPTION_BOOL(HWToSV,
                           AddVivadoRAMAddressConflictSynthesisBugWorkaround,
                           addVivadoRAMAddressConflictSynthesisBugWorkaround)

// ========== ExportVerilog ==========

DEFINE_FIRTOOL_OPTIONS_CREATE_DESTROY(ExportVerilog)
DEFINE_FIRTOOL_OPTION_BOOL(ExportVerilog, StripFirDebugInfo, stripFirDebugInfo)
DEFINE_FIRTOOL_OPTION_BOOL(ExportVerilog, StripDebugInfo, stripDebugInfo)
DEFINE_FIRTOOL_OPTION_BOOL(ExportVerilog, ExportModuleHierarchy,
                           exportModuleHierarchy)
DEFINE_FIRTOOL_OPTION_STRING(ExportVerilog, OutputPath, outputPath)

// ========== FinalizeIR ==========

DEFINE_FIRTOOL_OPTIONS_CREATE_DESTROY(FinalizeIR)

#undef DEFINE_FIRTOOL_OPTIONS_CREATE_DESTROY
#undef DEFINE_FIRTOOL_OPTION_STRING
#undef DEFINE_FIRTOOL_OPTION_BOOL
#undef DEFINE_FIRTOOL_OPTION_ENUM

//===----------------------------------------------------------------------===//
// Populate API.
//===----------------------------------------------------------------------===//

MlirLogicalResult firtoolPopulatePreprocessTransforms(
    MlirPassManager pm, FirtoolPreprocessTransformsOptions options) {
  return wrap(
      firtool::populatePreprocessTransforms(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult
firtoolPopulateCHIRRTLToLowFIRRTL(MlirPassManager pm,
                                  FirtoolCHIRRTLToLowFIRRTLOptions options) {
  return wrap(
      firtool::populateCHIRRTLToLowFIRRTL(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult
firtoolPopulateLowFIRRTLToHW(MlirPassManager pm,
                             FirtoolLowFIRRTLToHWOptions options) {
  return wrap(firtool::populateLowFIRRTLToHW(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult firtoolPopulateHWToSV(MlirPassManager pm,
                                        FirtoolHWToSVOptions options) {
  return wrap(firtool::populateHWToSV(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult
firtoolPopulateExportVerilog(MlirPassManager pm,
                             FirtoolExportVerilogOptions options,
                             MlirStringCallback callback, void *userData) {
  auto stream =
      std::make_unique<mlir::detail::CallbackOstream>(callback, userData);
  return wrap(firtool::populateExportVerilog(*unwrap(pm), *unwrap(options),
                                             std::move(stream)));
}

MlirLogicalResult
firtoolPopulateExportSplitVerilog(MlirPassManager pm,
                                  FirtoolExportVerilogOptions options) {
  return wrap(
      firtool::populateExportSplitVerilog(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult firtoolPopulateFinalizeIR(MlirPassManager pm,
                                            FirtoolFinalizeIROptions options) {
  return wrap(firtool::populateFinalizeIR(*unwrap(pm), *unwrap(options)));
}
