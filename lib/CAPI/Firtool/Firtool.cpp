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

DEFINE_C_API_PTR_METHODS(FirtoolOptions, firtool::FirtoolOptions)

FirtoolOptions firtoolOptionsCreateDefault() {
  static auto category = llvm::cl::OptionCategory{"Firtool Options"};
  auto *options = new firtool::FirtoolOptions{category};
  return wrap(options);
}

void firtoolOptionsDestroy(FirtoolOptions options) { delete unwrap(options); }

#define DEFINE_FIRTOOL_OPTION_STRING(name, field)                              \
  void firtoolOptionsSet##name(FirtoolOptions options, MlirStringRef value) {  \
    unwrap(options)->field = unwrap(value).str();                              \
  }                                                                            \
  MlirStringRef firtoolOptionsGet##name(FirtoolOptions options) {              \
    return wrap(unwrap(options)->field.getValue());                            \
  }

#define DEFINE_FIRTOOL_OPTION_BOOL(name, field)                                \
  void firtoolOptionsSet##name(FirtoolOptions options, bool value) {           \
    unwrap(options)->field = value;                                            \
  }                                                                            \
  bool firtoolOptionsGet##name(FirtoolOptions options) {                       \
    return unwrap(options)->field;                                             \
  }

#define DEFINE_FIRTOOL_OPTION_ENUM(name, field, enum_type, c_to_cpp, cpp_to_c) \
  void firtoolOptionsSet##name(FirtoolOptions options, enum_type value) {      \
    unwrap(options)->field = c_to_cpp(value);                                  \
  }                                                                            \
  enum_type firtoolOptionsGet##name(FirtoolOptions options) {                  \
    return cpp_to_c(unwrap(options)->field);                                   \
  }

DEFINE_FIRTOOL_OPTION_STRING(OutputFilename, outputFilename)
DEFINE_FIRTOOL_OPTION_BOOL(DisableAnnotationsUnknown, disableAnnotationsUnknown)
DEFINE_FIRTOOL_OPTION_BOOL(DisableAnnotationsClassless,
                           disableAnnotationsClassless)
DEFINE_FIRTOOL_OPTION_BOOL(LowerAnnotationsNoRefTypePorts,
                           lowerAnnotationsNoRefTypePorts)
DEFINE_FIRTOOL_OPTION_ENUM(
    PreserveAggregate, preserveAggregate, FirtoolPreserveAggregateMode,
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
DEFINE_FIRTOOL_OPTION_ENUM(
    PreserveValues, preserveMode, FirtoolPreserveValuesMode,
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
    BuildMode, buildMode, FirtoolBuildMode,
    [](FirtoolBuildMode value) {
      switch (value) {
      case FIRTOOL_BUILD_MODE_DEBUG:
        return firtool::FirtoolOptions::BuildModeDebug;
      case FIRTOOL_BUILD_MODE_RELEASE:
        return firtool::FirtoolOptions::BuildModeRelease;
      default: // NOLINT(clang-diagnostic-covered-switch-default)
        llvm_unreachable("unknown build mode");
      }
    },
    [](firtool::FirtoolOptions::BuildMode value) {
      switch (value) {
      case firtool::FirtoolOptions::BuildModeDebug:
        return FIRTOOL_BUILD_MODE_DEBUG;
      case firtool::FirtoolOptions::BuildModeRelease:
        return FIRTOOL_BUILD_MODE_RELEASE;
      default: // NOLINT(clang-diagnostic-covered-switch-default)
        llvm_unreachable("unknown build mode");
      }
    })
DEFINE_FIRTOOL_OPTION_BOOL(DisableOptimization, disableOptimization)
DEFINE_FIRTOOL_OPTION_BOOL(ExportChiselInterface, exportChiselInterface)
DEFINE_FIRTOOL_OPTION_STRING(ChiselInterfaceOutDirectory,
                             chiselInterfaceOutDirectory)
DEFINE_FIRTOOL_OPTION_BOOL(VbToBv, vbToBV)
DEFINE_FIRTOOL_OPTION_ENUM(
    CompanionMode, companionMode, FirtoolCompanionMode,
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

DEFINE_FIRTOOL_OPTION_BOOL(DisableAggressiveMergeConnections,
                           disableAggressiveMergeConnections)
DEFINE_FIRTOOL_OPTION_BOOL(EmitOMIR, emitOMIR)
DEFINE_FIRTOOL_OPTION_STRING(OMIROutFile, omirOutFile)
DEFINE_FIRTOOL_OPTION_BOOL(LowerMemories, lowerMemories)
DEFINE_FIRTOOL_OPTION_STRING(BlackBoxRootPath, blackBoxRootPath)
DEFINE_FIRTOOL_OPTION_BOOL(ReplSeqMem, replSeqMem)
DEFINE_FIRTOOL_OPTION_STRING(ReplSeqMemFile, replSeqMemFile)
DEFINE_FIRTOOL_OPTION_BOOL(ExtractTestCode, extractTestCode)
DEFINE_FIRTOOL_OPTION_BOOL(IgnoreReadEnableMem, ignoreReadEnableMem)
DEFINE_FIRTOOL_OPTION_ENUM(
    DisableRandom, disableRandom, FirtoolRandomKind,
    [](FirtoolRandomKind value) {
      switch (value) {
      case FIRTOOL_RANDOM_KIND_NONE:
        return firtool::FirtoolOptions::RandomKind::None;
      case FIRTOOL_RANDOM_KIND_MEM:
        return firtool::FirtoolOptions::RandomKind::Mem;
      case FIRTOOL_RANDOM_KIND_REG:
        return firtool::FirtoolOptions::RandomKind::Reg;
      case FIRTOOL_RANDOM_KIND_ALL:
        return firtool::FirtoolOptions::RandomKind::All;
      default: // NOLINT(clang-diagnostic-covered-switch-default)
        llvm_unreachable("unknown random kind");
      }
    },
    [](firtool::FirtoolOptions::RandomKind value) {
      switch (value) {
      case firtool::FirtoolOptions::RandomKind::None:
        return FIRTOOL_RANDOM_KIND_NONE;
      case firtool::FirtoolOptions::RandomKind::Mem:
        return FIRTOOL_RANDOM_KIND_MEM;
      case firtool::FirtoolOptions::RandomKind::Reg:
        return FIRTOOL_RANDOM_KIND_REG;
      case firtool::FirtoolOptions::RandomKind::All:
        return FIRTOOL_RANDOM_KIND_ALL;
      default: // NOLINT(clang-diagnostic-covered-switch-default)
        llvm_unreachable("unknown random kind");
      }
    })
DEFINE_FIRTOOL_OPTION_STRING(OutputAnnotationFilename, outputAnnotationFilename)
DEFINE_FIRTOOL_OPTION_BOOL(EnableAnnotationWarning, enableAnnotationWarning)
DEFINE_FIRTOOL_OPTION_BOOL(AddMuxPragmas, addMuxPragmas)
DEFINE_FIRTOOL_OPTION_BOOL(EmitChiselAssertsAsSVA, emitChiselAssertsAsSVA)
DEFINE_FIRTOOL_OPTION_BOOL(EmitSeparateAlwaysBlocks, emitSeparateAlwaysBlocks)
DEFINE_FIRTOOL_OPTION_BOOL(EtcDisableInstanceExtraction,
                           etcDisableInstanceExtraction)
DEFINE_FIRTOOL_OPTION_BOOL(EtcDisableRegisterExtraction,
                           etcDisableRegisterExtraction)
DEFINE_FIRTOOL_OPTION_BOOL(EtcDisableModuleInlining, etcDisableModuleInlining)
DEFINE_FIRTOOL_OPTION_BOOL(AddVivadoRAMAddressConflictSynthesisBugWorkaround,
                           addVivadoRAMAddressConflictSynthesisBugWorkaround)
DEFINE_FIRTOOL_OPTION_STRING(CkgModuleName, ckgModuleName)
DEFINE_FIRTOOL_OPTION_STRING(CkgInputName, ckgInputName)
DEFINE_FIRTOOL_OPTION_STRING(CkgOutputName, ckgOutputName)
DEFINE_FIRTOOL_OPTION_STRING(CkgEnableName, ckgEnableName)
DEFINE_FIRTOOL_OPTION_STRING(CkgTestEnableName, ckgTestEnableName)
DEFINE_FIRTOOL_OPTION_BOOL(ExportModuleHierarchy, exportModuleHierarchy)
DEFINE_FIRTOOL_OPTION_BOOL(StripFirDebugInfo, stripFirDebugInfo)
DEFINE_FIRTOOL_OPTION_BOOL(StripDebugInfo, stripDebugInfo)

#undef DEFINE_FIRTOOL_OPTION_STRING
#undef DEFINE_FIRTOOL_OPTION_BOOL
#undef DEFINE_FIRTOOL_OPTION_ENUM

//===----------------------------------------------------------------------===//
// Populate API.
//===----------------------------------------------------------------------===//

MlirLogicalResult firtoolPopulatePreprocessTransforms(MlirPassManager pm,
                                                      FirtoolOptions options) {
  return wrap(
      firtool::populatePreprocessTransforms(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult
firtoolPopulateCHIRRTLToLowFIRRTL(MlirPassManager pm, FirtoolOptions options,
                                  MlirStringRef inputFilename) {
  return wrap(firtool::populateCHIRRTLToLowFIRRTL(*unwrap(pm), *unwrap(options),
                                                  unwrap(inputFilename)));
}

MlirLogicalResult firtoolPopulateLowFIRRTLToHW(MlirPassManager pm,
                                               FirtoolOptions options) {
  return wrap(firtool::populateLowFIRRTLToHW(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult firtoolPopulateHWToSV(MlirPassManager pm,
                                        FirtoolOptions options) {
  return wrap(firtool::populateHWToSV(*unwrap(pm), *unwrap(options)));
}

MlirLogicalResult firtoolPopulateExportVerilog(MlirPassManager pm,
                                               FirtoolOptions options,
                                               MlirStringCallback callback,
                                               void *userData) {
  auto stream =
      std::make_unique<mlir::detail::CallbackOstream>(callback, userData);
  return wrap(firtool::populateExportVerilog(*unwrap(pm), *unwrap(options),
                                             std::move(stream)));
}

MlirLogicalResult firtoolPopulateExportSplitVerilog(MlirPassManager pm,
                                                    FirtoolOptions options,
                                                    MlirStringRef directory) {
  return wrap(firtool::populateExportSplitVerilog(*unwrap(pm), *unwrap(options),
                                                  unwrap(directory)));
}

MlirLogicalResult firtoolPopulateFinalizeIR(MlirPassManager pm,
                                            FirtoolOptions options) {
  return wrap(firtool::populateFinalizeIR(*unwrap(pm), *unwrap(options)));
}
