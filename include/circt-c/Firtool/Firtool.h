//===-- circt-c/Firtool/Firtool.h - C API for Firtool-lib ---------*- C -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_FIRTOOL_FIRTOOL_H
#define CIRCT_C_FIRTOOL_FIRTOOL_H

#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Option API.
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(FirtoolOptions, void);

#undef DEFINE_C_API_STRUCT

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FirtoolPreserveAggregateMode {
  FIRTOOL_PRESERVE_AGGREGATE_MODE_NONE,
  FIRTOOL_PRESERVE_AGGREGATE_MODE_ONE_DIM_VEC,
  FIRTOOL_PRESERVE_AGGREGATE_MODE_VEC,
  FIRTOOL_PRESERVE_AGGREGATE_MODE_ALL,
} FirtoolPreserveAggregateMode;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FirtoolPreserveValuesMode {
  FIRTOOL_PRESERVE_VALUES_MODE_NONE,
  FIRTOOL_PRESERVE_VALUES_MODE_NAMED,
  FIRTOOL_PRESERVE_VALUES_MODE_ALL,
} FirtoolPreserveValuesMode;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FirtoolCompanionMode {
  FIRTOOL_COMPANION_MODE_BIND,
  FIRTOOL_COMPANION_MODE_INSTANTIATE,
  FIRTOOL_COMPANION_MODE_DROP,
} FirtoolCompanionMode;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FirtoolBuildMode {
  FIRTOOL_BUILD_MODE_DEBUG,
  FIRTOOL_BUILD_MODE_RELEASE,
} FirtoolBuildMode;

// NOLINTNEXTLINE(modernize-use-using)
typedef enum FirtoolRandomKind {
  FIRTOOL_RANDOM_KIND_NONE,
  FIRTOOL_RANDOM_KIND_MEM,
  FIRTOOL_RANDOM_KIND_REG,
  FIRTOOL_RANDOM_KIND_ALL,
} FirtoolRandomKind;

MLIR_CAPI_EXPORTED FirtoolOptions firtoolOptionsCreateDefault();
MLIR_CAPI_EXPORTED void firtoolOptionsDestroy(FirtoolOptions options);

#define DECLARE_FIRTOOL_OPTION(name, type)                                     \
  MLIR_CAPI_EXPORTED void firtoolOptionsSet##name(FirtoolOptions options,      \
                                                  type value);                 \
  MLIR_CAPI_EXPORTED type firtoolOptionsGet##name(FirtoolOptions options)

DECLARE_FIRTOOL_OPTION(OutputFilename, MlirStringRef);
DECLARE_FIRTOOL_OPTION(DisableAnnotationsUnknown, bool);
DECLARE_FIRTOOL_OPTION(DisableAnnotationsClassless, bool);
DECLARE_FIRTOOL_OPTION(LowerAnnotationsNoRefTypePorts, bool);
DECLARE_FIRTOOL_OPTION(PreserveAggregate, FirtoolPreserveAggregateMode);
DECLARE_FIRTOOL_OPTION(PreserveValues, FirtoolPreserveValuesMode);
DECLARE_FIRTOOL_OPTION(BuildMode, FirtoolBuildMode);
DECLARE_FIRTOOL_OPTION(DisableOptimization, bool);
DECLARE_FIRTOOL_OPTION(ExportChiselInterface, bool);
DECLARE_FIRTOOL_OPTION(ChiselInterfaceOutDirectory, MlirStringRef);
DECLARE_FIRTOOL_OPTION(VbToBv, bool);
DECLARE_FIRTOOL_OPTION(CompanionMode, FirtoolCompanionMode);
DECLARE_FIRTOOL_OPTION(DisableAggressiveMergeConnections, bool);
DECLARE_FIRTOOL_OPTION(EmitOMIR, bool);
DECLARE_FIRTOOL_OPTION(OMIROutFile, MlirStringRef);
DECLARE_FIRTOOL_OPTION(LowerMemories, bool);
DECLARE_FIRTOOL_OPTION(BlackBoxRootPath, MlirStringRef);
DECLARE_FIRTOOL_OPTION(ReplSeqMem, bool);
DECLARE_FIRTOOL_OPTION(ReplSeqMemFile, MlirStringRef);
DECLARE_FIRTOOL_OPTION(ExtractTestCode, bool);
DECLARE_FIRTOOL_OPTION(IgnoreReadEnableMem, bool);
DECLARE_FIRTOOL_OPTION(DisableRandom, FirtoolRandomKind);
DECLARE_FIRTOOL_OPTION(OutputAnnotationFilename, MlirStringRef);
DECLARE_FIRTOOL_OPTION(EnableAnnotationWarning, bool);
DECLARE_FIRTOOL_OPTION(AddMuxPragmas, bool);
DECLARE_FIRTOOL_OPTION(EmitChiselAssertsAsSVA, bool);
DECLARE_FIRTOOL_OPTION(EmitSeparateAlwaysBlocks, bool);
DECLARE_FIRTOOL_OPTION(EtcDisableInstanceExtraction, bool);
DECLARE_FIRTOOL_OPTION(EtcDisableRegisterExtraction, bool);
DECLARE_FIRTOOL_OPTION(EtcDisableModuleInlining, bool);
DECLARE_FIRTOOL_OPTION(AddVivadoRAMAddressConflictSynthesisBugWorkaround, bool);
DECLARE_FIRTOOL_OPTION(CkgModuleName, MlirStringRef);
DECLARE_FIRTOOL_OPTION(CkgInputName, MlirStringRef);
DECLARE_FIRTOOL_OPTION(CkgOutputName, MlirStringRef);
DECLARE_FIRTOOL_OPTION(CkgEnableName, MlirStringRef);
DECLARE_FIRTOOL_OPTION(CkgTestEnableName, MlirStringRef);
DECLARE_FIRTOOL_OPTION(ExportModuleHierarchy, bool);
DECLARE_FIRTOOL_OPTION(StripFirDebugInfo, bool);
DECLARE_FIRTOOL_OPTION(StripDebugInfo, bool);

#undef DECLARE_FIRTOOL_OPTION

//===----------------------------------------------------------------------===//
// Populate API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirLogicalResult
firtoolPopulatePreprocessTransforms(MlirPassManager pm, FirtoolOptions options);

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulateCHIRRTLToLowFIRRTL(
    MlirPassManager pm, FirtoolOptions options, MlirStringRef inputFilename);

MLIR_CAPI_EXPORTED MlirLogicalResult
firtoolPopulateLowFIRRTLToHW(MlirPassManager pm, FirtoolOptions options);

MLIR_CAPI_EXPORTED MlirLogicalResult
firtoolPopulateHWToSV(MlirPassManager pm, FirtoolOptions options);

MLIR_CAPI_EXPORTED MlirLogicalResult
firtoolPopulateExportVerilog(MlirPassManager pm, FirtoolOptions options,
                             MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulateExportSplitVerilog(
    MlirPassManager pm, FirtoolOptions options, MlirStringRef directory);

MLIR_CAPI_EXPORTED MlirLogicalResult
firtoolPopulateFinalizeIR(MlirPassManager pm, FirtoolOptions options);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_FIRTOOL_FIRTOOL_H
