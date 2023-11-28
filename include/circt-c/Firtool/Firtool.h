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

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(CirctFirtoolFirtoolOptions, void);

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

MLIR_CAPI_EXPORTED CirctFirtoolFirtoolOptions
circtFirtoolOptionsCreateDefault();
MLIR_CAPI_EXPORTED void
circtFirtoolOptionsDestroy(CirctFirtoolFirtoolOptions options);

MLIR_CAPI_EXPORTED void
circtFirtoolOptionsSetOutputFilename(CirctFirtoolFirtoolOptions options,
                                     MlirStringRef filename);
MLIR_CAPI_EXPORTED void
circtFirtoolOptionsDisableUnknownAnnotations(CirctFirtoolFirtoolOptions options,
                                             bool disable);

//===----------------------------------------------------------------------===//
// Populate API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulatePreprocessTransforms(
    MlirPassManager pm, CirctFirtoolFirtoolOptions options);

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulateCHIRRTLToLowFIRRTL(
    MlirPassManager pm, CirctFirtoolFirtoolOptions options,
    MlirStringRef inputFilename);

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulateLowFIRRTLToHW(
    MlirPassManager pm, CirctFirtoolFirtoolOptions options);

MLIR_CAPI_EXPORTED MlirLogicalResult
firtoolPopulateHWToSV(MlirPassManager pm, CirctFirtoolFirtoolOptions options);

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulateExportVerilog(
    MlirPassManager pm, CirctFirtoolFirtoolOptions options,
    MlirStringCallback callback, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulateExportSplitVerilog(
    MlirPassManager pm, CirctFirtoolFirtoolOptions options,
    MlirStringRef directory);

MLIR_CAPI_EXPORTED MlirLogicalResult firtoolPopulateFinalizeIR(
    MlirPassManager pm, CirctFirtoolFirtoolOptions options);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_FIRTOOL_FIRTOOL_H
