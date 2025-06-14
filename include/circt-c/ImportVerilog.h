//===-- circt-c/ImportVerilog.h - C API for importing Verilog -----*- C -*-===//
//
// This header declares the C interface for importing Verilog from Verilog
// source files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_IMPORTVERILOG_H
#define CIRCT_C_IMPORTVERILOG_H

#include "mlir-c/IR.h"
#include "llvm-c/Types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MlirImportVerilogMode {
  MLIR_IMPORT_VERILOG_MODE_ONLY_LINT,
  MLIR_IMPORT_VERILOG_MODE_ONLY_PARSE,
  MLIR_IMPORT_VERILOG_MODE_FULL,
} MlirImportVerilogMode;

typedef struct MlirImportVerilogOptions {
  MlirImportVerilogMode mode;
  bool debugInfo;
  bool lowerAlwaysAtStarAsComb;

  // Include paths
  const char **includeDirs;
  size_t includeDirsNum;
  const char **includeSystemDirs;
  size_t includeSystemDirsNum;
  const char **libDirs;
  size_t libDirsNum;
  const char **libExts;
  size_t libExtsNum;
  const char **excludeExts;
  size_t excludeExtsNum;
  const char **ignoreDirectives;
  size_t ignoreDirectivesNum;

  // Preprocessing
  uint32_t maxIncludeDepth; // UINT32_MAX for no limit
  const char **defines;
  size_t definesNum;
  const char **undefines;
  size_t undefinesNum;
  bool librariesInheritMacros;

  // Compilation
  const char *timeScale; // Optional
  bool allowUseBeforeDeclare;
  bool ignoreUnknownModules;
  const char **topModules;
  size_t topModulesNum;
  const char **paramOverrides;
  size_t paramOverridesNum;

  // Diagnostics Control
  uint32_t errorLimit; // UINT32_MAX for no limit
  const char **warningOptions;
  size_t warningOptionsNum;
  const char **suppressWarningsPaths;
  size_t suppressWarningsPathsNum;

  // File lists
  bool singleUnit;
  const char **libraryFiles;
  size_t libraryFilesNum;
} MlirImportVerilogOptions;

/// Parse the specified Verilog inputs into the specified MLIR context.
///
/// It takes the ownership of buffers.
MLIR_CAPI_EXPORTED MlirLogicalResult mlirImportVerilog(
    MlirContext context, MlirModule into, LLVMMemoryBufferRef *buffers,
    size_t numBuffers, const MlirImportVerilogOptions *options /* Optional */);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_IMPORTVERILOG_H
