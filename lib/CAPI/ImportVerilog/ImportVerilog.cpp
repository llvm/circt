//===- ImportVerilog.cpp - C Interface to ImportVerilog -------------------===//
//
//  Implements a C Interface for importing Verilog.
//
//===----------------------------------------------------------------------===//

#include "circt-c/ImportVerilog.h"

#include "circt/Conversion/ImportVerilog.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Support/Timing.h"
#include "llvm/Support/SourceMgr.h"

using namespace circt;
using namespace mlir;

namespace {
std::optional<ImportVerilogOptions>
convertOptions(const MlirImportVerilogOptions *c) {
  if (c == nullptr) {
    return std::nullopt;
  }

  ImportVerilogOptions ret;

  switch (c->mode) {
  case MLIR_IMPORT_VERILOG_MODE_ONLY_LINT:
    ret.mode = ImportVerilogOptions::Mode::OnlyLint;
    break;
  case MLIR_IMPORT_VERILOG_MODE_ONLY_PARSE:
    ret.mode = ImportVerilogOptions::Mode::OnlyParse;
    break;
  case MLIR_IMPORT_VERILOG_MODE_FULL:
    ret.mode = ImportVerilogOptions::Mode::Full;
    break;
  default:
    llvm_unreachable("unknown mode for importing verilog");
  }
  ret.debugInfo = c->debugInfo;
  ret.lowerAlwaysAtStarAsComb = c->lowerAlwaysAtStarAsComb;

  const auto toStrs = [](const char **strs, size_t num) {
    std::vector<std::string> vec;
    vec.reserve(num);
    for (size_t i = 0; i < num; ++i) {
      vec.emplace_back(strs[i]);
    }
    return vec;
  };

  ret.includeDirs = toStrs(c->includeDirs, c->includeDirsNum);
  ret.includeSystemDirs = toStrs(c->includeSystemDirs, c->includeSystemDirsNum);
  ret.libDirs = toStrs(c->libDirs, c->libDirsNum);
  ret.libExts = toStrs(c->libExts, c->libExtsNum);
  ret.excludeExts = toStrs(c->excludeExts, c->excludeExtsNum);
  ret.ignoreDirectives = toStrs(c->ignoreDirectives, c->ignoreDirectivesNum);

  ret.maxIncludeDepth =
      c->maxIncludeDepth == std::numeric_limits<uint32_t>::max()
          ? std::nullopt
          : std::optional{c->maxIncludeDepth};
  ret.defines = toStrs(c->defines, c->definesNum);
  ret.undefines = toStrs(c->undefines, c->undefinesNum);
  ret.librariesInheritMacros = c->librariesInheritMacros;

  ret.timeScale =
      c->timeScale == nullptr ? std::nullopt : std::optional{c->timeScale};
  ret.allowUseBeforeDeclare = c->allowUseBeforeDeclare;
  ret.ignoreUnknownModules = c->ignoreUnknownModules;
  ret.topModules = toStrs(c->topModules, c->topModulesNum);
  ret.paramOverrides = toStrs(c->paramOverrides, c->paramOverridesNum);

  ret.errorLimit = c->errorLimit == std::numeric_limits<uint32_t>::max()
                       ? std::nullopt
                       : std::optional{c->errorLimit};
  ret.warningOptions = toStrs(c->warningOptions, c->warningOptionsNum);
  ret.suppressWarningsPaths =
      toStrs(c->suppressWarningsPaths, c->suppressWarningsPathsNum);

  ret.singleUnit = c->singleUnit;
  ret.libraryFiles = toStrs(c->libraryFiles, c->libraryFilesNum);
  return ret;
}
} // namespace

MlirLogicalResult mlirImportVerilog(MlirContext context, MlirModule into,
                                    LLVMMemoryBufferRef *buffers,
                                    size_t numBuffers,
                                    const MlirImportVerilogOptions *options) {
  auto ctx = unwrap(context);
  TimingScope ts;
  llvm::SourceMgr sourceMgr;
  for (size_t i = 0; i < numBuffers; i++) {
    auto buffer = std::unique_ptr<llvm::MemoryBuffer>{llvm::unwrap(buffers[i])};
    sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc{});
  }
  auto opts = convertOptions(options);
  return wrap(
      importVerilog(sourceMgr, ctx, ts, unwrap(into),
                    opts.has_value() ? std::addressof(*opts) : nullptr));
}
