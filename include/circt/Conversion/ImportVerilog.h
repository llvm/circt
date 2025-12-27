//===- ImportVerilog.h - Slang Verilog frontend integration ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the Verilog frontend.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_IMPORTVERILOG_H
#define CIRCT_CONVERSION_IMPORTVERILOG_H

#include "circt/Support/LLVM.h"
#include "mlir/Pass/PassOptions.h"
#include "llvm/Support/CommandLine.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
class LocationAttr;
class PassManager;
class TimingScope;
} // namespace mlir

namespace circt {

/// Options that control how Verilog input files are parsed and processed.
///
/// See `slang::driver::Driver::Options` for inspiration. Also check out
/// `Driver::addStandardArgs()` for some inspiration on how to expose these on
/// the command line.
struct ImportVerilogOptions {
  /// Limit importing to linting or parsing only.
  enum class Mode {
    /// Only lint the input, without elaboration and lowering to CIRCT IR.
    OnlyLint,
    /// Only parse and elaborate the input, without mapping to CIRCT IR.
    OnlyParse,
    /// Perform a full import and mapping to CIRCT IR.
    Full
  };
  Mode mode = Mode::Full;

  /// Generate debug information in the form of debug dialect ops in the IR.
  bool debugInfo = false;

  /// Interpret `always @(*)` as `always_comb`.
  bool lowerAlwaysAtStarAsComb = true;

  //===--------------------------------------------------------------------===//
  // Include paths
  //===--------------------------------------------------------------------===//

  /// A list of include directories in which to search for files.
  std::vector<std::string> includeDirs;

  /// A list of system include directories in which to search for files.
  std::vector<std::string> includeSystemDirs;

  /// A list of library directories in which to search for missing modules.
  std::vector<std::string> libDirs;

  /// A list of extensions that will be used to search for library files.
  std::vector<std::string> libExts;

  /// A list of extensions that will be used to exclude files.
  std::vector<std::string> excludeExts;

  /// A list of preprocessor directives to be ignored.
  std::vector<std::string> ignoreDirectives;

  //===--------------------------------------------------------------------===//
  // Preprocessing
  //===--------------------------------------------------------------------===//

  /// The maximum depth of included files before an error is issued.
  std::optional<uint32_t> maxIncludeDepth;

  /// A list of macros that should be defined in each compilation unit.
  std::vector<std::string> defines;

  /// A list of macros that should be undefined in each compilation unit.
  std::vector<std::string> undefines;

  /// If true, library files will inherit macro definitions from primary source
  /// files.
  std::optional<bool> librariesInheritMacros;

  //===--------------------------------------------------------------------===//
  // Compilation
  //===--------------------------------------------------------------------===//

  /// A string that indicates the default time scale to use for any design
  /// elements that don't specify one explicitly.
  std::optional<std::string> timeScale;

  /// If true, allow various to be referenced before they are declared.
  std::optional<bool> allowUseBeforeDeclare;

  /// If true, ignore errors about unknown modules.
  std::optional<bool> ignoreUnknownModules;

  /// If non-empty, specifies the list of modules that should serve as the top
  /// modules in the design. If empty, this will be automatically determined
  /// based on which modules are unreferenced elsewhere.
  std::vector<std::string> topModules;

  /// A list of top-level module parameters to override, of the form
  /// `<name>=<value>`.
  std::vector<std::string> paramOverrides;

  //===--------------------------------------------------------------------===//
  // Diagnostics Control
  //===--------------------------------------------------------------------===//

  /// The maximum number of errors to print before giving up.
  std::optional<uint32_t> errorLimit;

  /// A list of warning options that will be passed to the DiagnosticEngine.
  std::vector<std::string> warningOptions;

  /// A list of paths in which to suppress warnings.
  std::vector<std::string> suppressWarningsPaths;

  //===--------------------------------------------------------------------===//
  // File lists
  //===--------------------------------------------------------------------===//

  /// If set to true, all source files will be treated as part of a single
  /// compilation unit, meaning all of their text will be merged together.
  std::optional<bool> singleUnit;

  /// A list of library files to include in the compilation.
  std::vector<std::string> libraryFiles;

  /// A list of command files to process for compilation.
  std::vector<std::string> commandFiles;
};

/// Parse files in a source manager as Verilog source code and populate the
/// given MLIR `module` with corresponding ops.
mlir::LogicalResult
importVerilog(llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context,
              mlir::TimingScope &ts, mlir::ModuleOp module,
              const ImportVerilogOptions *options = nullptr);

/// Optimize and simplify the Moore dialect IR.
void populateVerilogToMoorePipeline(mlir::OpPassManager &pm);

/// Convert Moore dialect IR into core dialect IR
void populateMooreToCorePipeline(mlir::OpPassManager &pm);

/// Convert LLHD dialect IR into core dialect IR
struct LlhdToCorePipelineOptions
    : mlir::PassPipelineOptions<LlhdToCorePipelineOptions> {
  Option<bool> detectMemories{
      *this, "detect-memories",
      llvm::cl::desc("Detect memories and lower them to `seq.firmem`"),
      llvm::cl::init(true)};
  Option<bool> sroa{
      *this, "sroa",
      llvm::cl::desc("Destructure arrays and structs into individual signals. "
                     "See https://github.com/llvm/circt/issues/8804."),
      llvm::cl::init(false)};
};
void populateLlhdToCorePipeline(mlir::OpPassManager &pm,
                                const LlhdToCorePipelineOptions &options);

/// Run the files in a source manager through Slang's Verilog preprocessor and
/// emit the result to the given output stream.
mlir::LogicalResult
preprocessVerilog(llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context,
                  mlir::TimingScope &ts, llvm::raw_ostream &os,
                  const ImportVerilogOptions *options = nullptr);

/// Register the `import-verilog` MLIR translation.
void registerFromVerilogTranslation();

/// Return a human-readable string describing the slang frontend version linked
/// into CIRCT.
std::string getSlangVersion();

} // namespace circt

#endif // CIRCT_CONVERSION_IMPORTVERILOG_H
