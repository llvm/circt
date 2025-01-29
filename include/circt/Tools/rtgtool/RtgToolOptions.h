//===- RtgToolOptions.h - Configuration Options for rtgtool -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines configuration options for the rtgtool.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_RTGTOOL_RTGTOOLOPTIONS_H
#define CIRCT_TOOLS_RTGTOOL_RTGTOOLOPTIONS_H

#include "circt/Support/LLVM.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
namespace rtg {

/// The set of options used to control the behavior of the RTG tool.
class RtgToolOptions {
public:
  enum class OutputFormat { MLIR, ElaboratedMLIR, ASM };

  RtgToolOptions(unsigned seed) : seed(seed) {}

  void setOutputFormat(OutputFormat format) { outputFormat = format; }
  OutputFormat getOutputFormat() const { return outputFormat; }

  RtgToolOptions &setSeed(unsigned seed) {
    this->seed = seed;
    return *this;
  }
  unsigned getSeed() const { return seed; }

  RtgToolOptions &setVerifyPasses(bool enable) {
    verifyPasses = enable;
    return *this;
  }
  bool getVerifyPasses() const { return verifyPasses; }

  RtgToolOptions &setVerbosePassExecution(bool enable) {
    verbosePassExecution = enable;
    return *this;
  }
  bool getVerbosePassExecution() const { return verbosePassExecution; }

  RtgToolOptions &setUnsupportedInstructions(SmallVector<std::string> &&instr) {
    unsupportedInstructions = instr;
    return *this;
  }
  RtgToolOptions &setUnsupportedInstructions(ArrayRef<std::string> instr) {
    unsupportedInstructions = SmallVector<std::string>(instr);
    return *this;
  }
  RtgToolOptions &addUnsupportedInstruction(const std::string &instr) {
    unsupportedInstructions.push_back(instr);
    return *this;
  }
  ArrayRef<std::string> getUnsupportedInstructions() const {
    return unsupportedInstructions;
  }

  RtgToolOptions &setUnsupportedInstructionsFile(StringRef filename) {
    unsupportedInstructionsFile = filename;
    return *this;
  }
  std::string getUnsupportedInstructionsFile() const {
    return unsupportedInstructionsFile;
  }

  RtgToolOptions &setSplitOutput(bool enable) {
    splitOutput = enable;
    return *this;
  }
  bool getSplitOutput() const { return splitOutput; }

  RtgToolOptions &setOutputPath(StringRef path) {
    outputPath = path;
    return *this;
  }
  std::string getOutputPath() const { return outputPath; }

private:
  OutputFormat outputFormat = OutputFormat::ElaboratedMLIR;
  unsigned seed;
  bool verifyPasses = true;
  bool verbosePassExecution = false;
  SmallVector<std::string> unsupportedInstructions;
  std::string unsupportedInstructionsFile;
  bool splitOutput = false;
  std::string outputPath;
};

/// Populates the passes necessary to lower IR with RTG randomization operations
/// to fully elaborated IR (i.e., IR without random constructs).
void populateRandomizerPipeline(mlir::PassManager &pm,
                                const RtgToolOptions &options);

} // namespace rtg
} // namespace circt

#endif // CIRCT_TOOLS_RTGTOOL_RTGTOOLOPTIONS_H
