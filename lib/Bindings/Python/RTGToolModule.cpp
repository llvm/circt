//===- RTGToolModule.cpp - RTG Tool API nanobind module -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/RtgTool.h"

#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include <nanobind/nanobind.h>
namespace nb = nanobind;

using namespace circt;

namespace {
/// Wrapper around an CirctRtgToolOptions.
class PyRtgToolOptions {
public:
  PyRtgToolOptions(unsigned seed, CirctRtgToolOutputFormat outputFormat,
                   bool verifyPasses, bool verbosePassExecution,
                   const std::vector<std::string> &unsupportedInstructions,
                   const std::string &unsupportedInstructionsFile,
                   bool splitOutput, const std::string &outputPath,
                   bool memoriesAsImmediates)
      : options(circtRtgToolOptionsCreateDefault(seed)) {
    setOutputFormat(outputFormat);
    setVerifyPasses(verifyPasses);
    setVerbosePassExecution(verbosePassExecution);
    setUnsupportedInstructions(unsupportedInstructions);
    setUnsupportedInstructionsFile(unsupportedInstructionsFile);
    setSplitOutput(splitOutput);
    setOutputPath(outputPath);
    setMemoriesAsImmediates(memoriesAsImmediates);
  }
  ~PyRtgToolOptions() { circtRtgToolOptionsDestroy(options); }

  operator CirctRtgToolOptions() const { return options; }
  CirctRtgToolOptions get() const { return options; }

  void setOutputFormat(CirctRtgToolOutputFormat format) {
    circtRtgToolOptionsSetOutputFormat(options, format);
  }

  void setSeed(unsigned seed) { circtRtgToolOptionsSetSeed(options, seed); }

  void setVerifyPasses(bool enable) {
    circtRtgToolOptionsSetVerifyPasses(options, enable);
  }

  void setVerbosePassExecution(bool enable) {
    circtRtgToolOptionsSetVerbosePassExecution(options, enable);
  }

  void
  setUnsupportedInstructions(const std::vector<std::string> &instructions) {
    circtRtgToolOptionsSetUnsupportedInstructions(
        options, instructions.size(),
        reinterpret_cast<const void **>(
            const_cast<std::string *>(instructions.data())));
  }

  void addUnsupportedInstruction(const std::string &instruction) {
    circtRtgToolOptionsAddUnsupportedInstruction(options, instruction.c_str());
  }

  void setUnsupportedInstructionsFile(const std::string &filename) {
    circtRtgToolOptionsSetUnsupportedInstructionsFile(options,
                                                      filename.c_str());
  }

  void setSplitOutput(bool enable) {
    circtRtgToolOptionsSetSplitOutput(options, enable);
  }

  void setOutputPath(const std::string &path) {
    circtRtgToolOptionsSetOutputPath(options, path.c_str());
  }

  void setMemoriesAsImmediates(bool enable) {
    circtRtgToolOptionsSetMemoriesAsImmediates(options, enable);
  }

private:
  CirctRtgToolOptions options;
};
} // namespace

/// Populate the rtgtool python module.
void circt::python::populateDialectRTGToolSubmodule(nb::module_ &m) {
  m.doc() = "RTGTool Python native extension";

  nb::enum_<CirctRtgToolOutputFormat>(m, "OutputFormat")
      .value("MLIR", CIRCT_RTGTOOL_OUTPUT_FORMAT_MLIR)
      .value("ELABORATED_MLIR", CIRCT_RTGTOOL_OUTPUT_FORMAT_ELABORATED_MLIR)
      .value("ASM", CIRCT_RTGTOOL_OUTPUT_FORMAT_ASM);

  nb::class_<PyRtgToolOptions>(m, "Options")
      .def(nb::init<unsigned, CirctRtgToolOutputFormat, bool, bool,
                    const std::vector<std::string> &, const std::string &, bool,
                    const std::string &, bool>(),
           nb::arg("seed"),
           nb::arg("output_format") = CIRCT_RTGTOOL_OUTPUT_FORMAT_ASM,
           nb::arg("verify_passes") = true,
           nb::arg("verbose_pass_execution") = false,
           nb::arg("unsupported_instructions") = std::vector<const char *>(),
           nb::arg("unsupported_instructions_file") = "",
           nb::arg("split_output") = false, nb::arg("output_path") = "",
           nb::arg("memories_as_immediates") = true)
      .def("set_output_format", &PyRtgToolOptions::setOutputFormat,
           "Specify the output format of the tool", nb::arg("format"))
      .def("set_seed", &PyRtgToolOptions::setSeed,
           "Specify the seed for all RNGs used in the tool", nb::arg("seed"))
      .def("set_verify_passes", &PyRtgToolOptions::setVerifyPasses,
           "Specify whether the verifiers should be run after each pass",
           nb::arg("enable"))
      .def("set_verbose_pass_execution",
           &PyRtgToolOptions::setVerbosePassExecution,
           "Specify whether passes should run in verbose mode",
           nb::arg("enable"))
      .def("set_unsupported_instructions",
           &PyRtgToolOptions::setUnsupportedInstructions,
           "Set the list of of instructions unsupported by the assembler",
           nb::arg("instructions"))
      .def("add_unsupported_instruction",
           &PyRtgToolOptions::addUnsupportedInstruction,
           "Add the instruction given by name to the list of instructions not "
           "supported by the assembler",
           nb::arg("instruction_name"))
      .def("set_unsupported_instructions_file",
           &PyRtgToolOptions::setUnsupportedInstructionsFile,
           "Register a file containing a comma-separated list of instruction "
           "names which are not supported by the assembler.",
           nb::arg("filename"))
      .def("set_split_output", &PyRtgToolOptions::setSplitOutput,
           "Determines whether each test should be emitted to a separate file.",
           nb::arg("enable"))
      .def("output_path", &PyRtgToolOptions::setOutputPath,
           "The path of a file to be emitted to or a directory if "
           "'split_output' is enabled.",
           nb::arg("filename"))
      .def("set_memories_as_immediates",
           &PyRtgToolOptions::setMemoriesAsImmediates,
           "Determines whether memories are lowered to immediates or labels.",
           nb::arg("enable"));

  m.def("populate_randomizer_pipeline",
        [](MlirPassManager pm, const PyRtgToolOptions &options) {
          circtRtgToolRandomizerPipeline(pm, options.get());
        });
}
