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
                   const std::vector<const char *> &unsupportedInstructions,
                   const std::string &unsupportedInstructionsFile)
      : options(circtRtgToolOptionsCreateDefault(seed)) {
    setOutputFormat(outputFormat);
    setVerifyPasses(verifyPasses);
    setVerbosePassExecution(verbosePassExecution);
    setUnsupportedInstructions(unsupportedInstructions);
    setUnsupportedInstructionsFile(unsupportedInstructionsFile);
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
  setUnsupportedInstructions(const std::vector<const char *> &instructions) {
    circtRtgToolOptionsSetUnsupportedInstructions(
        options, instructions.size(),
        const_cast<const char **>(instructions.data()));
  }

  void addUnsupportedInstruction(const std::string &instruction) {
    circtRtgToolOptionsAddUnsupportedInstruction(options, instruction.c_str());
  }

  void setUnsupportedInstructionsFile(const std::string &filename) {
    circtRtgToolOptionsSetUnsupportedInstructionsFile(options,
                                                      filename.c_str());
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
                    const std::vector<const char *> &, const std::string &>(),
           nb::arg("seed"),
           nb::arg("output_format") = CIRCT_RTGTOOL_OUTPUT_FORMAT_ASM,
           nb::arg("verify_passes") = true,
           nb::arg("verbose_pass_execution") = false,
           nb::arg("unsupported_instructions") = std::vector<const char *>(),
           nb::arg("unsupported_instructions_file") = "")
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
           nb::arg("filename"));

  m.def("populate_randomizer_pipeline",
        [](MlirPassManager pm, const PyRtgToolOptions &options) {
          circtRtgToolRandomizerPipeline(pm, options.get());
        });
}
