//===-- RTGPasses.td - RTG pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that operate on the RTG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_TRANSFORMS_RTGPASSES_TD
#define CIRCT_DIALECT_RTG_TRANSFORMS_RTGPASSES_TD

include "mlir/Pass/PassBase.td"

def ElaborationPass : Pass<"rtg-elaborate", "mlir::ModuleOp"> {
  let summary = "elaborate the randomization parts";
  let description = [{
    This pass interprets most RTG operations to perform the represented
    randomization and in the process get rid of those operations. This means,
    after this pass the IR does not contain any random constructs within tests
    anymore.
  }];

  let options = [
    Option<"seed", "seed", "unsigned", /*default=*/"",
      "The seed for any RNG constructs used in the pass.">,
  ];

  let dependentDialects = ["mlir::index::IndexDialect"];
}

def EmitRTGISAAssemblyPass : Pass<"rtg-emit-isa-assembly", "mlir::ModuleOp"> {
  let summary = "Elaborate the contexts of RTG";
  let description = [{
    Emits all 'rtg.test's in the IR in a format understood by assemblers.

    There are two options to specify lists of instructions that are not
    supported by the assembler. For instructions in any of those lists, this
    pass will emit the equivalent binary representation.

    This pass operates on the `InstructionOpInterface` and folds constant-like
    operations to support downstream dialects.
  }];

  let options = [
    Option<"splitOutput", "split-output", "bool", /*default=*/"false",
      "If 'true' emits one file per 'rtg.test' in the IR. The name of the "
      "file matches the test name and is placed in 'path'. Otherwise, path "
      "is interpreted as the full file path including filename.">,
    Option<"path", "path", "std::string", /*default=*/"",
      "The directory or file path in which the output files should be "
      "created. If empty is is emitted to stderr (not allowed if "
      "'split-output' is set to 'true')">,
    Option<"unsupportedInstructionsFile", "unsupported-instructions-file",
      "std::string", /*default=*/"",
      "An absolute path to a file with a list of instruction names not "
      "supported by the assembler.">,
    ListOption<"unsupportedInstructions", "unsupported-instructions",
      "std::string",
      "A list of ISA instruction names not supported by the assembler.">,
  ];
}

#endif // CIRCT_DIALECT_RTG_TRANSFORMS_RTGPASSES_TD
