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

  let options = [Option<"seed", "seed", "unsigned", /*default=*/"",
                        "The seed for any RNG constructs used in the pass.">,
                 Option<
                     "deleteUnmatchedTests", "delete-unmatched-tests", "bool",
                     /*default=*/"true",
                     "Delete tests that could not be matched with a target.">,
  ];

  let dependentDialects = ["mlir::index::IndexDialect"];
}

def EmitRTGISAAssemblyPass : Pass<"rtg-emit-isa-assembly", "emit::FileOp"> {
  let summary = "emits the instructions in a format understood by assemblers";
  let description = [{
    This pass expects all instructions to be inside 'emit.file' operations with
    an appropriate filename attribute. There are two special filenames:
    - "-" means that the output should be emitted to stdout.
    - "" means that the output should be emitted to stderr.

    In order to operate on 'emit.file' operations in parallel, the pass
    requires that all 'emit.file' operations have a unique filename (this is not
    checked by the pass and violations will result in race conditions).

    There are two options to specify lists of instructions that are not
    supported by the assembler. For instructions in any of those lists, this
    pass will emit the equivalent binary representation.
  }];

  let options = [
    Option<"unsupportedInstructionsFile", "unsupported-instructions-file",
      "std::string", /*default=*/"",
      "An absolute path to a file with a list of instruction names not "
      "supported by the assembler.">,
    ListOption<"unsupportedInstructions", "unsupported-instructions",
      "std::string",
      "A list of ISA instruction names not supported by the assembler.">,
  ];
}

def InlineSequencesPass : Pass<"rtg-inline-sequences", "mlir::ModuleOp"> {
  let summary = "inline and interleave sequences";
  let description = [{
    Inline all sequences into tests and remove the 'rtg.sequence' operations.
    Also computes and materializes all interleaved sequences
    ('interleave_sequences' operation).
  }];

  let statistics = [
    Statistic<"numSequencesInlined", "num-sequences-inlined",
      "Number of sequences inlined into another sequence or test.">,
    Statistic<"numSequencesInterleaved", "num-sequences-interleaved",
      "Number of sequences interleaved with another sequence.">,
  ];
}

def LinearScanRegisterAllocationPass : Pass<
    "rtg-linear-scan-register-allocation"> {

  let summary = "simple linear scan register allocation for RTG";
  let description = [{
    Performs a simple version of the linear scan register allocation algorithm
    based on the 'rtg.virtual_reg' operations.

    This pass is expected to be run after elaboration.
  }];

  let statistics = [
    Statistic<"numRegistersSpilled", "num-registers-spilled",
      "Number of registers spilled to the stack.">,
  ];
}

def LowerUniqueLabelsPass : Pass<"rtg-lower-unique-labels"> {
  let summary = "lower label_unique_decl to label_decl operations";
  let description = [{
    This pass lowers label_unique_decl operations to label_decl operations by
    creating a unique label string based on all the existing unique and
    non-unique label declarations in the module.
  }];

  let statistics = [
    Statistic<"numLabelsLowered", "num-labels-lowered",
      "Number of unique labels lowered to regular label declarations.">,
  ];
}

def UniqueValidateOpsPass : Pass<"rtg-unique-validate"> {
  let summary = "compute unique IDs for validate operations";
  let description = [{
    This pass visits all 'rtg.validate' operations without an ID attribute and
    assigns a unique ID to them.
  }];
}

def MemoryAllocationPass : Pass<"rtg-memory-allocation", "rtg::TestOp"> {
  let summary = "lower memories to immediates or labels";
  let description = [{
    This pass lowers 'memory_alloc' and other memory handling operations to
    immediates or labels by computing offsets within memory blocks according to
    the memory allocation's size and alignments.
  }];

  let options = [
    Option<"useImmediates", "use-immediates", "bool", /*default=*/"true",
           "Whether the pass should lower memories to immediates instead of "
           "labels.">,
  ];

  let statistics = [
    Statistic<"numMemoriesAllocated", "num-memories-allocated",
              "Number of memories allocated from memory blocks.">,
  ];
}

def EmbedValidationValuesPass : Pass<"rtg-embed-validation-values",
                                     "mlir::ModuleOp"> {
  let summary = "lower validate operations to the externally provided values";
  let description = [{
    This pass replaces 'rtg.validate' operations with concrete values read from
    an external file. The file should contain the expected validation values
    with matching IDs computed, e.g., by running the program in a simulator.

    Each validate operation is matched with its corresponding value using the
    unique identifier. If an identifier occurs multiple times in the file, the
    pass fails. If an identifier for a 'validate' operation is missing in the
    file, the pass will not modify that operation. Otherwise, the values are
    parsed according to the implementation provided by the
    'ValidationTypeInterface' implementation of the 'ref' operand type and
    materialized in the IR as constants to replace the 'validate' operation.

    This pass is typically used as part of a two-phase compilation process that
    forks after the 'rtg-unique-valiate' pass is run:
    1. Run the 'rtg-lower-validate-ops-to-labels' pass and the rest of the
       pipeline, then simulate the output in a reference simulator to generate
       a file with the expected values.
    2. Run this pass and the rest of the pipeline to produce the final test.
  }];

  let options = [
    Option<"filename", "filename", "std::string", "",
           "The file with the validation values.">,
  ];
}

def LowerValidateToLabelsPass : Pass<"rtg-lower-validate-to-labels"> {
  let summary = "lower validation operations to intrinsic labels";
  let description = [{
    Lowers the 'rtg.validate' operations to special intrinsic labels understood
    by the target simulator to print the register contents.
  }];
}

def PrintTestNamesPass : Pass<"rtg-print-test-names", "mlir::ModuleOp"> {
  let summary = "print the names of all tests to the given file";
  let description = [{
    Prints the names of all tests in the module to the given file.
    A CSV format is used with the first column being the properly uniqued name
    of the test and the second column being the original name of the test as it
    appeared in the frontend. The original name of a test may occur several
    times because the test might have been duplicated for multiple targets or
    because multiple elaborations of the same test/target pair were requested.
  }];

  let options = [
    Option<"filename", "filename", "std::string", "\"-\"",
           "The file to print the test names to.">,
  ];
}

def InsertTestToFileMappingPass : Pass<"rtg-insert-test-to-file-mapping",
                                       "mlir::ModuleOp"> {
  let summary = "insert Emit dialect ops to prepare for emission";
  let description = [{
    This pass inserts emit dialect operations to group tests to output files.
    All tests can be put in a single output file, each test in its own file, or
    tests can be grouped according to some properties (e.g., machine mode vs.
    user mode tests) (TODO).
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
  ];

  let dependentDialects = ["emit::EmitDialect"];
}

def SimpleTestInlinerPass : Pass<"rtg-simple-test-inliner", "mlir::ModuleOp"> {
  let summary = "inline test contents";
  let description = [{
    This is a simple pass to inline test contents into 'emit.file' operations
    in which they are referenced. No "glue code" is inserted between tests
    added to the same file. Thus this pass is not intended to be used in a
    production pipeline but just to bring the IR into a structure understood by
    the RTG ISA assembly emission pass to avoid making that pass more complex.
  }];
}

#endif // CIRCT_DIALECT_RTG_TRANSFORMS_RTGPASSES_TD
