//===-- Passes.td - FIRRTL pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_PASSES_TD
#define CIRCT_DIALECT_FIRRTL_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerFIRRTLAnnotations : Pass<"firrtl-lower-annotations", "firrtl::CircuitOp"> {
  let summary = "Lower FIRRTL annotations to usable entities";
  let description = [{
    Lower FIRRTL annotations to usable forms.  FIRRTL annotations are a big bag
    of semi-structured, irregular json.  This pass normalizes all supported
    annotations and annotation paths.
  }];
  let constructor = "circt::firrtl::createLowerFIRRTLAnnotationsPass()";
  let options = [
    Option<"ignoreAnnotationClassless", "disable-annotation-classless", "bool", "false",
      "Ignore classless annotations.">,
    Option<"ignoreAnnotationUnknown", "disable-annotation-unknown", "bool", "true",
      "Ignore unknown annotations.">
  ];
  let dependentDialects = ["hw::HWDialect"];
}

def LowerFIRRTLTypes : Pass<"firrtl-lower-types", "firrtl::CircuitOp"> {
  let summary = "Lower FIRRTL types to ground types";
  let description = [{
    Lower aggregate FIRRTL types to ground types.  Memories, ports, wires, etc
    are split appart by elements of aggregate types.  The only aggregate types
    which exist after this pass are memory ports, though memory data types are
    split.

    Connect and partial connect expansion and canonicalization happen in this
    pass.
  }];
  let constructor = "circt::firrtl::createLowerFIRRTLTypesPass()";
  let options = [
    Option<"flattenAggregateMemData", "flatten-mem", "bool", "false",
           "Concat all elements of the aggregate data into a single element.">,
    Option<"preserveAggregate", "preserve-aggregate", "bool", "false",
           "Preserve passive aggregate types in the module.">,
    Option<"preservePublicTypes", "preserve-public-types", "bool",
           "true", "Force to lower ports of toplevel and external modules even"
           "when aggregate preservation mode.">
  ];
  let dependentDialects = ["hw::HWDialect"];
}

def IMConstProp : Pass<"firrtl-imconstprop", "firrtl::CircuitOp"> {
  let summary = "Intermodule constant propagation and dead code elimination";
  let description = [{
    Use optimistic constant propagation to delete ports and unreachable IR.
  }];
  let constructor = "circt::firrtl::createIMConstPropPass()";
  let statistics = [
    Statistic<"numFoldedOp", "num-folded-op", "Number of operations folded">,
    Statistic<"numErasedOp", "num-erased-op", "Number of operations erased">
  ];
}

def RemoveUnusedPorts : Pass<"firrtl-remove-unused-ports", "firrtl::CircuitOp"> {
  let summary = "Remove unused ports";
  let description = [{
    This pass removes unused ports without annotations or symbols. Implementation
    wise, this pass iterates over the instance graph in a topological order from
    leaves to the top so that we can remove unused ports optimally.
  }];
  let constructor = "circt::firrtl::createRemoveUnusedPortsPass()";
  let statistics = [
    Statistic<"numRemovedPorts", "num-removed-ports", "Number of ports erased">,
  ];
}

def Inliner : Pass<"firrtl-inliner", "firrtl::CircuitOp"> {
  let summary = "Performs inlining, flattening, and dead module elimination";
  let description = [{
    This inliner pass will inline any instance of module marked as inline, and
    recursively inline all instances inside of a module marked with flatten.
    This pass performs renaming of every entity with a name that is inlined by
    prefixing it with the instance name.  This pass also will remove any module
    which is not reachable from the top level module.

    The inline and flatten annotation attributes are attached to module
    definitions, and they are:
    ```mlir
      {class = "firrtl.passes.InlineAnnotation"}
      {class = "firrtl.transforms.FlattenAnnotation"}
    ```
  }];
  let constructor = "circt::firrtl::createInlinerPass()";
}

def BlackBoxMemory : Pass<"firrtl-blackbox-memory", "firrtl::CircuitOp"> {
  let summary = "Replace all FIRRTL memories with an external module black box.";
  let description = [{
    This pass replaces all sequential memory operations with an external module
    black box.  For each memory operation, it creates a FModuleOp which
    replicates the return types of the memory operation, and replaces the
    MemoryOp with an instance of the module.  Inside the new module, an
    instance of an FExtModule black box is created.  The black box module must
    use the same parameter naming conventions used by the ReplaceSeqMemories
    pass in the Scala FIRRTL compiler.
  }];
  let constructor = "circt::firrtl::createBlackBoxMemoryPass()";
  let options = [
    Option<"emitWrapper", "emit-wrapper", "bool", "true",
           "Create a wrapper module around the black box external module.">
  ];
}

def CreateSiFiveMetadata : Pass<"firrtl-emit-metadata", "firrtl::CircuitOp"> {
  let summary = "Emit metadata of the FIRRTL modules";
  let description = [{
    This pass handles the emission of several different kinds of metadata.
  }];
  let constructor = "circt::firrtl::createCreateSiFiveMetadataPass()";
  let options = [Option<"replSeqMem", "repl-seq-mem", "bool", "false",
      "Lower the seq mem for macro replacement and emit relevant metadata">,
    Option<"replSeqMemCircuit", "repl-seq-mem-circuit", "std::string", "",
      "Circuit root for seq mem metadata">,
    Option<"replSeqMemFile", "repl-seq-mem-file", "std::string", "",
      "File to which emit seq meme metadata">
  ];
  let dependentDialects = ["hw::HWDialect"];
}

def WireDFT : Pass<"firrtl-dft", "firrtl::CircuitOp"> {
  let summary = "Wires test enables to clock gates for DFT infrastructure";
  let description = [{
    This pass will take a 1-bit signal targeted by
    `DFTTestModeEnableAnnotation` and wires it to the `test_en` port of every
    module named `EICG_wrapper`. This will create ports in any intermediate
    module on the path from the signal to the `EICG_wrapper` modules.  This
    pass is used to enable the "Design For Testing" style of design when the
    intermediate modules were not originally built with DFT in mind.
  }];
  let constructor = "circt::firrtl::createWireDFTPass()";
}

def EmitOMIR : Pass<"firrtl-emit-omir", "firrtl::CircuitOp"> {
  let summary = "Emit OMIR annotations";
  let description = [{
    This pass gathers the `OMIRAnnotation`s in the design, updates the contained
    targets with the trackers that were scattered throughout the design upon
    reading the OMIR, and serializes the resulting data into a JSON file.
  }];
  let constructor = "circt::firrtl::createEmitOMIRPass()";
  let options = [Option<"outputFilename", "file", "std::string", "",
      "Output file for the JSON-serialized OMIR data">];
  let dependentDialects = ["sv::SVDialect", "hw::HWDialect"];
}

def ExpandWhens : Pass<"firrtl-expand-whens", "firrtl::FModuleOp"> {
  let summary = "Remove all when conditional blocks.";
  let description = [{
    This pass will:
    1. Resolve last connect semantics.
    2. Remove all when operations.

    When a wire has multiple connections, only the final connection is used,
    all previous connections are overwritten. When there is a conditional
    connect, the previous connect is only overwritten when the condition
    holds:

    ```firrtl
    w <= a
    when c :
      w <= b

    ; Equivalent to:
    w <= mux(c, b, a)
    ```

    This pass requires that all connects are expanded.
  }];
  let constructor = "circt::firrtl::createExpandWhensPass()";
}

def LowerCHIRRTLPass : Pass<"firrtl-lower-chirrtl", "firrtl::FModuleOp"> {
  let summary = "Infer the memory ports of SeqMem and CombMem";
  let description = [{
      This pass finds the CHIRRTL behavioral memories and their ports, and
      transforms them into standard FIRRTL memory operations.  For each
      `seqmem` or `combmem`, a new memory is created.  For every `memoryport`
      operation using a CHIRRTL memory, a memory port is defined on the
      new standard memory.

      The direction or kind of the port is inferred from how each of the memory
      ports is used in the IR. If a memory port is only written to, it becomes
      a `Write` port.  If a memory port is only read from, it become a `Read`
      port.  If it is used both ways, it becomes a `ReadWrite` port.

      `Write`, `ReadWrite` and combinational `Read` ports are disabled by
      default, but then enabled when the CHIRRTL memory port is declared.
      Sequential `Read` ports have more complicated enable inference:

      1. If a wire or register is used as the index of the memory port, then
         the memory is enabled whenever a non-invalid value is driven to the
         address. 
      2. If a node is used as the index of the memory port, then the memory is
         enabled at the declaration of the node.
      3. In all other cases, the memory is never enabled.

      In the first two cases, they can easily produce a situation where we try
      to enable the memory before it is even declared. This produces a
      compilation error.

  }];
  let constructor = "circt::firrtl::createLowerCHIRRTLPass()";
}

def InferWidths : Pass<"firrtl-infer-widths", "firrtl::CircuitOp"> {
  let summary = "Infer the width of types";
  let description = [{
    This pass infers the widths of all types throughout a FIRRTL module, and
    emits diagnostics for types that could not be inferred.
  }];
  let constructor = "circt::firrtl::createInferWidthsPass()";
}

def InferResets : Pass<"firrtl-infer-resets", "firrtl::CircuitOp"> {
  let summary = "Infer reset synchronicity and add implicit resets";
  let description = [{
    This pass infers whether resets are synchronous or asynchronous, and extends
    reset-less registers with an asynchronous reset based on the following
    annotations:

    - `sifive.enterprise.firrtl.FullAsyncResetAnnotation`
    - `sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation`
  }];
  let constructor = "circt::firrtl::createInferResetsPass()";
}

def BlackBoxReader : Pass<"firrtl-blackbox-reader", "CircuitOp"> {
  let summary = "Load source files for black boxes into the IR";
  let description = [{
    This pass handles reads the Verilog source files for black boxes and adds
    them as `sv.verbatim.file` operations into the IR. Later passes can then
    write these files back to disk to ensure that they can be accessed by other
    tools down the line in a well-known location. Supports inline, resource, and
    path annotations for black box source files.

    The supported `firrtl.circuit` annotations are:

    - `{class = "firrtl.transforms.BlackBoxTargetDirAnno", targetDir = "..."}`
      Overrides the target directory into which black box source files are
      emitted.
    - `{class = "firrtl.transforms.BlackBoxResourceFileNameAnno", resourceFileName = "xyz.f"}`
      Specifies the output file name for the list of black box source files that
      is generated as a collateral of the pass.

    The supported `firrtl.extmodule` annotations are:

    - ```
      {
        class = "firrtl.transforms.BlackBoxInlineAnno",
        name = "myfile.v",
        text = "..."
      }
      ```
      Specifies the black box source code (`text`) inline. Generates a file with
      the given `name` in the target directory.
    - ```
      {
        class = "firrtl.transforms.BlackBoxPathAnno",
        path = "myfile.v"
      }
      ```
      Specifies the file `path` as source code for the module. Copies the file
      to the target directory.
    - ```
      {
        class = "firrtl.transforms.BlackBoxResourceAnno",
        resourceId = "myfile.v"
      }
      ```
      Specifies the file `path` as source code for the module. In contrast to
      the `BlackBoxPathAnno`, the file is searched for in the black box resource
      search path. This is a remnant of the Scala origins of FIRRTL. Copies the
      file to the target directory.
  }];

  let constructor = "circt::firrtl::createBlackBoxReaderPass()";
  let options = [
    Option<"inputPrefix", "input-prefix", "std::string", "",
      "Prefix for input paths in black box annotations. This should be the "
      "directory where the input file was located, to allow for annotations "
      "relative to the input file.">,
    Option<"resourcePrefix", "resource-prefix", "std::string",
      "\"src/main/resources\"",
      "Search path for black box sources specified via the "
      "`BlackBoxResourceAnno` annotation.">
  ];
  let dependentDialects = ["sv::SVDialect", "hw::HWDialect"];
}

def PrefixModules : Pass<"firrtl-prefix-modules", "firrtl::CircuitOp"> {
  let summary = "Prefixes names of modules and mems in a hiearchy";
  let description = [{

    This pass looks for modules annotated with the
    `NestedPrefixModulesAnnotation` and prefixes the names of all modules
    instantiated underneath it.  If `inclusive` is true, it includes the target
    module in the renaming.  If `inclusive` is false, it will only rename
    modules instantiated underneath the target module.  If a module is required
    to have two different prefixes, it will be cloned.

    The supported annotation is:
    ```
      {
        class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
        prefix = "MyPrefix_",
        inclusive = true
      }
    ```

  }];
  let constructor = "circt::firrtl::createPrefixModulesPass()";
}

def PrintInstanceGraph
    : Pass<"firrtl-print-instance-graph", "firrtl::CircuitOp"> {
  let summary = "Print a DOT graph of the module hierarchy.";
  let constructor =  "circt::firrtl::createPrintInstanceGraphPass()";
}

def GrandCentral : Pass<"firrtl-grand-central", "CircuitOp"> {
  let summary = "Remove Grand Central Annotations";
  let description = [{
    Processes annotations associated with SiFive's Grand Central utility.
  }];

  let constructor = "circt::firrtl::createGrandCentralPass()";
  let dependentDialects = ["circt::sv::SVDialect", "circt::hw::HWDialect"];
}

def GrandCentralTaps : Pass<"firrtl-grand-central-taps", "firrtl::CircuitOp"> {
  let summary = "Generate code for grand central data and memory taps";
  let constructor = "circt::firrtl::createGrandCentralTapsPass()";
  let dependentDialects = ["sv::SVDialect", "circt::hw::HWDialect"];
}

def GrandCentralSignalMappings : Pass<"firrtl-grand-central-signal-mappings",
                                      "FModuleOp"> {
  let summary = "Generate signal mappings that force/probe remote signals";
  let constructor = "circt::firrtl::createGrandCentralSignalMappingsPass()";
}

def CheckCombCycles : Pass<"firrtl-check-comb-cycles", "firrtl::CircuitOp"> {
  let summary = "Check combinational cycles and emit errors";
  let description = [{
    This pass checks combinational cycles in the IR and emit errors.
  }];
  let constructor = "circt::firrtl::createCheckCombCyclesPass()";
}

def RemoveResets : Pass<"firrtl-remove-resets", "firrtl::FModuleOp"> {
  let summary = "Remove module-scoped invalidated resets";
  let constructor = "circt::firrtl::createRemoveResetsPass()";
}

def InferReadWrite : Pass<"firrtl-infer-rw", "firrtl::FModuleOp"> {
  let summary = "Infer the read-write memory port";
  let description = [{
    This pass merges the read and write ports of a memory, based a simple
    module-scoped heuristic. The heuristic checks if the read and write enable
    conditions are mutually exclusive.
    The heuristic tries to break up the read enable and write enable logic into an
    `AND` expression tree. It then compares the read and write `AND` terms,
    looking for a situation where the read/write is the complement of the write/read. 
  }];
  let constructor = "circt::firrtl::createInferReadWritePass()";
}

#endif // CIRCT_DIALECT_FIRRTL_PASSES_TD
