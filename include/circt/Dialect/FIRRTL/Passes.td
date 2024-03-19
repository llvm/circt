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

def ResolvePaths : Pass<"firrtl-resolve-paths", "firrtl::CircuitOp"> {
  let summary = "Lowers UnresolvedPathOps to PathOps";
  let description = [{
    FIRRTL path operations are initially create as unresolved path operations,
    which encode their target as a string. This pass parses and resolves those
    target strings to actual path operations.  Path operations refer to their
    targets using annotations with a unique identifier.
  }];
  let constructor = "circt::firrtl::createResolvePathsPass()";
}

def LowerFIRRTLAnnotations : Pass<"firrtl-lower-annotations", "firrtl::CircuitOp"> {
  let summary = "Lower FIRRTL annotations to usable entities";
  let description = [{
    Lower FIRRTL annotations to usable forms.  FIRRTL annotations are a big bag
    of semi-structured, irregular JSON.  This pass normalizes all supported
    annotations and annotation paths.
  }];
  let constructor = "circt::firrtl::createLowerFIRRTLAnnotationsPass()";
  let options = [
    Option<"ignoreAnnotationClassless", "disable-annotation-classless", "bool", "false",
      "Ignore classless annotations.">,
    Option<"ignoreAnnotationUnknown", "disable-annotation-unknown", "bool", "true",
      "Ignore unknown annotations.">,
    Option<"noRefTypePorts", "no-ref-type-ports", "bool", "false",
      "Create normal ports, not ref type ports.">,
  ];
  let dependentDialects = ["hw::HWDialect"];
  let statistics = [
    Statistic<"numRawAnnotations", "num-raw-annos",
      "Number of raw annotations on circuit">,
    Statistic<"numAddedAnnos", "num-added-annos",
       "Number of additional annotations">,
    Statistic<"numAnnos", "num-annos",
       "Total number of annotations processed">,
    Statistic<"numUnhandled", "num-unhandled-annos",
      "Number of unhandled annotations">,
    Statistic<"numReusedHierPathOps", "num-reused-hierpath",
      "Number of reused HierPathOp's">,
  ];
}

def LowerFIRRTLTypes : Pass<"firrtl-lower-types", "firrtl::CircuitOp"> {
  let summary = "Lower FIRRTL types to ground types";
  let description = [{
    Lower aggregate FIRRTL types to ground types.  Memories, ports, wires, etc
    are split apart by elements of aggregate types.  The only aggregate types
    which exist after this pass are memory ports, though memory data types are
    split.

    Connect and expansion and canonicalization happen in this pass.
  }];
  let constructor = "circt::firrtl::createLowerFIRRTLTypesPass()";
  let options = [
    Option<"flattenAggregateMemData", "flatten-mem", "bool", "false",
           "Concat all elements of the aggregate data into a single element.">,
    Option<"preserveAggregate", "preserve-aggregate", "PreserveAggregate::PreserveMode",
          "PreserveAggregate::None",
          "Specify aggregate preservation mode",
          [{::llvm::cl::values(
            clEnumValN(PreserveAggregate::None, "none", "Preserve no aggregate"),
            clEnumValN(PreserveAggregate::OneDimVec, "1d-vec", "Preserve 1d vectors"),
            clEnumValN(PreserveAggregate::Vec, "vec", "Preserve vectors"),
            clEnumValN(PreserveAggregate::All, "all", "Preserve vectors and bundles")
          )}]>,
    Option<"preserveMemories", "preserve-memories", "PreserveAggregate::PreserveMode",
          "PreserveAggregate::None",
          "Specify memory preservation mode",
          [{::llvm::cl::values(
            clEnumValN(PreserveAggregate::None, "none", "Preserve no aggregate"),
            clEnumValN(PreserveAggregate::OneDimVec, "1d-vec", "Preserve 1d vectors"),
            clEnumValN(PreserveAggregate::Vec, "vec", "Preserve vectors"),
            clEnumValN(PreserveAggregate::All, "all", "Preserve vectors and bundles")
          )}]>
  ];
  let dependentDialects = ["hw::HWDialect"];
}

def LowerSignatures : Pass<"firrtl-lower-signatures", "firrtl::CircuitOp"> {
  let summary = "Lower FIRRTL module signatures";
  let description = [{
    Lower aggregate FIRRTL types in Modules as indicated by the calling
    convention.
  }];
  let constructor = "circt::firrtl::createLowerSignaturesPass()";
}

def PassiveWires : Pass<"firrtl-passive-wires", "firrtl::FModuleOp"> {
  let summary = "Make FIRRTL wires have passive type";
  let description = [{
    Eliminate flips from aggregate types on wires.
  }];
  let constructor = "circt::firrtl::createPassiveWiresPass()";
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

def RegisterOptimizer : Pass<"firrtl-register-optimizer", "firrtl::FModuleOp"> {
  let summary = "Optimizer Registers";
  let description = [{
    This pass applies classic FIRRTL register optimizations.  These
    optimizations are isolated to this pass as they can change the visible
    behavior of the register, especially before reset.
  }];
  let constructor = "circt::firrtl::createRegisterOptimizerPass()";
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

def IMDeadCodeElim : Pass<"firrtl-imdeadcodeelim", "mlir::ModuleOp"> {
  let summary = "Intermodule dead code elimination";
  let description = [{
    This pass performs inter-module liveness analysis and deletes dead code
    aggressively. A value is considered as alive if it is connected to a port
    of public modules or a value with a symbol. We first populate alive values
    into a set, and then propagate the liveness by looking at their dataflow.
  }];
  let constructor = "circt::firrtl::createIMDeadCodeElimPass()";
  let statistics = [
    Statistic<"numErasedOps", "num-erased-ops", "Number of operations erased">,
    Statistic<"numErasedModules", "num-erased-modules", "Number of modules erased">,
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
  let dependentDialects = ["debug::DebugDialect"];
}

def AddSeqMemPorts : Pass<"firrtl-add-seqmem-ports", "firrtl::CircuitOp"> {
  let summary = "Add extra ports to memory modules";
  let description = [{
    This pass looks for `AddSeqMemPortAnnotation` annotations and adds extra
    ports to memories.  It will emit metadata based if the
    `AddSeqMemPortsFileAnnotation` annotation is specified.

    This pass requires that FIRRTL MemOps have been lowered to modules to add
    the extra ports.
  }];
  let constructor = "circt::firrtl::createAddSeqMemPortsPass()";
  let dependentDialects = [
    "emit::EmitDialect",
    "sv::SVDialect",
    "hw::HWDialect"
  ];
  let statistics = [
    Statistic<"numAddedPorts", "num-added-ports", "Number of extra ports added">,
  ];
}

def CreateSiFiveMetadata : Pass<"firrtl-emit-metadata", "mlir::ModuleOp"> {
  let summary = "Emit metadata of the FIRRTL modules";
  let description = [{
    This pass handles the emission of several different kinds of metadata.
  }];
  let constructor = "circt::firrtl::createCreateSiFiveMetadataPass()";
  let options = [Option<"replSeqMem", "repl-seq-mem", "bool", "false",
      "Lower the seq mem for macro replacement and emit relevant metadata">,
    Option<"replSeqMemFile", "repl-seq-mem-file", "std::string", "",
      "File to which emit seq meme metadata">
  ];
  let dependentDialects = [
      "emit::EmitDialect",
      "hw::HWDialect",
      "om::OMDialect",
      "sv::SVDialect",
  ];
}

def FlattenMemory : Pass<"firrtl-flatten-memory", "firrtl::FModuleOp"> {
  let summary = "Flatten aggregate memory data to a UInt";
  let description = [{
    This pass flattens the aggregate data of memory into a UInt, and inserts
    appropriate bitcasts to access the data.
  }];
  let constructor = "circt::firrtl::createFlattenMemoryPass()";
  let statistics = [
    Statistic<"numFlattenedMems", "num-flatten-mems",
      "Number of memories flattened">,
  ];
}

def LowerMemory : Pass<"firrtl-lower-memory", "firrtl::CircuitOp"> {
  let summary = "Lower memories to generated modules";
  let description = [{
    This pass lowers FIRRTL memory operations to generated modules.
  }];
  let constructor = "circt::firrtl::createLowerMemoryPass()";
  let statistics = [
    Statistic<"numCreatedMemModules", "num-created-mem-modules",
      "Number of modules created">,
    Statistic<"numLoweredMems", "num-lowered-mems",
      "Number of memories lowered">,
  ];
}

def Dedup : Pass<"firrtl-dedup", "firrtl::CircuitOp"> {
  let summary = "Deduplicate modules which are structurally equivalent";
  let description = [{
    This pass detects modules which are structurally equivalent and removes the
    duplicate module by replacing all instances of one with the other.
    Structural equivalence ignores the naming of operations and fields in
    bundles, and any annotations. Deduplicating a module may cause the result
    type of instances to change if the field names of a bundle type change.  To
    handle this, the pass will update any bulk-connections so that the correct
    fields are legally connected. Deduplicated modules will have their
    annotations merged, which tends to create many non-local annotations.
  }];
  let statistics = [
    Statistic<"erasedModules", "num-erased-modules",
      "Number of modules which were erased by deduplication">
  ];
  let constructor = "circt::firrtl::createDedupPass()";
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
  let dependentDialects = [
      "emit::EmitDialect",
      "sv::SVDialect",
      "hw::HWDialect"
  ];
}

def LowerMatches : Pass<"firrtl-lower-matches", "firrtl::FModuleOp"> {
  let summary = "Remove all matchs conditional blocks";
  let description = [{
    Lowers FIRRTL match statements in to when statements, which can later be
    lowered with ExpandWhens.
  }];
  let constructor = "circt::firrtl::createLowerMatchesPass()";
}

def ExpandWhens : InterfacePass<"firrtl-expand-whens", "firrtl::FModuleLike"> {
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
  let statistics = [
    Statistic<"numCreatedMems", "num-created-mems",
      "Number of memories created">,
    Statistic<"numLoweredMems", "num-lowered-mems",
      "Number of memories lowered">,
    Statistic<"numPortlessMems", "num-portless-mems",
      "Number of memories dropped as having no valid ports">,
  ];
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
    This pass reads the Verilog source files for black boxes and adds them as
    `sv.verbatim.file` operations into the IR. Later passes can then write
    these files back to disk to ensure that they can be accessed by other tools
    down the line in a well-known location. Supports inline and path
    annotations for black box source files.

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
  }];

  let constructor = "circt::firrtl::createBlackBoxReaderPass()";
  let options = [
    Option<"inputPrefix", "input-prefix", "std::string", "",
      "Prefix for input paths in black box annotations. This should be the "
      "directory where the input file was located, to allow for annotations "
      "relative to the input file.">
  ];
  let dependentDialects = [
      "emit::EmitDialect",
      "hw::HWDialect",
      "sv::SVDialect"
  ];
}

def PrefixModules : Pass<"firrtl-prefix-modules", "firrtl::CircuitOp"> {
  let summary = "Prefixes names of modules and mems in a hierarchy";
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

def PrintFIRRTLFieldSourcePass
    : Pass<"firrtl-print-field-source", "firrtl::FModuleOp"> {
  let summary = "Print field source information.";
  let constructor =  "circt::firrtl::createFIRRTLFieldSourcePass()";
}

def PrintInstanceGraph
    : Pass<"firrtl-print-instance-graph", "firrtl::CircuitOp"> {
  let summary = "Print a DOT graph of the module hierarchy.";
  let constructor =  "circt::firrtl::createPrintInstanceGraphPass()";
}

def PrintNLATable
    : Pass<"firrtl-print-nla-table", "firrtl::CircuitOp"> {
  let summary = "Print the NLA Table.";
  let constructor =  "circt::firrtl::createPrintNLATablePass()";
}

def GrandCentral : Pass<"firrtl-grand-central", "CircuitOp"> {
  let summary = "Remove Grand Central Annotations";
  let description = [{
    Processes annotations associated with SiFive's Grand Central utility.
  }];

  let constructor = "circt::firrtl::createGrandCentralPass()";
  let dependentDialects = ["circt::sv::SVDialect", "circt::hw::HWDialect"];
  let options = [
    Option<"companionMode", "companion-mode", "CompanionMode",
           "CompanionMode::Bind",
           "specify the handling of companion modules", [{
            ::llvm::cl::values(
              clEnumValN(CompanionMode::Bind, "bind",
                "Lower companion instances to SystemVerilog binds"),
              clEnumValN(CompanionMode::Instantiate, "instantiate",
                "Instantiate companions in the design"),
              clEnumValN(CompanionMode::Drop, "drop",
                "Remove companions from the design"))
           }]>
  ];
  let statistics = [
    Statistic<"numViews", "num-views-created",
      "Number of top-level SystemVerilog interfaces that were created">,
    Statistic<"numInterfaces", "num-interfaces-created",
      "Number of SystemVerilog interfaces that were created">,
    Statistic<"numXMRs", "num-xmrs-created",
      "Number of SystemVerilog XMRs added">,
    Statistic<"numAnnosRemoved", "num-annotations-removed",
      "Number of annotations removed">
  ];
}

def CheckCombLoops : Pass<"firrtl-check-comb-loops", "firrtl::CircuitOp"> {
  let summary = "Check combinational cycles and emit errors";
  let description = [{
    This pass checks combinational cycles in the IR and emit errors.
  }];
  let constructor = "circt::firrtl::createCheckCombLoopsPass()";
}

def SFCCompat : Pass<"firrtl-sfc-compat", "firrtl::FModuleOp"> {
  let summary = "Perform SFC Compatibility fixes";
  let constructor = "circt::firrtl::createSFCCompatPass()";
}

def MergeConnections : Pass<"merge-connections", "firrtl::FModuleOp"> {
  let summary = "Merge field-level connections into full bundle connections";
  let constructor = "circt::firrtl::createMergeConnectionsPass()";
  let options = [
    Option<"enableAggressiveMerging", "aggressive-merging", "bool", "false",
      "Merge connections even when source values won't be simplified.">
  ];
}

def Vectorization : Pass<"vectorization", "firrtl::FModuleOp"> {
  let summary = "Transform firrtl primitive operations into vector operations";
  let constructor = "circt::firrtl::createVectorizationPass()";
}

def InferReadWrite : Pass<"firrtl-infer-rw", "firrtl::FModuleOp"> {
  let summary = "Infer the read-write memory port";
  let description = [{
    This pass merges the read and write ports of a memory, using a simple
    module-scoped heuristic. The heuristic checks if the read and write enable
    conditions are mutually exclusive.
    The heuristic tries to break up the read enable and write enable logic into an
    `AND` expression tree. It then compares the read and write `AND` terms,
    looking for a situation where the read/write is the complement of the write/read.
  }];
  let constructor = "circt::firrtl::createInferReadWritePass()";
  let statistics = [
    Statistic<"numRWPortMemoriesInferred", "num-rw-port-mems-inferred",
      "Number of memories inferred to use RW port">,
  ];
}

def InjectDUTHierarchy : Pass<"firrtl-inject-dut-hier", "firrtl::CircuitOp"> {
  let summary = "Add a level of hierarchy outside the DUT";
  let description = [{
    This pass takes the DUT (as indicated by the presence of a
    MarkDUTAnnotation) and moves all the contents of it into a new module
    insided the DUT named by an InjectDUTHierarchyAnnotation.  This pass is
    intended to be used in conjunction with passes that pull things out of the
    DUT, e.g., SRAM extraction, to give the extracted modules a new home that is
    still inside the original DUT.
  }];
  let constructor = "circt::firrtl::createInjectDUTHierarchyPass()";
}

def ExtractInstances : Pass<"firrtl-extract-instances", "firrtl::CircuitOp"> {
  let summary = "Move annotated instances upwards in the module hierarchy";
  let description = [{
    This pass takes instances in the design annotated with one out of a
    particular set of annotations and pulls them upwards to a location further
    up in the module hierarchy.

    The annotations that control the behaviour of this pass are:
    - `MarkDUTAnnotation`
    - `ExtractBlackBoxAnnotation`
    - `ExtractClockGatesFileAnnotation`
  }];
  let constructor = "circt::firrtl::createExtractInstancesPass()";
  let dependentDialects = [
    "emit::EmitDialect",
    "circt::hw::HWDialect",
    "sv::SVDialect",
  ];
}

def MemToRegOfVec : Pass<"firrtl-mem-to-reg-of-vec", "firrtl::CircuitOp"> {
  let summary = "Convert combinational memories to a vector of registers";
  let description = [{
    This pass generates the logic to implement a memory using Registers.
  }];
  let options = [
    Option<"replSeqMem", "repl-seq-mem", "bool",
                "false", "Prepare seq mems for macro replacement">,
    Option<"ignoreReadEnable", "ignore-read-enable-mem", "bool",
                "false",
    "ignore the read enable signal, instead of assigning X on read disable">
   ];
  let constructor = "circt::firrtl::createMemToRegOfVecPass()";
  let statistics = [
    Statistic<"numConvertedMems", "num-converted-mems",
      "Number of memories converted to registers">,
  ];
}

def DropName : Pass<"firrtl-drop-names", "firrtl::FModuleOp"> {
  let summary = "Drop interesting names";
  let description = [{
    This pass changes names of namable ops to droppable so that we can disable
    full name preservation. For example,
    before:
    ```mlir
    %a = firrtl.node interesting_name %input
    ```
    after:
    ```mlir
    %a = firrtl.node droppable_name %input
    ```
  }];
  let constructor = "circt::firrtl::createDropNamesPass()";
  let options = [
    Option<"preserveMode", "preserve-values", "PreserveValues::PreserveMode",
           "PreserveValues::None",
           "specify the values which can be optimized away", [{
            ::llvm::cl::values(
              clEnumValN(PreserveValues::Strip, "strip",
                "Strip all names. No name is preserved"),
              clEnumValN(PreserveValues::None, "none",
                "Preserve no named values. Names could be preserved by best-effort unlike `strip`"),
              clEnumValN(PreserveValues::Named, "named",
                "Preserve values with meaningful names"),
              clEnumValN(PreserveValues::All, "all",
                "Preserve all values"))
           }]>
  ];
  let statistics = [
    Statistic<"numNamesDropped", "num-names-dropped",
      "Number of names dropped">,
    Statistic<"numNamesConverted", "num-names-converted",
      "Number of interesting names made droppable">,
  ];
}

def RandomizeRegisterInit :
    Pass<"firrtl-randomize-register-init", "firrtl::FModuleOp"> {
  let summary = "Randomize register initialization.";
  let description = [{
    This pass eagerly creates large vectors of randomized bits for initializing
    registers, and marks each register with attributes indicating which bits to
    read. If the registers survive until LowerToHW, their initialization logic
    will pick up the correct bits.

    This ensures a stable initialization, so registers should always see the
    same initial value for the same seed, regardless of optimization levels.
  }];
  let constructor = "circt::firrtl::createRandomizeRegisterInitPass()";
}

def LowerXMR : Pass<"firrtl-lower-xmr", "firrtl::CircuitOp"> {
  let summary = "Lower ref ports to XMR";
  let description = [{
    This pass lowers RefType ops and ports to verbatim encoded XMRs.
  }];
  let constructor = "circt::firrtl::createLowerXMRPass()";
  let dependentDialects = [
    "emit::EmitDialect",
    "sv::SVDialect"
  ];
}

def IntrinsicInstancesToOps : Pass<"firrtl-intrinsic-instances-to-ops", "firrtl::CircuitOp"> {
  let summary = "Lower instances instances of intrinsic modules to ops.";
  let description = [{
    This pass replaces instances of intrinsic modules (intmodule) with 
    `firrtl.int.generic` operations, and removes the intmodule's from
    the circuit.
 
    Use this before running LowerIntrinsics.
  }];
  let constructor = "circt::firrtl::createIntrinsicInstancesToOpsPass()";
  let options = [
    Option<"fixupEICGWrapper", "fixup-eicg-wrapper", "bool", "false",
      "Lower `EICG_wrapper` modules into clock gate intrinsics">,
  ];
}

def LowerIntrinsics : Pass<"firrtl-lower-intrinsics", "firrtl::FModuleOp"> {
  let summary = "Lower intrinsics";
  let description = [{
    This pass lowers generic intrinsic ops to their implementation or op.
  }];
  let constructor = "circt::firrtl::createLowerIntrinsicsPass()";
}

def LowerOpenAggs : Pass<"firrtl-lower-open-aggs", "firrtl::CircuitOp"> {
  let summary = "Lower 'Open' aggregates by splitting out non-hardware elements";
  let description = [{
     This pass lowers aggregates of the more open varieties into their equivalents
     using only hardware types, by pulling out non-hardware to other locations.
  }];
  let constructor = "circt::firrtl::createLowerOpenAggsPass()";
}

def ResolveTraces : Pass<"firrtl-resolve-traces", "firrtl::CircuitOp"> {
  let summary = "Write out TraceAnnotations to an output annotation file";
  let description = [{
    This pass implements Chisel's Trace API.  It collects all TraceAnnotations
    that exist in the circuit, updates them with information about the final
    target in a design, and writes these to an output annotation file.  This
    exists for Chisel users to build tooling around them that needs to query the
    final output name/path of some component in a Chisel circuit.

    Note: this pass and API are expected to be eventually replaced via APIs and
    language bindings that enable users to directly query the MLIR.
  }];
  let constructor = "circt::firrtl::createResolveTracesPass()";
  let options = [Option<"outputAnnotationFilename", "file", "std::string", "",
      "Output file for the JSON-serialized Trace Annotations">];
  let dependentDialects = ["emit::EmitDialect", "sv::SVDialect"];
}

def InnerSymbolDCE : Pass<"firrtl-inner-symbol-dce", "mlir::ModuleOp"> {
  let summary = "Eliminate dead inner symbols";
  let description = [{
    This pass deletes all inner symbols which have no uses. This is necessary to
    unblock optimizations and removal of the operations which have these unused
    inner symbols.
  }];
  let constructor = "circt::firrtl::createInnerSymbolDCEPass()";
  let statistics = [
    Statistic<"numInnerRefsFound", "num-inner-refs-found", "Number of inner-refs found">,
    Statistic<"numInnerSymbolsFound", "num-inner-sym-found", "Number of inner symbols found">,
    Statistic<"numInnerSymbolsRemoved", "num-inner-sym-removed", "Number of inner symbols removed">
  ];
}

def VBToBV : Pass<"firrtl-vb-to-bv", "firrtl::CircuitOp"> {
  let summary = "Transform vector-of-bundles to bundle-of-vectors";
  let description = [{
    This pass converts vectors containing bundles, into bundles containing
    vectors.
  }];
  let constructor = "circt::firrtl::createVBToBVPass()";
}

def DropConst : Pass<"firrtl-drop-const", "firrtl::CircuitOp"> {
  let summary = "Drop 'const' modifier from types";
  let description = [{
    This pass drops the 'const' modifier from all types and removes all
    const-cast ops.

    This simplifies downstream passes and folds so that they do not need to
    take 'const' into account.
  }];
  let constructor = "circt::firrtl::createDropConstPass()";
}

def FinalizeIR : Pass<"firrtl-finalize-ir", "mlir::ModuleOp"> {
  let summary = "Perform final IR mutations after ExportVerilog";
  let description = [{
    This pass finalizes the IR after it has been exported with ExportVerilog,
    and before firtool emits the final IR. This includes mutations like dropping
    verbatim ops that represent sideband files and are not required in the IR.
  }];
  let constructor = "circt::firrtl::createFinalizeIRPass()";
  let dependentDialects = ["hw::HWDialect", "sv::SVDialect"];
}

def LowerClasses : Pass<"firrtl-lower-classes", "firrtl::CircuitOp"> {
  let summary = "Lower FIRRTL classes and objects to OM classes and objects";
  let description = [{
    This pass walks all FIRRTL classes and creates OM classes. It also lowers
    FIRRTL objects to OM objects. OM classes are created with parameters
    corresponding to FIRRTL input properties, and OM class fields corresponding
    to FIRRTL output properties. FIRRTL operations are converted to OM
    operations.
  }];
  let constructor = "circt::firrtl::createLowerClassesPass()";
  let dependentDialects = ["om::OMDialect"];
}

def LowerLayers : Pass<"firrtl-lower-layers", "firrtl::CircuitOp"> {
  let summary = "Lower layers and layer blocks to instances";
  let description = [{
    This pass lowers FIRRTL layers as defined by their convention.  After this
    pass, all layer blocks and layers are removed.
  }];
  let constructor = "circt::firrtl::createLowerLayersPass()";
  let dependentDialects = ["sv::SVDialect"];
}

def LayerMerge : Pass<"firrtl-layer-merge", "firrtl::FModuleOp"> {
  let summary = "Merge layer blocks";
  let description = [{
    Combine all layer blocks in a module which reference the same layer
    definition.
  }];
  let constructor = "::circt::firrtl::createLayerMergePass()";
  let statistics = [
    Statistic<"numMerged", "num-merged", "Number of layers merged">,
  ];
}

def LayerSink : Pass<"firrtl-layer-sink", "firrtl::FModuleOp"> {
  let summary = "Sink operations into layer blocks";
  let description = [{

  }];
  let constructor = "::circt::firrtl::createLayerSinkPass()";
  let statistics = [
    Statistic<"numSunk", "num-sunk", "Number of operations sunk">,
  ];
}

def HoistPassthrough : Pass<"firrtl-hoist-passthrough", "firrtl::CircuitOp"> {
  let summary = "Hoist passthrough logic";
  let description = [{
    This pass hoists simple passthrough connections up.
  }];
  let constructor = "circt::firrtl::createHoistPassthroughPass()";
  let options = [
    Option<"hoistHWDrivers", "hoist-hw", "bool", "true",
      "Hoist HW drivers.">
  ];
  let statistics = [
    Statistic<"numUTurnsHoisted", "num-u-turns",
      "Number of u-turns hoisted">,
    Statistic<"numRefDrivers", "num-ref-drivers",
      "Number of ref-type drivers processed">,
    Statistic<"numHWDrivers", "num-hw-drivers",
      "Number of hw-type drivers processed">
  ];
}

def ProbeDCE : Pass<"firrtl-probe-dce", "firrtl::CircuitOp"> {
  let summary = "Delete dead probe ports and operations.";
  let description = [{
    Simple pass to delete dead probe ports and operations.
    Diagnoses illegal or unsupported input probe uses.
    Post-condition: no input probe ports.
  }];
  let constructor = "circt::firrtl::createProbeDCEPass()";
  let statistics = [
    Statistic<"numErasedPorts", "num-erased-ports", "Number of ports erased">,
    Statistic<"numErasedOps", "num-erased-ops", "Number of operations erased">
  ];
}

def MaterializeDebugInfo :
    Pass<"firrtl-materialize-debug-info", "firrtl::FModuleOp"> {
  let summary = "Generate debug ops to track FIRRTL values";
  let description = [{
    This pass creates debug ops to track FIRRTL-level ports, nodes, wires,
    registers, and instances throughout the pipeline. The `DebugInfo` analysis
    can then be used at a later point in the pipeline to obtain a source
    language view into the lowered IR.
  }];
  let constructor = "circt::firrtl::createMaterializeDebugInfoPass()";
  let dependentDialects = ["debug::DebugDialect"];
}

def Lint :
    Pass<"firrtl-lint", "firrtl::FModuleOp"> {
  let summary = "An analysis pass to detect static simulation failures.";
  let description = [{
    This pass detects operations that will trivially fail any simulation.
    Currently it detects assertions whose predicate condition can be statically
    inferred to be false. The pass emits error on such failing ops.
  }];
  let constructor = "circt::firrtl::createLintingPass()";
}

def SpecializeOption :
    Pass<"firrtl-specialize-option", "firrtl::CircuitOp"> {
  let summary = "Specialize a configurable design for a given option.";
  let description = [{
    If the design has option groups, instead of preserving the information
    up to the emitted SystemVerilog, this pass specializes it early on for a
    given choice.
  }];
  let constructor = "circt::firrtl::createSpecializeOptionPass()";

  let options = [
    ListOption<"select", "select", "std::string",
               "Options to specialize, in option=case format">,
  ];
  let statistics = [
    Statistic<"numInstances", "num-instances",
      "Number of instances specialized">
  ];
}

#endif // CIRCT_DIALECT_FIRRTL_PASSES_TD
