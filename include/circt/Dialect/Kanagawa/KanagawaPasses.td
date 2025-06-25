//===-- Passes.td - Kanagawa pass definition file ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_KANAGAWA_PASSES_TD
#define CIRCT_DIALECT_KANAGAWA_PASSES_TD

include "mlir/Pass/PassBase.td"

def KanagawaCallPrep : Pass<"kanagawa-call-prep", "DesignOp"> {
  let summary = "Convert kanagawa method calls to use `dc.value`";

  let constructor = "::circt::kanagawa::createCallPrepPass()";
  let dependentDialects = [
    "::circt::hw::HWDialect"];
}


def KanagawaContainerize : Pass<"kanagawa-containerize", "DesignOp"> {
  let summary = "Kanagawa containerization pass";
  let description = [{
    Convert Kanagawa classes to containers, and outlines containers inside classes.
  }];
  let constructor = "::circt::kanagawa::createContainerizePass()";
}

def KanagawaTunneling : Pass<"kanagawa-tunneling", "DesignOp"> {
  let summary = "Kanagawa tunneling pass";
  let description = [{
    Tunnels relative `get_port` ops through the module hierarchy, based on
    `kanagawa.path` ops. The result of this pass is that various new in- and output
    ports of `!kanagawa.portref<...>` type are created.
    After this pass, `get_port` ops should only exist at the same scope of
    container instantiations.

    The user may provide options for `readSuffix` and `writeSuffix`, respectively,
    which is to be used to generate the name of the ports that are tunneled
    through the hierarchy, with respect to how the port was requested to be accessed.
    Suffixes must be provided in cases where a port is tunneled for both read and
    write accesses, and the suffixes must be different (in this case, the suffixes
    will be appended to the target port name, and thus de-alias the resulting ports).
  }];
  let constructor = "::circt::kanagawa::createTunnelingPass()";
  let options = [
    Option<"readSuffix", "read-suffix", "std::string", "\".rd\"",
           "Suffix to be used for the port when a port is tunneled for read access">,
    Option<"writeSuffix", "write-suffix", "std::string", "\".wr\"",
            "Suffix to be used for the port when a port is tunneled for write access">
  ];
}

def KanagawaPortrefLowering : Pass<"kanagawa-lower-portrefs", "::mlir::ModuleOp"> {
  let summary = "Kanagawa portref lowering pass";
  let description = [{
    Lower `kanagawa.portref<portref <T>>` to T (i.e. portref resolution).

    We do this by analyzing how a portref is used
    inside a container, and then creating an in- or output port based on that.
    That is:
    - write to `portref<in portref<in, T>>` becomes `out T`
        i.e this container wants to write to an input of another container, hence
        it will produce an output value that will drive that input port.
    - read from `portref<in portref<out, T>>` becomes `in T`
        i.e. this container wants to read from an output of another container,
        hence it will have an input port that will be driven by that output port.
    - write to `portref<out portref<out, T>>` becomes `out T`
        i.e. a port reference inside the module will be driven by a value from
        the outside.
    - read from `portref<out portref<in, T>>` becomes `in T`
        i.e. a port reference inside the module will be driven by a value from
        the outside.

    A benefit of having portref lowering separate from portref tunneling is that
    portref lowering can be done on an `kanagawa.container` granularity, allowing
    for a bit of parallelism in the flow.
  }];
  let constructor = "::circt::kanagawa::createPortrefLoweringPass()";
}

def KanagawaCleanSelfdrivers : Pass<"kanagawa-clean-selfdrivers", "DesignOp"> {
  let summary = "Kanagawa clean selfdrivers pass";
  let description = [{
    - Removes `kanagawa.port.input`s which are driven by operations within the same
    container.
    - Removes reads of instance ports which are also written to within the same
    container.
  }];

  let constructor = "::circt::kanagawa::createCleanSelfdriversPass()";
  let dependentDialects = ["::circt::hw::HWDialect"];
}

def KanagawaContainersToHW : Pass<"kanagawa-convert-containers-to-hw", "::mlir::ModuleOp"> {
  let summary = "Kanagawa containers to hw conversion pass";
  let description = [{
    Converts `kanagawa.container` ops to `hw.module` ops.
  }];
  let constructor = "::circt::kanagawa::createContainersToHWPass()";
  let dependentDialects = ["::circt::hw::HWDialect"];
}

def KanagawaEliminateRedundantOps : Pass<"kanagawa-eliminate-redundant-ops", "kanagawa::ContainerOp"> {
  let summary = "Kanagawa eliminate redundant operations pass";
  let description = [{
    Eliminates redundant operations within Kanagawa containers to optimize the IR.
    This pass analyzes operations within containers and removes unnecessary or
    duplicate operations that do not affect the semantic behavior.

    Redundant operations can (read: will) cause issues in other passes. So this
    pass needs to be run after any pass which can introduce redundant
    operations.
  }];
  let constructor = "::circt::kanagawa::createEliminateRedundantOpsPass()";
}

def KanagawaArgifyBlocks : Pass<"kanagawa-argify-blocks"> {
  let summary = "Add arguments to kanagawa blocks";
  let description = [{
    Analyses `kanagawa.sblock` operations and converts any SSA value defined outside
    the `kanagawa.sblock` to a block argument. As a result, `kanagawa.sblock.isolated`
    are produced.
  }];
  let constructor = "::circt::kanagawa::createArgifyBlocksPass()";
}

def KanagawaReblock : Pass<"kanagawa-reblock", "kanagawa::MethodOp"> {
  let summary = "Recreates `kanagawa.sblock` operations from a CFG";
  let description = [{
    Recreates `kanagawa.sblock` operations from a CFG. Any `kanagawa.block.attributes`
    operations at the parent operation will be added to the resulting blocks.

    The IR is expected to be in maximal SSA form prior to this pass, given that
    the pass will only inspect the terminator operation of a block for any
    values that are generated within the block. Maximum SSA form thus ensures
    that any value defined within the block is never used outside of the block.

    It is expected that `kanagawa.call` operations have been isolated into
    their own basic blocks before this pass is run. This implies that all
    operations within a block (except for the terminator operation) can be
    statically scheduled with each other.

    e.g.
    ```mlir
    ^bb_foo(%arg0 : i32, %arg1 : i32):
      %res = arith.addi %arg0, %arg1 : i32
      %v = ...
      cf.br ^bb_bar(%v : i32)
    ```

    becomes
    ```mlir
    ^bb_foo(%arg0 : i32, %arg1 : i32):
      %v_outer = kanagawa.sblock(%a0 : i32 = %arg0, %a1 : i32 = %arg1) -> (i32) {
        %res = arith.addi %arg0, %arg1 : i32
        %v = ...
        kanagawa.sblock.return %v : i32
      }
      cf.br ^bb_bar(%v_outer : i32)
    ```
  }];
  let constructor = "::circt::kanagawa::createReblockPass()";
}

def KanagawaInlineSBlocks : Pass<"kanagawa-inline-sblocks", "kanagawa::MethodOp"> {
  let summary = "Inlines `kanagawa.sblock` operations as MLIR blocks";
  let description = [{
    Inlines `kanagawa.sblock` operations, by creating MLIR blocks and `cf`
    operations, while adding attributes to the parent operation about
    `sblock`-specific attributes.

    The parent attributes are located under the `kanagawa.blockinfo` identifier as
    a dictionary attribute.
    Each entry in the dictionary consists of:
    - Key: an ID (numerical) string identifying the block.
    - Value: a dictionary of attributes. As a minimum this will contain a
      `loc`-keyed attribute specifying the location of the block.
  }];
  let constructor = "::circt::kanagawa::createInlineSBlocksPass()";
  let dependentDialects = ["::mlir::cf::ControlFlowDialect"];
}

def KanagawaConvertCFToHandshake : Pass<"kanagawa-convert-cf-to-handshake", "kanagawa::ClassOp"> {
  let summary = "Converts an `kanagawa.method` to `kanagawa.method.df`";
  let description = [{
    Converts an `kanagawa.method` from using `cf` operations and MLIR blocks to
    an `kanagawa.method.df` operation, using the `handshake` dialect to represent
    control flow through the `handshake` fine grained dataflow operations.
  }];

  let constructor = "::circt::kanagawa::createConvertCFToHandshakePass()";
  let dependentDialects = ["::circt::handshake::HandshakeDialect", "::mlir::cf::ControlFlowDialect"];
}

def KanagawaConvertHandshakeToDC : Pass<"kanagawa-convert-handshake-to-dc", "kanagawa::ClassOp"> {
  let summary = "Converts an `kanagawa.method.df` to use DC";
  let description = [{
    Converts an `kanagawa.method.df` from using `handshake` operations to
    `dc` operations.
  }];
  let constructor = "::circt::kanagawa::createConvertHandshakeToDCPass()";
  let dependentDialects = [
    "::circt::dc::DCDialect",
    "::circt::handshake::HandshakeDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def KanagawaPrepareScheduling : Pass<"kanagawa-prepare-scheduling", "kanagawa::IsolatedStaticBlockOp"> {
  let summary = "Prepare `kanagawa.sblock` operations for scheduling";
  let description = [{
    Prepares `kanagawa.sblock` operations for scheduling by:
    - creating an `kanagawa.pipleine.header` operation
    - moving operations of an `kanagawa.sblock` into a `pipeline.unscheduled`
      operation, which is connected to the pipeline header.
  }];
  let constructor = "::circt::kanagawa::createPrepareSchedulingPass()";
  let dependentDialects = ["::circt::pipeline::PipelineDialect"];
}

def KanagawaConvertMethodsToContainers : Pass<"kanagawa-convert-methods-to-containers", "kanagawa::ClassOp"> {
  let summary = "Converts `kanagawa.method.df` to `kanagawa.container`s";
  let constructor = "::circt::kanagawa::createConvertMethodsToContainersPass()";
}

def KanagawaAddOperatorLibrary : Pass<"kanagawa-add-operator-library", "::mlir::ModuleOp"> {
  let summary = "Injects the Kanagawa operator library into the IR";
  let description = [{
    Injects the Kanagawa operator library into the IR, which contains the
    definitions of the Kanagawa operators.
  }];
  let constructor = "::circt::kanagawa::createAddOperatorLibraryPass()";
  let dependentDialects = ["::circt::ssp::SSPDialect"];
}

#endif // CIRCT_DIALECT_KANAGAWA_PASSES_TD
