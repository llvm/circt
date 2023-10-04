//===-- Passes.td - Ibis pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_PASSES_TD
#define CIRCT_DIALECT_IBIS_PASSES_TD

include "mlir/Pass/PassBase.td"

def IbisCallPrep : Pass<"ibis-call-prep", "ModuleOp"> {
  let summary = "Convert ibis method calls to use `dc.value`";

  let constructor = "circt::ibis::createCallPrepPass()";
  let dependentDialects = [
    "circt::hw::HWDialect", "circt::dc::DCDialect"];
}

def IbisContainerize : Pass<"ibis-containerize", "ModuleOp"> {
  let summary = "Ibis containerization pass";
  let description = [{
    Convert Ibis classes to containers, and outlines containers inside classes.
  }];
  let constructor = "circt::ibis::createContainerizePass()";
}

def IbisTunneling : Pass<"ibis-tunneling", "mlir::ModuleOp"> {
  let summary = "Ibis tunneling pass";
  let description = [{
    Tunnels relative `get_port` ops through the module hierarchy, based on
    `ibis.path` ops. The result of this pass is that various new in- and output
    ports of `!ibis.portref<...>` type are created.
    After this pass, `get_port` ops should only exist at the same scope of
    container instantiations.

    The user may provide options for `readSuffix` and `writeSuffix`, respectively,
    which is to be used to generate the name of the ports that are tunneled
    through the hierarchy, with respect to how the port was requested to be accessed.
    Suffixes must be provided in cases where a port is tunneled for both read and
    write accesses, and the suffixes must be different (in this case, the suffixes
    will be appended to the target port name, and thus de-alias the resulting ports).
  }];
  let constructor = "circt::ibis::createTunnelingPass()";
  let options = [
    Option<"readSuffix", "read-suffix", "std::string", "\".rd\"",
           "Suffix to be used for the port when a port is tunneled for read access">,
    Option<"writeSuffix", "write-suffix", "std::string", "\".wr\"",
            "Suffix to be used for the port when a port is tunneled for write access">
  ];
}

def IbisPortrefLowering : Pass<"ibis-lower-portrefs", "mlir::ModuleOp"> {
  let summary = "Ibis portref lowering pass";
  let description = [{
    Lower `ibis.portref<portref <T>>` to T (i.e. portref resolution).

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
    portref lowering can be done on an `ibis.container` granularity, allowing
    for a bit of parallelism in the flow.
  }];
  let constructor = "circt::ibis::createPortrefLoweringPass()";
}

def IbisCleanSelfdrivers : Pass<"ibis-clean-selfdrivers", "mlir::ModuleOp"> {
  let summary = "Ibis clean selfdrivers pass";
  let description = [{
    - Removes `ibis.port.input`s which are driven by operations within the same
    container.
    - Removes reads of instance ports which are also written to within the same
    container.
  }];

  let constructor = "circt::ibis::createCleanSelfdriversPass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}

def IbisContainersToHW : Pass<"ibis-convert-containers-to-hw", "mlir::ModuleOp"> {
  let summary = "Ibis containers to hw conversion pass";
  let description = [{
    Converts `ibis.container` ops to `hw.module` ops.
  }];
  let constructor = "circt::ibis::createContainersToHWPass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}

def IbisArgifyBlocks : Pass<"ibis-argify-blocks"> {
  let summary = "Add arguments to ibis blocks";
  let description = [{
    Analyses `ibis.sblock` operations and converts any SSA value defined outside
    the `ibis.sblock` to a block argument. As a result, `ibis.sblock.isolated`
    are produced.
  }];
  let constructor = "circt::ibis::createArgifyBlocksPass()";
}

def IbisReblock : Pass<"ibis-reblock", "ibis::MethodOp"> {
  let summary = "Recreates `ibis.sblock` operations from a CFG";
  let description = [{
    Recreates `ibis.sblock` operations from a CFG. Any `ibis.block.attributes`
    operations at the parent operation will be added to the resulting blocks.

    The IR is expected to be in maximal SSA form prior to this pass, given that
    the pass will only inspect the terminator operation of a block for any
    values that are generated within the block. Maximum SSA form thus ensures
    that any value defined within the block is never used outside of the block.

    It is expected that `ibis.call` operations have been isolated into
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
      %v_outer = ibis.sblock(%a0 : i32 = %arg0, %a1 : i32 = %arg1) -> (i32) {
        %res = arith.addi %arg0, %arg1 : i32
        %v = ...
        ibis.sblock.return %v : i32
      }
      cf.br ^bb_bar(%v_outer : i32)
    ```
  }];
  let constructor = "circt::ibis::createReblockPass()";
}

def IbisInlineSBlocks : Pass<"ibis-inline-sblocks", "ibis::MethodOp"> {
  let summary = "Inlines `ibis.sblock` operations as MLIR blocks";
  let description = [{
    Inlines `ibis.sblock` operations, by creating MLIR blocks and `cf`
    operations, while adding attributes to the parent operation about
    `sblock`-specific attributes.

    The parent attributes are located under the `ibis.blockinfo` identifier as
    a dictionary attribute.
    Each entry in the dictionary consists of:
    - Key: an ID (numerical) string identifying the block.
    - Value: a dictionary of attributes. As a minimum this will contain a
      `loc`-keyed attribute specifying the location of the block.
  }];
  let constructor = "circt::ibis::createInlineSBlocksPass()";
  let dependentDialects = ["mlir::cf::ControlFlowDialect"];
}

def IbisConvertCFToHandshake : Pass<"ibis-convert-cf-to-handshake", "ibis::ClassOp"> {
  let summary = "Converts an `ibis.method` to `ibis.method.df`";
  let description = [{
    Converts an `ibis.method` from using `cf` operations and MLIR blocks to
    an `ibis.method.df` operation, using the `handshake` dialect to represent
    control flow through the `handshake` fine grained dataflow operations.
  }];

  let constructor = "circt::ibis::createConvertCFToHandshakePass()";
  let dependentDialects = ["circt::handshake::HandshakeDialect", "mlir::cf::ControlFlowDialect"];
}

def IbisConvertHandshakeToDC : Pass<"ibis-convert-handshake-to-dc", "ibis::ClassOp"> {
  let summary = "Converts an `ibis.method.df` to use DC";
  let description = [{
    Converts an `ibis.method.df` from using `handshake` operations to
    `dc` operations.
  }];
  let constructor = "circt::ibis::createConvertHandshakeToDCPass()";
  let dependentDialects = [
    "circt::dc::DCDialect",
    "circt::handshake::HandshakeDialect",
    "mlir::arith::ArithDialect"
  ];
}

def IbisPrepareScheduling : Pass<"ibis-prepare-scheduling", "ibis::IsolatedStaticBlockOp"> {
  let summary = "Prepare `ibis.sblock` operations for scheduling";
  let description = [{
    Prepares `ibis.sblock` operations for scheduling by:
    - creating an `ibis.pipleine.header` operation
    - moving operations of an `ibis.sblock` into a `pipeline.unscheduled`
      operation, which is connected to the pipeline header.
  }];
  let constructor = "circt::ibis::createPrepareSchedulingPass()";
  let dependentDialects = ["circt::pipeline::PipelineDialect"];
}

def IbisConvertMethodsToContainers : Pass<"ibis-convert-methods-to-containers", "ibis::ClassOp"> {
  let summary = "Converts `ibis.method.df` to `ibis.container`s";
  let constructor = "circt::ibis::createConvertMethodsToContainersPass()";
}

def IbisAddOperatorLibrary : Pass<"ibis-add-operator-library", "mlir::ModuleOp"> {
  let summary = "Injects the Ibis operator library into the IR";
  let description = [{
    Injects the Ibis operator library into the IR, which contains the
    definitions of the Ibis operators.
  }];
  let constructor = "circt::ibis::createAddOperatorLibraryPass()";
  let dependentDialects = ["circt::ssp::SSPDialect"];
}

#endif // CIRCT_DIALECT_IBIS_PASSES_TD
