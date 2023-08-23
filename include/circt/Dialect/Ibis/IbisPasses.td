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
  }];
  let constructor = "circt::ibis::createTunnelingPass()";
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

#endif // CIRCT_DIALECT_IBIS_PASSES_TD
