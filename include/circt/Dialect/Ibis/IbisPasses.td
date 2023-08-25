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

#endif // CIRCT_DIALECT_IBIS_PASSES_TD
