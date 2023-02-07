//===- ESIStructure.td - ESI modules ---------------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ESI system and other ESI structural components.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def ESIPureModuleOp : ESI_Op<"pure_module",
      [Symbol, NoTerminator, RegionKindInterface, NoRegionArguments,
       SingleBlock, HasParent<"mlir::ModuleOp">]> {
  let summary = "ESI pure module";
  let description = [{
    A module containing only ESI channels and modules with only ESI ports. All
    non-local connectivity is done through ESI services. If this module is the
    top level in the design, then the design's actual top level ports are
    defined by a BSP.

    Useful on its own for simulation and BSPs which don't define a top-level.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs); 
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{ $sym_name attr-dict-with-keyword $body }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph;}
  }];
}
