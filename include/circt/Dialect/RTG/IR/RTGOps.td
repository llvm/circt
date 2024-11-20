//===- RTGOps.td - RTG operations --------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the RTG MLIR operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/Properties.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operation in this dialect.
class RTGOp<string mnemonic, list<Trait> traits = []> :
    Op<RTGDialect, mnemonic, traits>;

//===- Sequence Handling Operations ---------------------------------------===//

def SequenceOp : RTGOp<"sequence", [
  IsolatedFromAbove,
  Symbol,
  SingleBlock,
  NoTerminator,
  HasParent<"mlir::ModuleOp">,
]> {
  let summary = "a sequence of instructions";
  let description = [{
    This operation collects a sequence of instructions such that they can be
    placed as one unit. This is effectively the way to impose a constraint on
    the order and presence of some instructions.

    It is allowed to contain randomization constructs and invokations on any
    contexts. It is not allowed to create new context resources inside a
    sequence, however.

    This operation can be invoked by the `invoke` and `on_context` operations.
    It is referred to by symbol and isolated from above to ease multi-threading
    and it allows the `rtg.test` operation to be isolated-from-above to provide
    stronger top-level isolation guarantees.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $bodyRegion
  }];
}
