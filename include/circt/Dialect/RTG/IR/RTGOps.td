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
include "circt/Dialect/RTG/IR/RTGInterfaces.td"

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

def SequenceClosureOp : RTGOp<"sequence_closure", [
  Pure,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "create a sequence closure with the provided arguments";
  let description = [{
    This operation creates a closure object for the provided sequence and
    arguments. This allows sequences to be passed around as an SSA value.
    For example, it can be inserted into a set and selected at random which
    is one of the main ways to do randomization. Not having to deal with
    sequence arguments after randomly selecting a sequence simplifies the
    problem of coming up with values to pass as arguments, but also provides a
    way for the user to constrain the arguments at the location where they are
    added to the set. In the future, it can also be possible to add sequence
    handles directly to a set and randomly pick arguments at the invokation
    site.
  }];

  let arguments = (ins SymbolNameAttr:$sequence, Variadic<AnyType>:$args);
  let results = (outs SequenceType:$ref);

  let assemblyFormat = [{
    $sequence (`(` $args^ `:` qualified(type($args)) `)`)? attr-dict
  }];
}

def InvokeSequenceOp : RTGOp<"invoke_sequence", []> {
  let summary = "invoke a sequence of instructions";
  let description = [{
    This operation takes a sequence closure as operand and acts as a placeholder
    for that sequence instantiated with the arguments in the closure in place.
    In particular, this is not any kind of function call, it doesn't set up a
    stack frame, etc. It behaves as if the sequence of instructions it refers to
    were directly inlined relacing this operation.
  }];

  let arguments = (ins SequenceType:$sequence);

  let assemblyFormat = "$sequence attr-dict";
}

//===- Set Operations ------------------------------------------------------===//

def SetCreateOp : RTGOp<"set_create", [Pure, SameTypeOperands]> {
  let summary = "constructs a set of the given values";

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs SetType:$set);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def SetSelectRandomOp : RTGOp<"set_select_random", [
  Pure,
  TypesMatchWith<"output must be of the element type of input set", 
                 "set", "output",
                 "llvm::cast<rtg::SetType>($_self).getElementType()">
]> {
  let summary = "selects an element uniformly at random from a set";
  let description = [{
    This operation returns an element from the given set uniformly at random.
    Applying this operation to an empty set is undefined behavior.
  }];

  let arguments = (ins SetType:$set);
  let results = (outs AnyType:$output);

  let assemblyFormat = "$set `:` qualified(type($set)) attr-dict";
}

def SetDifferenceOp : RTGOp<"set_difference", [
  Pure,
  AllTypesMatch<["original", "diff", "output"]>
]> {
  let summary = "computes the difference of two sets";

  let arguments = (ins SetType:$original,
                       SetType:$diff);
  let results = (outs SetType:$output);

  let assemblyFormat = [{
    $original `,` $diff `:` qualified(type($output)) attr-dict
  }];
}
