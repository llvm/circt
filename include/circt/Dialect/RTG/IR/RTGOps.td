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
include "circt/Dialect/RTG/IR/RTGInterfaces.td"

// Base class for the operation in this dialect.
class RTGOp<string mnemonic, list<Trait> traits = []> :
    Op<RTGDialect, mnemonic, traits>;


def SequenceOp : RTGOp<"sequence", [
  SingleBlock,
  NoTerminator
]> {
  let summary = "a sequence of instructions";
  let description = [{
  }];

  let results = (outs SequenceType:$sequence);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    attr-dict-with-keyword $bodyRegion `->` qualified(type($sequence))
  }];

  let hasRegionVerifier = true;
}

def SelectRandomOp : RTGOp<"select_random", [AttrSizedOperandSegments]> {
  let summary = "places a random sequence in anohter sequence";

  let arguments = (ins
    Variadic<SequenceType>:$sequences,
    Variadic<I32>:$ratios,
    VariadicOfVariadic<AnyType,
                       "sequence_args_operand_segments">:$sequence_args,
    DenseI32ArrayAttr:$sequence_args_operand_segments);

  // TODO: write a nice custom assembly
  let assemblyFormat = [{
    ` ` `[` $sequences `]` `(` $sequence_args `:` type($sequence_args) `)`
    `,` `[` $ratios `]` attr-dict `:` type($sequences)
  }];

  let hasVerifier = true;
}

def LabelDeclOp : RTGOp<"label.decl", [
  DeclareOpInterfaceMethods<ResourceOpInterface>,
]> {
  let summary = "declares a label for an instruction sequence";
  let description = [{
    Declares a label that can then be placed by an `rtg.label` operation in an
    instruction sequence, passed on to sequences via their arguments, and used
    by instructions (e.g., as jump targets).
    This operation returns a value of the target architectures word size because
    a label is an alias for an immediate value of that will be determined after
    rendering, assemblying, and linking.
  }];

  // TODO: 'args' can be generalized to more types (maybe based on a
  // type interface?)
  let arguments = (ins StrAttr:$formatString,
                       Variadic<AnySignlessInteger>:$args);
  // Note: this is an index because it must match the machine word size
  let results = (outs AnySignlessInteger:$label);

  let assemblyFormat = [{
    $formatString (`,` $args^ `:` type($args))? attr-dict `->` type($label)
  }];
}

def LabelOp : RTGOp<"label", []> {
  let summary = "places a label in an instruction sequence";
  let description = [{
    Any declared label must only be placed at most once in any rendered
    instruction sequence. Otherwise, it would be ambiguous as a jump
    target.
    However, multiple snippets can place the same label. If they do so,
    they must never be instantiated in the same rendered sequence.
  }];

  let arguments = (ins AnySignlessInteger:$label, UnitAttr:$global);
  let assemblyFormat = "(`global` $global^)? $label attr-dict `:` type($label)";
}


//===----------------------------------------------------------------------===//



// Theory of structure:
// A sequence/snippet is expanded to all contexts for the current set of active 
// contexts.  This is maintained by the runtime.  The set of active contexts is
// changed by an OnContextOp.  The body of a sequence is expanded in textural 
// order.  This means a sequence can start adding operations to a context not
// in the current set of active contexts.

def OnContextOp : RTGOp<"on_context", [NoTerminator]> {
  let summary = "places a sequence on a context";

  let arguments = (ins ContextResourceSetType:$contexts);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    $contexts `:` type($contexts) attr-dict-with-keyword $bodyRegion
  }];
}

def RenderedContextOp : RTGOp<"rendered_context", 
    [SingleBlock, NoRegionArguments, NoTerminator]> {
  let summary = "a rendered context";
  let description = [{
  }];

  let regions = (region VariadicRegion<SizedRegion<1>>:$contextRegions);
  let arguments = (ins ArrayProperty<I64Property>:$selectors);
  let results = (outs);

  let assemblyFormat = "$selectors $contextRegions attr-dict-with-keyword";
}

def InvokeSequenceOp : RTGOp<"invoke", []> {
  let summary = "invoke a sequence of instructions";
  let description = [{
    Invoke a predefined sequence in place.
  }];

  let arguments = (ins SequenceType:$sequence);

  let assemblyFormat = [{
    $sequence attr-dict-with-keyword `:` type($sequence)
  }];
}


//===- Resource Set Operations --------------------------------------------===//

def SelectRandomResourceOp : RTGOp<"select_random_resource", [
  AllTypesMatch<["resources", "output"]>
]> {
  let summary = "places a random resource in an instruction sequence";

  let arguments = (ins ContextResourceSetType:$resources);
  let results = (outs ContextResourceSetType:$output);

  let assemblyFormat = "$resources `:` type($resources) attr-dict";
}

def SetDifferenceResourceOp : RTGOp<"set_difference_resource", [
  AllTypesMatch<["resourcesSrc", "resourcesDiff", "output"]>
]> {
  let summary = "places a random resource in an instruction sequence";

  let arguments = (ins ContextResourceSetType:$resourcesSrc,
                       ContextResourceSetType:$resourcesDiff);
  let results = (outs ContextResourceSetType:$output);

  let assemblyFormat = "$resourcesSrc `,` $resourcesDiff `:` type($output) attr-dict";
}
