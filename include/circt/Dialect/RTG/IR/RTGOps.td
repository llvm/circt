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

// Base class for the operation in this dialect.
class RTGOp<string mnemonic, list<Trait> traits = []> :
    Op<RTGDialect, mnemonic, traits>;


def SnippetOp : RTGOp<"snippet", [
  SingleBlock,
  NoRegionArguments,
  NoTerminator
]> {
  let summary = "a sequence of instructions";
  let description = [{
  }];

  let results = (outs SnippetType:$snippet);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "attr-dict-with-keyword $bodyRegion";
}

def SelectRandomOp : RTGOp<"select_random", [SameVariadicOperandSize]> {
  let summary = "places a random snippet in an instruction sequence";

  let arguments = (ins Variadic<SnippetType>:$snippets,
                       Variadic<I32>:$ratios);

  let assemblyFormat = "` ` `[` $snippets `]` `,` `[` $ratios `]` attr-dict";
}

def LabelOp : RTGOp<"label", []> {
  let summary = "places a label in an instruction sequence";

  // TODO: 'args' can be generalized to more types (maybe based on a
  // type interface?)
  let arguments = (ins StrAttr:$formatString,
                       Variadic<AnySignlessInteger>:$args);

  let assemblyFormat = "$formatString (`,` $args^ `:` type($args))? attr-dict";
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



def SelectRandomResource : RTGOp<"select_random_resource", [
  TypesMatchWith<"summary", "resources", "output", "::circt::rtg::ResourceType::get($_ctxt, llvm::cast<::circt::rtg::ResourceSetType>($_self).getElements())">
]> {
  let summary = "places a random resource in an instruction sequence";

  let arguments = (ins AnyResourceSetType:$resources);
  let results = (outs AnyResourceType:$output);

  let assemblyFormat = [{
    $resources `:` type($resources) attr-dict
  }];
}

//===- Resource Set Operations --------------------------------------------===//

def SelectRandomContextResource : RTGOp<"select_random_context_resource", [
  AllTypesMatch<["resources", "output"]>
]> {
  let summary = "places a random resource in an instruction sequence";

  let arguments = (ins ContextResourceSetType:$resources);
  let results = (outs ContextResourceSetType:$output);

  let assemblyFormat = "$resources `:` type($resources) attr-dict";
}

def SetDifferenceResource : RTGOp<"set_difference_resource", [
  AllTypesMatch<["resourcesSrc", "resourcesDiff", "output"]>
]> {
  let summary = "places a random resource in an instruction sequence";

  let arguments = (ins AnyResourceSetType:$resourcesSrc,
                       AnyResourceSetType:$resourcesDiff);
  let results = (outs AnyResourceSetType:$output);

  let assemblyFormat = [{
    $resourcesSrc `,` $resourcesDiff `:` type($output) attr-dict
  }];
}
