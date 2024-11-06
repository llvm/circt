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
    attr-dict-with-keyword $bodyRegion `->` type($sequence)
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

def OnContextOp : RTGOp<"on_context", [NoTerminator, SingleBlock]> {
  let summary = "places a sequence on a context";

  let arguments = (ins AnyTypeOf<[SetTypeOf<ContextResourceType>, ContextResourceType]>:$contexts);
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

//===- Set Operations -----------------------------------------------------===//

def SetCreateOp : RTGOp<"set_create", [Pure, SameTypeOperands]> {
  let summary = "constructs a mode set";

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs SetType:$set);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def SetSelectRandomOp : RTGOp<"set_select_random", [
  Pure,
  TypesMatchWith<"output must be element type of input set", "set", "output", "llvm::cast<rtg::SetType>($_self).getElementType()">
]> {
  let summary = "places a random resource in an instruction sequence";

  let arguments = (ins SetType:$set);
  let results = (outs AnyType:$output);

  let assemblyFormat = "$set `:` type($set) attr-dict";
}

def SetDifferenceOp : RTGOp<"set_difference", [
  Pure,
  AllTypesMatch<["original", "diff", "output"]>
]> {
  let summary = "computes the difference of two sets";

  let arguments = (ins SetType:$original,
                       SetType:$diff);
  let results = (outs SetType:$output);

  let assemblyFormat = "$original `,` $diff `:` type($output) attr-dict";
}

def TestOp : RTGOp<"test", [IsolatedFromAbove, Symbol, SingleBlock, NoTerminator, HasParent<"mlir::ModuleOp">]> {
  let summary = "root of a test";

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<TargetType>:$targetType);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "$sym_name `:` $targetType attr-dict-with-keyword $bodyRegion";

  let hasRegionVerifier = 1;
}

def TargetOp : RTGOp<"target", [IsolatedFromAbove, Symbol, NoRegionArguments, SingleBlockImplicitTerminator<"rtg::YieldOp">, HasParent<"mlir::ModuleOp">]> {
  let summary = "collection of target information";
  let description = [{
    Users must add the `HasParent<"rtg::TargetInfo">` trait on all operations
    that can create `!rtg.context_resource` or `!rtg.mode` values.
    If users do not agree to this contract, the RTG framework behavior is
    undefined.


    ```mlir
    rtg.target.info @machine0 {
      %core_0 = my_target.coreid 0 : !rtg.context_resource
      %core_1 = my_target.coreid 1 : !rtg.context_resource
      %0 = rtg.set_create %core_0, %core_1 : !rtg.set<!rtg.context_resource>
      %user_mode = my_target.user_mode : !rtg.mode
      %machine_mode = my_target.machine_mode : !rtg.mode
      %supervisor_mode = my_target.supervisor_mode : !rtg.mode
      %1 = rtg.set_create %user_mode, %machine_mode, %supervisor_mode : !rtg.set<!rtg.mode>
      rtg.target.yield %0, %1 : !rtg.target<cpus: !rtg.set<!rtg.context_resource>, modes: !rtg.set<!rtg.mode>, machine_mode: !rtg.mode>
    }

    rtg.test @test0 {
    ^bb0(%target: !rtg.target<cpus: !rtg.set<!rtg.context_resource>, modes: !rtg.set<!rtg.mode>>, machine_mode: !rtg.mode):
      %0 = rtg.target.get %target["cpus"] : !rtg.target<cpus: !rtg.set<!rtg.context_resource>, modes: !rtg.set<!rtg.mode>, machine_mode: !rtg.mode>
      %1 = rtg.target.get %target["modes"] : !rtg.target<cpus: !rtg.set<!rtg.context_resource>, modes: !rtg.set<!rtg.mode>, machine_mode: !rtg.mode>
      rtg.on_context %0 mode %1 {

      }
    }

    // Alternative:

    rtg.target @machine0 {
      %core_0 = my_target.coreid 0 : !rtg.context_resource
      %core_1 = my_target.coreid 1 : !rtg.context_resource
      %0 = rtg.set_create %core_0, %core_1 : !rtg.set<!rtg.context_resource>
      %user_mode = my_target.user_mode : !rtg.mode
      %machine_mode = my_target.machine_mode : !rtg.mode
      %supervisor_mode = my_target.supervisor_mode : !rtg.mode
      %1 = rtg.set_create %user_mode, %machine_mode, %supervisor_mode : !rtg.set<!rtg.mode>
      rtg.yield %0, %1 : !rtg.set<!rtg.context_resource>, !rtg.set<!rtg.mode>, !rtg.mode
    }

    %seq = rtg.sequence @sequence_name {
    
    }

    rtg.mode_transition @transition_name {
    ^bb0(%from: !rtg.mode, %to: !rtg.mode):
      // Need to create mode here and compare against block argument
      %success = arith.constant true
      rtg.yield %success : i1
    }

    rtg.test @test0 {
    ^bb0(%cpus: !rtg.set<!rtg.context_resource>, $modes: !rtg.set<!rtg.mode>>, %machine_mode: !rtg.mode):
      rtg.on_context %cpus mode %modes { // region is instantiated once for every core-mode pair
       
      }
    }

    // Alternative:

    rtg.target @machine0 : !rtg.target<cpus: !rtg.set<!rtg.context_resource>, modes: !rtg.set<!rtg.mode>, machine_mode: !rtg.mode> {
      %core_0 = my_target.coreid 0 : !rtg.context_resource
      %core_1 = my_target.coreid 1 : !rtg.context_resource
      %0 = rtg.set_create %core_0, %core_1 : !rtg.set<!rtg.context_resource>
      // NOTE: users can also create their own mode creating operations, it
      // doesn't need to be an attribute for 'rtg.mode'. However, automatic
      // transitions are only supported when the mode can be specified using a
      // mode typed attribute. We could support such transitions by generalizing
      // the mode transition operation to accept two block arguments of mode
      // type and allowing creation and comparison of modes inside of it.
      // However, we'd need to make sure this cannot be abused to create new
      // modes on the fly.
      %user_mode = rtg.mode #my_target.user_mode : !rtg.mode
      %machine_mode = rtg.mode #my_target.machine_mode : !rtg.mode
      %supervisor_mode = rtg.mode #my_target.supervisor_mode : !rtg.mode
      %1 = rtg.set_create %user_mode, %machine_mode, %supervisor_mode : !rtg.set<!rtg.mode>
      rtg.yield %0, %1 : !rtg.set<!rtg.context_resource>, !rtg.set<!rtg.mode>, !rtg.mode
    }

    %seq = rtg.sequence @sequence_name {
    
    }

    rtg.mode_transition @transition_name from #my_target.user_mode to #my_target.machine_mode {
      // instruction sequence to transition here 
    }

    rtg.test @test0 : !rtg.target<cpus: !rtg.set<!rtg.context_resource>, modes: !rtg.set<!rtg.mode>, machine_mode: !rtg.mode> {
    ^bb0(%cpus: !rtg.set<!rtg.context_resource>, $modes: !rtg.set<!rtg.mode>>, %machine_mode: !rtg.mode):
      rtg.on_context %cpus mode %modes { // region is instantiated once for every core-mode pair
       
      }
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<TargetType>:$targetType);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "$sym_name `:` $targetType attr-dict-with-keyword $bodyRegion";

  let hasRegionVerifier = 1;
}

def YieldOp : RTGOp<"yield", [Pure, Terminator]> {
  let summary = "terminates RTG operation regions";

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}
