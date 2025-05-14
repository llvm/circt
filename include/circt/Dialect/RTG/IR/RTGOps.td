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

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/Properties.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "circt/Dialect/RTG/IR/RTGInterfaces.td"
include "circt/Dialect/RTG/IR/RTGISAAssemblyInterfaces.td"

// Base class for the operation in this dialect.
class RTGOp<string mnemonic, list<Trait> traits = []> :
    Op<RTGDialect, mnemonic, traits>;


def ConstantOp : RTGOp<"constant", [
  Pure,
  ConstantLike,
  DeclareOpInterfaceMethods<InferTypeOpInterface>
]> {
  let summary = "create an SSA value from an attribute";

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

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

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<SequenceType>:$sequenceType);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

def GetSequenceOp : RTGOp<"get_sequence", [
  Pure,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "create a sequence value";
  let description = [{
    This operation creates a sequence value referring to the provided sequence
    by symbol. It allows sequences to be passed around as an SSA value. For
    example, it can be inserted into a set and selected at random which is one
    of the main ways to do randomization.
  }];

  let arguments = (ins SymbolNameAttr:$sequence);
  let results = (outs SequenceType:$ref);

  let assemblyFormat = "$sequence `:` qualified(type($ref)) attr-dict";
}

def RandomizeSequenceOp : RTGOp<"randomize_sequence", []> {
  let summary = "randomize the content of a sequence";
  let description = [{
    This operation takes a fully substituted sequence and randomizes its
    content. This means, no operations the returned sequence does not contain
    any randomization constructs anymore (such as random selection from sets and
    bags, or other 'randomize_sequence' operations).

    It is useful to have this operation separate from 'embed_sequence' such that
    the exact same sequence (i.e., with the same random choices taken) can be
    embedded at multiple places.
    It is also useful to have this separate from sequence substitution because
    this operation is sensitive to the context, but the substitution values for
    a sequence family might already be available in a parent sequence that is
    placed on a different context. Thus, not having it separated would mean that
    the substitution values must all be passed down as arguments to the child
    sequence instead of a a single fully substituted sequence value.
  }];

  let arguments = (ins FullySubstitutedSequenceType:$sequence);

  let results = (outs RandomizedSequenceType:$randomizedSequence);

  let assemblyFormat = "$sequence attr-dict";
}

def SubstituteSequenceOp : RTGOp<"substitute_sequence", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "partially substitute arguments of a sequence family";
  let description = [{
    This operation substitutes the first N of the M >= N arguments of the given
    sequence family, where N is the size of provided argument substitution list.
    A new sequence (if N == M) or sequence family with M-N will be returned.
    
    Not having to deal with sequence arguments after randomly selecting a
    sequence simplifies the problem of coming up with values to pass as
    arguments, but also provides a way for the user to constrain the arguments
    at the location where they are added to a set or bag.
  }];

  let arguments = (ins SequenceType:$sequence, Variadic<AnyType>:$replacements);
  let results = (outs SequenceType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def EmbedSequenceOp : RTGOp<"embed_sequence", []> {
  let summary = "embed a sequence of instructions into another sequence";
  let description = [{
    This operation takes a fully randomized sequence and embeds it into another
    sequence or test at the position of this operation.
    In particular, this is not any kind of function call, it doesn't set up a
    stack frame, etc. It behaves as if the sequence of instructions it refers to
    were directly inlined relacing this operation.
  }];

  let arguments = (ins RandomizedSequenceType:$sequence);

  let assemblyFormat = "$sequence attr-dict";
}

def InterleaveSequencesOp : RTGOp<"interleave_sequences", [Pure]> {
  let summary = "interleave a list of sequences";
  let description = [{
    This operation takes a list of (at least one) fully randomized sequences and
    interleaves them by taking the next `batchSize` number of operations
    implementing the `InstructionOpInterface` of each sequence round-robin.

    Therefore, if only one sequence is in the list, this operation returns that
    sequence unchanged.
  }];

  let arguments = (ins Variadic<RandomizedSequenceType>:$sequences,
                       DefaultValuedAttr<I32Attr, "1">:$batchSize);

  let results = (outs RandomizedSequenceType:$interleavedSequence);

  let assemblyFormat = "$sequences (`batch` $batchSize^)? attr-dict";

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===- Label Operations ---------------------------------------------------===//

class LabelDeclBase<string mnemonic,
                    list<Trait> traits> : RTGOp<mnemonic, traits> {
  let description = [{
    Declares a label that can then be placed by an `rtg.label` operation in an
    instruction sequence, passed on to sequences via their arguments, and used
    by instructions (e.g., as jump targets) by allowing ISA dialects to use them
    directly as an operand of an instruction or by casting it to a value
    representing an immediate.

    The format string may contain placeholders of the form `{i}` where `i`
    refers to the i-th element in `args`.
    The declared label is uniqued by the compiler to no collide with any other
    label declarations.
  }];

  // TODO: 'args' can be generalized to more types
  let arguments = (ins StrAttr:$formatString, Variadic<Index>:$args);

  let assemblyFormat = [{
    $formatString (`,` $args^)? attr-dict
  }];
}

def LabelDeclOp : LabelDeclBase<"label_decl", [Pure]> {
  let summary = "declares a label for an instruction sequence";
  let results = (outs LabelType:$label);
}

def LabelUniqueDeclOp : LabelDeclBase<"label_unique_decl", []> {
  let summary = "declares a unique label for an instruction sequence";
  let results = (outs Res<LabelType, "", [MemAlloc]>:$label);
}

def LabelVisibilityAttr : I32EnumAttr<"LabelVisibility",
                                      "visibility specifiers for labels", [
  I32EnumAttrCase<"local",  0>,
  I32EnumAttrCase<"global", 1>,
  I32EnumAttrCase<"external", 2>,
]> {
  let cppNamespace = "::circt::rtg";
}

def LabelOp : RTGOp<"label", []> {
  let summary = "places a label in an instruction sequence";
  let description = [{
    Any declared label must only be placed at most once in any fully elaborated
    instruction sequence.
  }];

  let arguments = (ins LabelVisibilityAttr:$visibility, LabelType:$label);

  let assemblyFormat = "$visibility $label attr-dict";
}

//===- Context Operations -------------------------------------------------===//

def OnContextOp : RTGOp<"on_context", []> {
  let summary = "places a sequence on a context";
  let description = [{
    This operation takes a context and a fully substituted, but not yet
    randomized sequence and inserts the necessary instructions to switch from
    the current context to the provided context, randomizes and embeds the given
    sequence under the given context, and inserts instructions to switch back to
    the original context.

    These instructions are provided by the 'rtg.context_switch' operation. If no
    'rtg.context_switch' for this transition is provided, the compiler will
    error out. If multiple such context switches apply, the most recently
    registered one takes precedence.
  }];

  let arguments = (ins ContextResourceTypeInterface:$context,
                       FullySubstitutedSequenceType:$sequence);

  let assemblyFormat = [{
    $context `,` $sequence `:` qualified(type($context)) attr-dict
  }];
}

def ContextSwitchOp : RTGOp<"context_switch", [
  HasParent<"rtg::TargetOp">,
]> {
  let summary = "a specification of how to switch contexts";
  let description = [{
    This operation allows the user to specify a sequence of instructions to
    switch from context 'from' to context 'to', randomize and embed a provided
    sequence, and switch back from context 'to' to context 'from'.  This
    sequence of instructions should be provided as the 'sequence' operand which
    is a sequence of the type '!rtg.sequence<context-type-interface,
    context-type-interface, !rtg.sequence>'.  The first parameter is the 'from'
    context, the second one the 'to' context, and the third is the sequence to
    randomize and embed under the 'to' context.
  }];

  let arguments = (ins ContextResourceAttrInterface:$from,
                       ContextResourceAttrInterface:$to,
                       SequenceType:$sequence);

  let assemblyFormat = [{
    $from `->` $to `,` $sequence `:` qualified(type($sequence)) attr-dict
  }];

  let hasVerifier = 1;
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

def SetUnionOp : RTGOp<"set_union", [
  Pure, SameOperandsAndResultType, Commutative
]> {
  let summary = "computes the union of sets";
  let description = [{
    Computes the union of the given sets. The list of sets must contain at
    least one element.
  }];

  let arguments = (ins Variadic<SetType>:$sets);
  let results = (outs SetType:$result);

  let assemblyFormat = [{
    $sets `:` qualified(type($result)) attr-dict
  }];
}

def SetSizeOp : RTGOp<"set_size", [Pure]> {
  let summary = "returns the number of elements in the set";

  let arguments = (ins SetType:$set);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $set `:` qualified(type($set)) attr-dict
  }];
}

def SetCartesianProductOp : RTGOp<"set_cartesian_product", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "computes the n-ary cartesian product of sets";
  let description = [{
    This operation computes a set of tuples from a list of input sets such that
    each combination of elements from the input sets is present in the result
    set. More formally, for n input sets it computes
    `X_1 x ... x X_n = {(x_1, ..., x_n) | x_i \in X_i for i  \in {1, ..., n}}`.
    At least one input set has to be provided (i.e., `n > 0`).

    For example, given two sets A and B with elements
    `A = {a0, a1}, B = {b0, b1}` the result set R will be
    `R = {(a0, b0), (a0, b1), (a1, b0), (a1, b1)}`.

    Note that an RTG set does not provide any guarantees about the order of
    elements an can thus not be iterated over or indexed into, however, a
    random element can be selected and subtracted from the set until it is
    empty. This procedure is determinstic and will yield the same sequence of
    elements for a fixed seed and RTG version. If more guarantees about the
    order of elements is necessary, use arrays instead (and compute the
    cartesian product manually using nested loops).
  }];

  let arguments = (ins Variadic<SetType>:$inputs);
  let results = (outs SetType:$result);

  let assemblyFormat = "$inputs `:` qualified(type($inputs)) attr-dict";
}

def SetConvertToBagOp : RTGOp<"set_convert_to_bag", [
  Pure,
  TypesMatchWith<"element type of set must match the bag's element type",
                 "input", "result",
                 "BagType::get(cast<SetType>($_self).getElementType())">,
]> {
  let summary = "convert a set to a bag";
  let description = [{
    This operation converts a set to a bag. Each element in the set occurs
    exactly once in the resulting bag.
  }];

  let arguments = (ins SetType:$input);
  let results = (outs BagType:$result);

  let assemblyFormat = "$input `:` qualified(type($input)) attr-dict";
}

//===- Bag Operations ------------------------------------------------------===//

def BagCreateOp : RTGOp<"bag_create", [Pure, SameVariadicOperandSize]> {
  let summary = "constructs a bag";
  let description = [{
    This operation constructs a bag with the provided values and associated
    multiples. This means the bag constructed in the following example contains
    two of each `%arg0` and `%arg0` (`{%arg0, %arg0, %arg1, %arg1}`).

    ```mlir
    %0 = arith.constant 2 : index
    %1 = rtg.bag_create (%0 x %arg0, %0 x %arg1) : i32
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$elements,
                       Variadic<Index>:$multiples);
  let results = (outs BagType:$bag);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def BagSelectRandomOp : RTGOp<"bag_select_random", [
  Pure,
  TypesMatchWith<"output must be element type of input bag", "bag", "output",
                 "llvm::cast<rtg::BagType>($_self).getElementType()">
]> {
  let summary = "select a random element from the bag";
  let description = [{
    This operation returns an element from the bag selected uniformely at
    random. Therefore, the number of duplicates of each element can be used to
    bias the distribution.
    If the bag does not contain any elements, the behavior of this operation is
    undefined.
  }];

  let arguments = (ins BagType:$bag);
  let results = (outs AnyType:$output);

  let assemblyFormat = "$bag `:` qualified(type($bag)) attr-dict";
}

def BagDifferenceOp : RTGOp<"bag_difference", [
  Pure,
  AllTypesMatch<["original", "diff", "output"]>
]> {
  let summary = "computes the difference of two bags";
  let description = [{
    For each element the resulting bag will have as many fewer than the
    'original' bag as there are in the 'diff' bag. However, if the 'inf'
    attribute is attached, all elements of that kind will be removed (i.e., it
    is assumed the 'diff' bag has infinitely many copies of each element).
  }];

  let arguments = (ins BagType:$original,
                       BagType:$diff,
                       UnitAttr:$inf);
  let results = (outs BagType:$output);

  let assemblyFormat = [{
    $original `,` $diff (`inf` $inf^)? `:` qualified(type($output)) attr-dict
  }];
}

def BagUnionOp : RTGOp<"bag_union", [
  Pure, SameOperandsAndResultType, Commutative
]> {
  let summary = "computes the union of bags";
  let description = [{
    Computes the union of the given bags. The list of sets must contain at
    least one element.
  }];

  let arguments = (ins Variadic<BagType>:$bags);
  let results = (outs BagType:$result);

  let assemblyFormat = [{
    $bags `:` qualified(type($result)) attr-dict
  }];
}

def BagUniqueSizeOp : RTGOp<"bag_unique_size", [Pure]> {
  let summary = "returns the number of unique elements in the bag";
  let description = [{
    This operation returns the number of unique elements in the bag, i.e., for
    the bag `{a, a, b, c, c}` it returns 3.
  }];

  let arguments = (ins BagType:$bag);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $bag `:` qualified(type($bag)) attr-dict
  }];
}

def BagConvertToSetOp : RTGOp<"bag_convert_to_set", [
  Pure,
  TypesMatchWith<"element type of set must match the bag's element type",
                 "input", "result",
                 "SetType::get(cast<BagType>($_self).getElementType())">,
]> {
  let summary = "convert a bag to a set";
  let description = [{
    This operation converts a bag to a set by dropping all duplicate elements.
    For example, the bag `{a, a, b}` is converted to `{a, b}`.
  }];

  let arguments = (ins BagType:$input);
  let results = (outs SetType:$result);

  let assemblyFormat = "$input `:` qualified(type($input)) attr-dict";
}

//===- Array Operations -------------------------------------------------===//

def ArrayCreateOp : RTGOp<"array_create", [
  Pure,
  SameTypeOperands,
]> {
  let summary = "create an array with an initial list of elements";
  let description = [{
    This operation creates an array from a list of values. The element on the
    left-most position in the MLIR assembly format ends up at index 0.
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs ArrayType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def ArrayExtractOp : RTGOp<"array_extract", [
  Pure,
  TypesMatchWith<"result type must match array element type",
                 "array", "result",
                 "cast<ArrayType>($_self).getElementType()">,
]> {
  let summary = "get an element from an array";
  let description = [{
    This operation returns the element at the given index of the array.
    Accessing out-of-bounds indices is (immediate) UB.
  }];

  let arguments = (ins ArrayType:$array, Index:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $array `[` $index `]` `:` qualified(type($array)) attr-dict
  }];
}

def ArrayInjectOp : RTGOp<"array_inject", [
  Pure,
  AllTypesMatch<["array", "result"]>,
  TypesMatchWith<"value type must match array element type",
                 "array", "value",
                 "cast<ArrayType>($_self).getElementType()">,
]> {
  let summary = "set an element in an array";
  let description = [{
    This operation produces a new array of the same type as the input array and
    sets the element at the given index to the given value. All other values
    remain the same. An OOB access is (immediate) undefined behavior.
  }];

  let arguments = (ins ArrayType:$array, Index:$index, AnyType:$value);
  let results = (outs ArrayType:$result);

  let assemblyFormat = [{
    $array `[` $index `]` `,` $value `:` qualified(type($array)) attr-dict
  }];
}

def ArraySizeOp : RTGOp<"array_size", [Pure]> {
  let summary = "return the size of an array";

  let arguments = (ins ArrayType:$array);
  let results = (outs Index:$result);

  let assemblyFormat = "$array `:` qualified(type($array)) attr-dict";
}

//===- Tuple Operations ---------------------------------------------------===//

def TupleCreateOp : RTGOp<"tuple_create", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "create a tuple";

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs Builtin_Tuple:$result);

  let assemblyFormat = [{
    ($elements^ `:` qualified(type($elements)))? attr-dict
  }];
}

def TupleExtractOp : RTGOp<"tuple_extract", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "get an element from a tuple";

  let arguments = (ins Builtin_Tuple:$tuple, IndexAttr:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $tuple `at` $index `:` qualified(type($tuple)) attr-dict
  }];
}

//===- Integer Operations -------------------------------------------------===//

def RandomNumberInRangeOp : RTGOp<"random_number_in_range", []> {
  let summary = "returns a number uniformly at random within the given range";
  let description = [{
    This operation computes a random number based on a uniform distribution
    within the given range. The lower bound is inclusive while the upper bound
    is exclusive. If the range is empty, compilation will fail.
    This is (obviously) more performant than inserting all legal numbers into a
    set and using 'set_select_random', but yields the same behavior.
  }];

  let arguments = (ins Index:$lowerBound, Index:$upperBound);
  let results = (outs Index:$result);

  let assemblyFormat = "` ` `[` $lowerBound `,` $upperBound `)` attr-dict";
}

//===- Misc Operations ----------------------------------------------------===//

def CommentOp : RTGOp<"comment", []> {
  let summary = "emit a comment in instruction stream";

  let arguments = (ins StrAttr:$comment);

  let assemblyFormat = "$comment attr-dict";
}

//===- ISA Register Handling Operations -----------------------------------===//

def FixedRegisterOp : RTGOp<"fixed_reg", [
  Pure, ConstantLike,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "returns a value representing a fixed register";
  let description = [{
    This operation creates a value representing the register given as the 'reg'
    attribute. This is always a concrete ISA register.
    The return type always matches the register attribute type.
  }];

  let arguments = (ins RegisterAttrInterface:$reg);
  let results = (outs RegisterTypeInterface:$result);

  let assemblyFormat = "$reg attr-dict";
  let hasFolder = 1;
}

def VirtualRegisterOp : RTGOp<"virtual_reg", [
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "returns a value representing a virtual register";
  let description = [{
    This operation creates a value representing a virtual register. The
    'allowedRegisters' attribute specifies the concrete registers that may be
    chosen during register allocation.
  }];

  // ArrayAttr of RegisterAttrInterfaces
  let arguments = (ins ArrayAttr:$allowedRegs);
  let results = (outs RegisterTypeInterface:$result);

  let assemblyFormat = "$allowedRegs attr-dict";
  let hasVerifier = 1;
}

//===- Test Specification Operations --------------------------------------===//

def TestOp : RTGOp<"test", [
  IsolatedFromAbove,
  Symbol,
  SingleBlock,
  NoTerminator,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
  HasParent<"mlir::ModuleOp">
]> {
  let summary = "the root of a test";
  let description = [{
    This operation declares the root of a randomized or directed test.
    The target attribute specifies requirements of this test. These can be
    refined by `rtg.require` operations inside this operation's body. A test
    can only be matched with a target if the target fulfills all the test's
    requirements. However, the target may provide more than the test requires.
    For example, if the target allows execution in a user and privileged mode,
    but the test only requires and runs in user mode, it can still be matched
    with that target.

    By default each test can be matched with all targets that fulfill its
    requirements, but the user should be able to specify more constraints on the
    matching procedure.

    The body of this operation shall be processed the same way as an
    `rtg.sequence`'s body with the exception of the block arguments.
    The arguments must match the fields of the dict type in the target attribute
    exactly. The test must not have any additional arguments and cannot be
    referenced by an `rtg.get_sequence` operation.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<DictType>:$target);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

def TargetOp : RTGOp<"target", [
  IsolatedFromAbove,
  Symbol,
  NoRegionArguments,
  SingleBlockImplicitTerminator<"rtg::YieldOp">,
  HasParent<"mlir::ModuleOp">
]> {
  let summary = "defines a test target";
  let description = [{
    This operation specifies capabilities of a specific test target and can
    provide additional information about it. These are added as operands to the
    `yield` terminator and implicitly packed up into an `!rtg.dict` type which
    is passed to tests that are matched with this target.

    These capabilities can, for example, consist of the number of CPUs, supported
    priviledge modes, available memories, etc.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<DictType>:$target);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{
    $sym_name `:` $target attr-dict-with-keyword $bodyRegion
  }];

  let hasRegionVerifier = 1;
}

def YieldOp : RTGOp<"yield", [Pure, Terminator]> {
  let summary = "terminates RTG operation regions";

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

//===----------------------------------------------------------------------===//
// ISA Target Operations
//===----------------------------------------------------------------------===//

class RTGISAOp<string mnemonic, list<Trait> traits = []> :
    Op<RTGDialect, "isa." # mnemonic, traits>;

//===- ISA Immediate Handling Operations ----------------------------------===//

def IntToImmediateOp : RTGISAOp<"int_to_immediate", [Pure]> {
  let summary = "construct an immediate from an integer";
  let description = [{
    Create an immediate of static bit-width from the provided integer. If the
    integer does not fit in the specified bit-width, an error shall be emitted
    when executing this operation.
  }];

  let arguments = (ins Index:$input);
  let results = (outs ImmediateType:$result);

  let assemblyFormat = "$input `:` qualified(type($result)) attr-dict";
}

//===- ISA Memory Block Operations ----------------------------------------===//

def MemoryBlockDeclareOp : RTGISAOp<"memory_block_declare", [
  HasParent<"rtg::TargetOp">,
]> {
  let summary = "declare a memory block with the provided properties";
  let description = [{
    This operation declares a memory block to be allocated with the provided
    properties. It is only allowed to declare new memory blocks in the
    `rtg.target` operations and must be passed as argument to the `rtg.test`.
    This is because the available memory blocks are specified by the hardware
    design. This specification is fixed from the start and thus a test should
    not be able to declare new memory blocks on-the-fly. However, tests are
    allowed to allocate memory regions from these memory blocks.

    The 'baseAddress' attribute specifies the first memory address (lowest
    address representing a valid access to the memory) and the 'endAddress'
    represents the last address (highest address that is valid to access the
    memory).
  }];

  let arguments = (ins APIntAttr:$baseAddress, APIntAttr:$endAddress);
  let results = (outs MemoryBlockType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===- ISA Memory Handling Operations -------------------------------------===//

def MemoryAllocOp : RTGISAOp<"memory_alloc", [
  TypesMatchWith<"memory must have the same address width as the memory block",
                 "memoryBlock", "result",
                 "MemoryType::get($_ctxt, " #
                 "cast<MemoryBlockType>($_self).getAddressWidth())">,
]> {
  let summary = "allocate a memory with the provided properties";
  let description = [{
    This operation declares a memory to be allocated with the provided
    properties. It is only allowed to declare new memories in the `rtg.target`
    operations and must be passed as argument to the `rtg.test`.
  }];

  let arguments = (ins MemoryBlockType:$memoryBlock,
                       Index:$size,
                       Index:$alignment);

  let results = (outs MemoryType:$result);

  let assemblyFormat = [{
    $memoryBlock `,` $size `,` $alignment
    `:` qualified(type($memoryBlock)) attr-dict
  }];
}

def MemoryBaseAddressOp : RTGISAOp<"memory_base_address", [
  Pure,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "get the memory base address as an immediate";
  let description = [{
    This operation returns the base address of the given memory. The bit-width
    of the returned immediate must match the address width of the given memory.
  }];

  let arguments = (ins MemoryType:$memory);
  let results = (outs ImmediateType:$result);

  let assemblyFormat = "$memory `:` qualified(type($memory)) attr-dict";
}

def MemorySizeOp : RTGISAOp<"memory_size", [Pure]> {
  let summary = "get the size of the memory in bytes";

  let arguments = (ins MemoryType:$memory);
  let results = (outs Index:$result);

  let assemblyFormat = "$memory `:` qualified(type($memory)) attr-dict";
}
