//===- HWAggregates.td - HW ops for structs/arrays/etc -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for working with aggregate values like structs
// and arrays.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWAGGREGATES_TD
#define CIRCT_DIALECT_HW_HWAGGREGATES_TD

include "circt/Dialect/HW/HWDialect.td"
include "circt/Dialect/HW/HWTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

def AggregateConstantOp : HWOp<"aggregate_constant", [Pure, ConstantLike]> {
  let summary = "Produce a constant aggregate value";
  let description = [{
    This operation produces a constant value of an aggregate type. Clock and
    reset values are supported. For nested aggregates, embedded arrays are
    used.

    Examples:
    ```mlir
      %result = hw.aggregate.constant [1 : i1, 2 : i2, 3 : i2] : !hw.struct<a: i8, b: i8, c: i8>
      %result = hw.aggregate.constant [1 : i1, [2 : i2, 3 : i2]] : !hw.struct<a: i8, b: vector<i8, 2>>
    ```
  }];

  let arguments = (ins ArrayAttr:$fields);
  let results = (outs HWAggregateType:$result);
  let assemblyFormat = "$fields attr-dict `:` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Packed Array Processing Operations
//===----------------------------------------------------------------------===//

def ArrayCreateOp : HWOp<"array_create", [Pure, SameTypeOperands]> {
  let summary = "Create an array from values";
  let description = [{
    Creates an array from a variable set of values. One or more values must be
    listed.

    ```
    // %a, %b, %c are all i4
    %array = hw.array_create %a, %b, %c : i4
    ```

    See the HW-SV rationale document for details on operand ordering.
  }];

  let arguments = (ins Variadic<HWNonInOutType>:$inputs);
  let results = (outs ArrayType:$result);

  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;

  let hasCustomAssemblyFormat = 1;
  let builders = [
    // ValueRange needs to contain at least one element.
    OpBuilder<(ins "ValueRange":$input)>
  ];

  let extraClassDeclaration = [{
    /// If the all elements of the array are identical, returns that element
    /// value. Otherwise returns a null value.
    Value getUniformElement();
    /// Returns true if all array elements are identical.
    bool isUniform() { return !!getUniformElement(); }
  }];
}

def ArrayConcatOp : HWOp<"array_concat", [Pure]> {
  let summary = "Concatenate some arrays";
  let description = [{
    Creates an array by concatenating a variable set of arrays. One or more
    values must be listed.

    ```
    // %a, %b, %c are hw arrays of i4 with sizes 2, 5, and 4 respectively.
    %array = hw.array_concat %a, %b, %c : (2, 5, 4 x i4)
    // %array is !hw.array<11 x i4>
    ```

    See the HW-SV rationale document for details on operand ordering.
  }];

  let arguments = (ins Variadic<ArrayType>:$inputs);
  let results = (outs ArrayType:$result);

  let assemblyFormat = [{
    $inputs attr-dict `:` custom<ArrayConcatTypes>(type($inputs), qualified(type($result)))
  }];

  let builders = [
    // ValueRange needs to contain at least one element.
    OpBuilder<(ins "ValueRange":$inputs)>
  ];
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def ArraySliceOp : HWOp<"array_slice", [Pure]> {
  let summary = "Get a range of values from an array";
  let description = [{
    Extracts a sub-range from an array. The range is from `lowIndex` to
    `lowIndex` + the number of elements in the return type, non-inclusive on
    the high end. For instance,

    ```
    // Slices 16 elements starting at '%offset'.
    %subArray = hw.slice %largerArray at %offset :
        (!hw.array<1024xi8>) -> !hw.array<16xi8>
    ```

    Width of 'idx' is defined to be the precise number of bits required to
    index the 'input' array. More precisely: for an input array of size M,
    the width of 'idx' is ceil(log2(M)). Lower and upper bound indexes which
    are larger than the size of the 'input' array results in undefined
    behavior.
  }];

  let arguments = (ins ArrayType:$input, HWIntegerType:$lowIndex);
  let results = (outs ArrayType:$dst);

  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;

  let assemblyFormat = [{
    $input`[`$lowIndex`]` attr-dict `:`
      `(` custom<SliceTypes>(type($input), qualified(type($lowIndex))) `)` `->` qualified(type($dst))
  }];
}

class IndexBitWidthConstraint<string index, string input> : PredOpTrait<
  index # " bit width equals ceil(log2(length(" # input # ")))," #
  " or 0 or 1 if " # input # " contains only one element",
  CPred<"isValidIndexBitWidth($" # index # ", $" # input # ")">
>;

class ArrayElementTypeConstraint<string result, string input> : TypesMatchWith<
  "must match array element type",
  input, result,
  "type_cast<ArrayType>($_self).getElementType()"
>;

def ArrayGetOp : HWOp<"array_get", [
  ArrayElementTypeConstraint<"result", "input">,
  IndexBitWidthConstraint<"index", "input">,
  Pure,
]> {
  let summary = "Extract an element from an array";
  let description = [{
    Extracts the element at `index` from the given `input` array. The index must
    be exactly `ceil(log2(length(input)))` bits wide.
  }];
  let arguments = (ins ArrayType:$input, HWIntegerType:$index);
  let results = (outs HWNonInOutType:$result);
  let assemblyFormat = [{
    $input `[` $index `]`
    attr-dict `:` type($input) `,` type($index)
  }];
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def ArrayInjectOp : HWOp<"array_inject", [
  AllTypesMatch<["input", "result"]>,
  ArrayElementTypeConstraint<"element", "input">,
  IndexBitWidthConstraint<"index", "input">,
  Pure,
]> {
  let summary = "Inject an element into an array";
  let description = [{
    Takes an `input` array, changes the element at `index` to the given
    `element` value, and returns the updated array value as a result. The index
    must be exactly `ceil(log2(length(input)))` bits wide. The element type
    must match the input array's element type.
  }];
  let arguments = (ins
    ArrayType:$input,
    HWIntegerType:$index,
    AnyType:$element
  );
  let results = (outs ArrayType:$result);
  let assemblyFormat = [{
    $input `[` $index `]` `,` $element
    attr-dict `:` type($input) `,` type($index)
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Structure Processing Operations
//===----------------------------------------------------------------------===//

def StructCreateOp : HWOp<"struct_create", [Pure]> {
  let summary = "Create a struct from constituent parts.";
  let arguments = (ins Variadic<HWNonInOutType>:$input);
  let results = (outs StructType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

// Extract the value of a field of a structure.
def StructExtractOp : HWOp<"struct_extract",
    [Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Extract a named field from a struct.";
  let description = [{
    ```
    %result = hw.struct_extract %input["field"] : !hw.struct<field: type>
    ```
  }];

  let arguments = (ins StructType:$input, I32Attr:$fieldIndex);
  let results = (outs HWNonInOutType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "StructType::FieldInfo":$field)>,
    OpBuilder<(ins "Value":$input, "StringAttr":$fieldName)>,
    OpBuilder<(ins "Value":$input, "StringRef":$fieldName), [{
      build($_builder, $_state, input, $_builder.getStringAttr(fieldName));
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the name attribute of the accessed field.
    StringAttr getFieldNameAttr() {
      StructType type = getInput().getType();
      return type.getElements()[getFieldIndex()].name;
    }

    /// Return the name of the accessed field.
    StringRef getFieldName() {
      return getFieldNameAttr().getValue();
    }
  }];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

// Create a structure by replacing a field with a value in an existing one.
def StructInjectOp : HWOp<"struct_inject", [Pure,
                           AllTypesMatch<["input", "result"]>]> {
  let summary = "Inject a value into a named field of a struct.";
  let description = [{
    ```
    %result = hw.struct_inject %input["field"], %newValue
        : !hw.struct<field: type>
    ```
  }];

  let arguments = (ins StructType:$input, I32Attr:$fieldIndex,
                   HWNonInOutType:$newValue);
  let results = (outs StructType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "StringAttr":$fieldName, "Value":$newValue)>,
    OpBuilder<(ins "Value":$input, "StringRef":$fieldName, "Value":$newValue), [{
      build($_builder, $_state, input, $_builder.getStringAttr(fieldName), newValue);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the name attribute of the accessed field.
    StringAttr getFieldNameAttr() {
      StructType type = getInput().getType();
      return type.getElements()[getFieldIndex()].name;
    }

    /// Return the name of the accessed field.
    StringRef getFieldName() {
      return getFieldNameAttr().getValue();
    }
  }];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def StructExplodeOp : HWOp<"struct_explode", [Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Expand a struct into its constituent parts.";
  let description = [{
    ```
    %result:2 = hw.struct_explode %input : !hw.struct<foo: i19, bar: i7>
    ```
  }];
  let arguments = (ins StructType:$input);
  let results = (outs Variadic<HWNonInOutType>:$result);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
 }

//===----------------------------------------------------------------------===//
// Union operations
//===----------------------------------------------------------------------===//

def UnionCreateOp : HWOp<"union_create", [Pure]> {
  let summary = "Create a union with the specified value.";
  let description = [{
    Create a union with the value 'input', which can then be accessed via the
    specified field.

    ```
      %x = hw.constant 0 : i3
      %z = hw.union_create "bar", %x : !hw.union<bar: i3, baz: i8>
    ```
  }];

  let arguments = (ins I32Attr:$fieldIndex, HWNonInOutType:$input);
  let results = (outs UnionType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Type":$unionType, "StringAttr":$fieldName, "Value":$input)>,
    OpBuilder<(ins "Type":$unionType, "StringRef":$fieldName, "Value":$input), [{
      build($_builder, $_state, unionType, $_builder.getStringAttr(fieldName), input);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the name attribute of the accessed field.
    StringAttr getFieldNameAttr() {
      UnionType type = getType();
      return type.getElements()[getFieldIndex()].name;
    }

    /// Return the name of the accessed field.
    StringRef getFieldName() {
      return getFieldNameAttr().getValue();
    }
  }];
}

def UnionExtractOp : HWOp<"union_extract", [Pure, 
    DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  ]> {
  let summary = "Get a union member.";
  let description = [{
    Get the value of a union, interpreting it as the type of the specified
    member field.  Extracting a value belonging to a different field than the
    union was initially created will result in undefined behavior.

    ```
      %u = ...
      %v = hw.union_extract %u["foo"] : !hw.union<foo: i3, bar: i16>
      // %v is of type 'i3'
    ```
  }];

  let arguments = (ins UnionType:$input, I32Attr:$fieldIndex);
  let results = (outs HWNonInOutType:$result);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "StringAttr":$fieldName)>,
    OpBuilder<(ins "Value":$input, "StringRef":$fieldName), [{
      build($_builder, $_state, input, $_builder.getStringAttr(fieldName));
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the name attribute of the accessed field.
    StringAttr getFieldNameAttr() {
      UnionType type = getInput().getType();
      return type.getElements()[getFieldIndex()].name;
    }

    /// Return the name of the accessed field.
    StringRef getFieldName() {
      return getFieldNameAttr().getValue();
    }
  }];
}

#endif // CIRCT_DIALECT_HW_HWAGGREGATES_TD
