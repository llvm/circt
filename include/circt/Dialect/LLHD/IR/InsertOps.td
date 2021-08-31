//===- InsertOps.td - LLHD insert operations ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for field and slice insertions.
//
//===----------------------------------------------------------------------===//

def LLHD_InsertSliceOp : LLHD_Op<"insert_slice", [
    NoSideEffect,
    AllTypesMatch<["target", "result"]>,
    SameTypeArbitraryWidth<
      "'target' and 'slice' have to be both either signless integers or arrays"
      " with the same element type",
      "$target", "$slice">,
    PredOpTrait<
      "'start' + size of the 'slice' have to be smaller or equal to the "
      "'target' size",
      CPred<
        "$start.cast<IntegerAttr>().getInt() + this->getSliceSize() <= "
        "this->getTargetSize()">>
  ]> {
  let summary = "Insert a slice of consecutive elements.";
  let description = [{
    The `llhd.insert_slice` operation allows insertion of a slice represented by
    the `$slice` operand into the `$target` operand. The `$start` attribute
    defines the index of the first element. The return type is the same as
    `$target`. Note that the `$target` is not changed, but a new value with
    the slice inserted is returned.

    Example:

    ```mlir
    %itarget = llhd.const 123 : i32
    %islice = llhd.const 2 : i2
    %0 = llhd.insert_slice %itarget, %islice, 0 : i32, i2

    %init1 = llhd.const 2 : i32
    %init2 = llhd.const 3 : i32
    %vtarget = llhd.array_uniform %init1 : !llhd.array<3xi32>
    %vslice = llhd.array_uniform %init2 : !llhd.array<2xi32>
    %1 = llhd.insert_slice %vtarget, %vslice, 0
      : !llhd.array<3xi32>, !llhd.array<2xi32>
    ```
  }];

  let arguments = (ins AnyTypeOf<[HWIntegerType, ArrayType]>: $target,
                       AnyTypeOf<[HWIntegerType, ArrayType]>: $slice,
                       IndexAttr: $start);

  let results = (outs AnyTypeOf<[HWIntegerType, ArrayType]>: $result);

  let assemblyFormat = [{
    $target `,` $slice `,` $start attr-dict `:` type($target) `,` type($slice)
  }];

  let extraClassDeclaration = [{
    unsigned getSliceSize() { return getLLHDTypeWidth(slice().getType()); }
    unsigned getTargetSize() { return getLLHDTypeWidth(target().getType()); }
  }];
}

def LLHD_InsertElementOp : LLHD_Op<"insert_element", [
    NoSideEffect,
    AllTypesMatch<["target", "result"]>,
    PredOpTrait<"'index' has to be smaller than the 'target' size",
      CPred<"$index.cast<IntegerAttr>().getInt() < this->getTargetSize()">>,
    TypesMatchWith<"'element' type has to match type at 'index' of 'target'",
      "target", "element",
      "this->getElementTypeAtIndex($index.cast<IntegerAttr>().getInt())">
  ]> {
  let summary = "Insert an element into an array or tuple.";
  let description = [{
    The `llhd.insert_element` operation allows insertion of an element
    represented by the `$element` operand into the `$target` operand. The
    `$index` attribute defines the index where to insert the element. The return
    type is the same as `$target`. Note that the `$target` is not changed, but a
    new value with the element inserted is returned.

    Example:

    ```mlir
    %init = llhd.const 1 : i8
    %target = llhd.array_uniform : !llhd.array<3xi8>
    %element = llhd.const 2 : i8
    %0 = llhd.insert_element %target, %element, 0 : !llhd.array<3xi8>, i8

    %tuptarget = llhd.tuple %element, %target : tuple<i8, !llhd.array<3xi8>
    %newelement = llhd.const 4 : i8
    %1 = llhd.insert_element %tuptarget, %newelement, 0
      : tuple<i8, !llhd.array<3xi8>>, i8
    ```
  }];

  let arguments = (ins AnyTypeOf<[ArrayType, StructType]>: $target,
                       AnyType: $element,
                       IndexAttr: $index);

  let results = (outs AnyTypeOf<[ArrayType, StructType]>: $result);

  let assemblyFormat = [{
    $target `,` $element `,` $index attr-dict `:`
    type($target) `,` type($element)
  }];

  let extraClassDeclaration = [{
    unsigned getTargetSize() { return getLLHDTypeWidth(target().getType()); }

    Type getElementTypeAtIndex(int64_t index) {
      Type targetType = target().getType();
      if (auto array = targetType.dyn_cast<hw::ArrayType>())
        return array.getElementType();
      return targetType.cast<hw::StructType>().getElements()[index].type;
    }
  }];
}
