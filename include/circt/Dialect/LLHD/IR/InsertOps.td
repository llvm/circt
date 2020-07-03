//===- InsertOps.td - LLHD insert operations ---------------*- tablegen -*-===//
//
// This describes the MLIR ops for field and slice insertions.
//
//===----------------------------------------------------------------------===//

def LLHD_InssOp : LLHD_Op<"inss", [
    NoSideEffect,
    AllTypesMatch<["target", "result"]>,
    SameTypeArbitraryWidth<
      "'target' and 'slice' have to be both either signless integers or vectors"
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
    The `llhd.inss` operation allows insertion of a slice represented by the
    `$slice` operand into the `$target` operand. The `$start` attribute
    defines the index of the first element. The return type is the same as
    `$target`. Note that the `$target` is not changed, but a new value with
    the slice inserted is returned.

    Example:

    ```mlir
    %itarget = llhd.const 123 : i32
    %islice = llhd.const 2 : i2
    %0 = llhd.inss %itarget, %islice, 0 : i32, i2

    %vtarget = constant dense<[1,2,3]> : vector<3xi32>
    %vslice = constant dense<[4,5]> : vector<2xi32>
    %1 = llhd.inss %vtarget, %vslice, 0 : vector<3xi32>, vector<2xi32>
    ```
  }];

  let arguments = (ins AnyTypeOf<[AnySignlessInteger, AnyVector]>: $target,
                       AnyTypeOf<[AnySignlessInteger, AnyVector]>: $slice,
                       IndexAttr: $start);

  let results = (outs AnyTypeOf<[AnySignlessInteger, AnyVector]>: $result);

  let assemblyFormat = [{
    $target `,` $slice `,` $start attr-dict `:` type($target) `,` type($slice)
  }];

  let extraClassDeclaration = [{
    unsigned getSliceSize() { return getLLHDTypeWidth(slice().getType()); }
    unsigned getTargetSize() { return getLLHDTypeWidth(target().getType()); }
  }];
}

def LLHD_InsfOp : LLHD_Op<"insf", [
    NoSideEffect,
    AllTypesMatch<["target", "result"]>,
    PredOpTrait<"'index' has to be smaller than the 'target' size",
      CPred<"$index.cast<IntegerAttr>().getInt() < this->getTargetSize()">>,
    TypesMatchWith<"'element' type has to match type at 'index' of 'target'",
      "target", "element",
      "this->getElementTypeAtIndex($index.cast<IntegerAttr>().getInt())">
  ]> {
  let summary = "Insert an element into a vector or tuple.";
  let description = [{
    The `llhd.insf` operation allows insertion of an element represented by
    the `$element` operand into the `$target` operand. The `$index`
    attribute defines the index where to insert the element. The return type
    is the same as `$target`. Note that the `$target` is not changed, but a
    new value with the element inserted is returned.

    Example:

    ```mlir
    %target = constant dense<[1,2,3]> : vector<3xi8>
    %element = llhd.const 2 : i8
    %0 = llhd.insf %target, %element, 0 : vector<3xi8>, i8

    %tuptarget = llhd.tuple %element, %target : tuple<i8, vector<3xi8>
    %newelement = llhd.const 4 : i8
    %1 = llhd.insf %tuptarget, %newelement, 0 : tuple<i8, vector<3xi8>>, i8
    ```
  }];

  let arguments = (ins AnyTypeOf<[AnyVector, AnyTuple]>: $target,
                       AnyType: $element,
                       IndexAttr: $index);

  let results = (outs AnyTypeOf<[AnyVector, AnyTuple]>: $result);

  let assemblyFormat = [{
    $target `,` $element `,` $index attr-dict `:`
    type($target) `,` type($element)
  }];

  let extraClassDeclaration = [{
    unsigned getTargetSize() { return getLLHDTypeWidth(target().getType()); }

    Type getElementTypeAtIndex(int64_t index) {
      Type targetType = target().getType();
      if (auto vec = targetType.dyn_cast<VectorType>())
        return vec.getElementType();
      return targetType.cast<TupleType>().getTypes()[index];
    }
  }];
}
