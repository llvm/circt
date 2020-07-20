//===- ExtractOps.td - LLHD extract operations -------------*- tablegen -*-===//
//
// This describes the MLIR ops for field and slice extractions.
//
//===----------------------------------------------------------------------===//

def LLHD_ExtractSliceOp : LLHD_Op<"extract_slice", [
    NoSideEffect,
    PredOpTrait<
      "'start' + size of the slice have to be smaller or equal to the 'target' "
      "size",
      CPred<"$start.cast<IntegerAttr>().getInt() + this->getSliceSize() <= "
            "this->getTargetSize()">>,
    SameTypeArbitraryWidth<
      "'target' and 'result' have to be both either signless integers, signals "
      "or arrays with the same element type",
      "$target", "$result">
  ]> {
  let summary = "Extract a slice of consecutive elements.";
  let description = [{
    The `llhd.extract_slice` operation allows access to a slice of the `$target`
    operand. The `$start` attribute defines the index of the first element.
    The return type is the same as `$target` but with the width of the
    specified result type.
    If `%target` is a signal, a new subsignal aliasing the slice will be
    returned.

    Example:

    ```mlir
    %0 = llhd.const 123 : i32
    %1 = llhd.extract_slice %0, 0 : i32 -> i2

    %2 = llhd.sig %0 : i32
    %3 = llhd.extract_slice %2, 0 : !llhd.sig<i32> -> !llhd.sig<i5>
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[AnySignlessInteger, LLHD_ArrayType,
               LLHD_SigType<[AnySignlessInteger, LLHD_ArrayType]>]>: $target,
    IndexAttr: $start);

  let results = (outs
    AnyTypeOf<[AnySignlessInteger, LLHD_ArrayType,
               LLHD_SigType<[AnySignlessInteger, LLHD_ArrayType]>]>: $result);

  let assemblyFormat = [{
    $target `,` $start attr-dict `:` type($target) `->` type($result)
  }];

  let extraClassDeclaration = [{
    unsigned getSliceSize()  { return getLLHDTypeWidth(result().getType()); }
    unsigned getTargetSize() { return getLLHDTypeWidth(target().getType()); }
  }];
}

def LLHD_DynExtractSliceOp : LLHD_Op<"dyn_extract_slice", [
    NoSideEffect,
    SameTypeArbitraryWidth<
      "'target' and 'result' types have to match apart from their width",
      "$target", "$result">,
    PredOpTrait<
      "the result width cannot be larger than the target operand width",
      CPred<"this->getTargetWidth() >= this->getSliceWidth()">>
  ]> {
  let summary = "Dynamically extract a slice of consecutive elements";
  let description = [{
    The `llhd.dyn_extract_slice` operation allows to dynamically access a slice
    of the `$target` operand, starting at the index given by the `$start`
    operand. The resulting slice length is defined by the result type.
    The `$target` operand kind has to match the result kind.
    If `$target` is an array, only the number of elements can change, while
    the element type has to remain the same.

    Example:

    ```mlir
    %0 = llhd.const 0x0f0 : i12
    %1 = llhd.const 4 : i3

    %3 = llhd.dyn_extract_slice %0, %1 : (i12, i3) -> i4    // %3: 0xf
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[AnySignlessInteger, LLHD_ArrayType,
               LLHD_SigType<[AnySignlessInteger, LLHD_ArrayType]>]>: $target,
    AnySignlessInteger: $start);

  let results = (outs
    AnyTypeOf<[AnySignlessInteger, LLHD_ArrayType,
               LLHD_SigType<[AnySignlessInteger, LLHD_ArrayType]>]>: $result);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    unsigned getSliceWidth()  { return getLLHDTypeWidth(result().getType()); }
    unsigned getTargetWidth() { return getLLHDTypeWidth(target().getType()); }
  }];
}

def LLHD_ExtractElementOp : LLHD_Op<"extract_element", [
    NoSideEffect,
    PredOpTrait<"'index' has to be smaller than the width of the 'target' type",
      CPred<"$index.cast<IntegerAttr>().getInt() < getTargetWidth()">>,
    TypesMatchWith<
      "'result' type must match the type of 'target' at position 'index', or "
      "in case 'target' is a signal, it must be a signal of the underlying type"
      " of 'target' at position 'index'",
      "target", "result",
      "($_self.isa<llhd::SigType>() "
        "? llhd::SigType::get("
          "getElementTypeAtIndex($index.cast<IntegerAttr>().getInt())) "
        ": getElementTypeAtIndex($index.cast<IntegerAttr>().getInt()))">
  ]> {
  let summary = [{
    Extract an element from an array, tuple, or signal of an array or tuple.
  }];
  let description = [{
    The `llhd.extract_element` operation allows access to an element of the
    `$target` operand. The `$index` attribute defines the index of the element
    to extract. If `%target` is a signal, a new subsignal aliasing the element
    will be returned.

    Example:

    ```mlir
    %init = llhd.const 3 : i8
    %0 = llhd.array_uniform %init : !llhd.array<3xi8>
    %1 = llhd.extract_element %0, 0 : !llhd.array<3xi8> -> i8

    %2 = llhd.sig %0 : !llhd.array<3xi8>
    %3 = llhd.extract_element %2, 0
      : !llhd.sig<!llhd.array<3xi8>> -> !llhd.sig<i8>

    %4 = llhd.const 8 : i16
    %5 = llhd.tuple %0, %4 : tuple<!llhd.array<3xi8>, i16>
    %6 = llhd.extract_element %5, 1 : tuple<!llhd.array<3xi8>, i16> -> i16
    ```
  }];

  let arguments = (ins AnyTypeOf<[
      LLHD_ArrayType,
      AnyTuple,
      LLHD_SigType<[LLHD_ArrayType, AnyTuple]>
    ]>: $target,
    IndexAttr: $index);

  let results = (outs AnyType: $result);

  let assemblyFormat = [{
    $target `,` $index attr-dict `:` type($target) `->` type($result)
  }];

  let extraClassDeclaration = [{
    unsigned getTargetWidth() { return getLLHDTypeWidth(target().getType()); };

    Type getElementTypeAtIndex(unsigned index) {
      Type targetType = target().getType();
      if (auto sig = targetType.dyn_cast<llhd::SigType>())
        targetType = sig.getUnderlyingType();
      if (auto array = targetType.dyn_cast<llhd::ArrayType>())
        return array.getElementType();
      return targetType.cast<TupleType>().getTypes()[index];
    }
  }];
}

def LLHD_DynExtractElementOp : LLHD_Op<"dyn_extract_element", [
  NoSideEffect,
  TypesMatchWith<"'result' must be the element type of the 'target' array, in "
    "case 'target' is a signal of an array, 'result' also is a signal of the "
    "array element type",
    "target", "result",
    "($_self.isa<llhd::SigType>() ? llhd::SigType::get(getElementType()) "
                                 ": getElementType())">
  ]> {
  let summary = [{
    Dynamically extract an element from an array or signal of array.
  }];
  let description = [{
    The `llhd.dyn_extract_element` operation allows to dynamically access an
    element of the `$target` operand. The `$index` operand defines the index of
    the element to extract. If `%target` is a signal, a new subsignal aliasing
    the element will be returned.

    Example:

    ```mlir
    %index = llhd.const 1 : i2

    %init = llhd.const 3 : i8
    %0 = llhd.array_uniform %init : !llhd.array<3xi8>
    %1 = llhd.dyn_extract_element %0, %index : (!llhd.array<3xi8>, i2) -> i8

    %2 = llhd.sig %0 : !llhd.array<3xi8>
    %3 = llhd.dyn_extract_element %2, %index
      : (!llhd.sig<!llhd.array<3xi8>>, i2) -> !llhd.sig<i8>
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[LLHD_ArrayType, LLHD_SigType<[LLHD_ArrayType]>]>: $target,
    AnySignlessInteger: $index);

  let results = (outs AnyType: $result);

  let assemblyFormat = [{
    $target `,` $index attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    unsigned getTargetWidth() { return getLLHDTypeWidth(target().getType()); }

    Type getElementType() {
      Type targetType = target().getType();
      if (auto sig = targetType.dyn_cast<llhd::SigType>())
        targetType = sig.getUnderlyingType();
      return targetType.cast<llhd::ArrayType>().getElementType();
    }
  }];
}
