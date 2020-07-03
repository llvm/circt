//===- ExtractOps.td - LLHD extract operations -------------*- tablegen -*-===//
//
// This describes the MLIR ops for field and slice extractions.
//
//===----------------------------------------------------------------------===//

def LLHD_ExtsOp : LLHD_Op<"exts", [
    NoSideEffect,
    PredOpTrait<
      "'start' + size of the slice have to be smaller or equal to the 'target' "
      "size",
      CPred<"$start.cast<IntegerAttr>().getInt() + this->getSliceSize() <= "
            "this->getTargetSize()">>,
    SameTypeArbitraryWidth<
      "'target' and 'result' have to be both either signless integers, signals "
      "or vectors with the same element type",
      "$target", "$result">
  ]> {
  let summary = "Extract a slice of consecutive elements.";
  let description = [{
    The `llhd.exts` operation allows access to a slice of the `$target`
    operand. The `$start` attribute defines the index of the first element.
    The return type is the same as `$target` but with the width of the
    specified result type.
    If `%target` is a signal, a new subsignal aliasing the slice will be
    returned.

    Example:

    ```mlir
    %0 = llhd.const 123 : i32
    %1 = llhd.exts %0, 0 : i32 -> i2

    %2 = llhd.sig %0 : i32
    %3 = llhd.exts %2, 0 : !llhd.sig<i32> -> !llhd.sig<i5>
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[AnySignlessInteger, LLHD_AnySigType, AnyVector]>: $target,
    IndexAttr: $start);

  let results = (outs
    AnyTypeOf<[AnySignlessInteger, LLHD_AnySigType, AnyVector]>: $result);

  let assemblyFormat = [{
    $target `,` $start attr-dict `:` type($target) `->` type($result)
  }];

  let extraClassDeclaration = [{
    unsigned getSliceSize() {
      Type sliceType = result().getType();
      if (auto vec = sliceType.dyn_cast<VectorType>())
        return vec.getNumElements();
      if (auto sig = sliceType.dyn_cast<llhd::SigType>()) {
        Type ty = sig.getUnderlyingType();
        if (auto vec = ty.dyn_cast<VectorType>())
          return vec.getNumElements();
        if (auto tup = ty.dyn_cast<TupleType>())
          return tup.size();
        return ty.getIntOrFloatBitWidth();
      }
      return sliceType.getIntOrFloatBitWidth();
    }

    unsigned getTargetSize() {
      Type targetType = target().getType();
      if (auto vec = targetType.dyn_cast<VectorType>())
        return vec.getNumElements();
      if (auto sig = targetType.dyn_cast<llhd::SigType>()) {
        Type ty = sig.getUnderlyingType();
        if (auto vec = ty.dyn_cast<VectorType>())
          return vec.getNumElements();
        if (auto tup = ty.dyn_cast<TupleType>())
          return tup.size();
        return ty.getIntOrFloatBitWidth();
      }
      return targetType.getIntOrFloatBitWidth();
    }
  }];
}

def LLHD_DextsOp : LLHD_Op<"dexts", [
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
    The `llhd.dexts` operation allows to dynamically access a slice of the
    `$target` operand, starting at the index given by the `$start` operand.
    The resulting slice length is defined by the result type.
    The `$target` operand kind has to match the result kind.
    If `$target` is a vector, only the number of elements can change, while
    the element type has to remain the same.

    Example:

    ```mlir
    %0 = llhd.const 0x0f0 : i12
    %1 = llhd.const 4 : i3

    %3 = llhd.dexts %0, %1 : (i12, i3) -> i4    // %3: 0xf
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[AnySignlessInteger, AnyVector, LLHD_AnySigType]>: $target,
    AnySignlessInteger: $start);

  let results = (outs
    AnyTypeOf<[AnySignlessInteger, AnyVector, LLHD_AnySigType]>: $result);

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    unsigned getSliceWidth();
    unsigned getTargetWidth();
  }];
}
