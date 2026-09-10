//===- HLSOps.td - HLS op definitions ----------------------*- tablegen -*-===//

#ifndef CIRCT_DIALECT_HLS_HLSOPS_TD
#define CIRCT_DIALECT_HLS_HLSOPS_TD

include "circt/Dialect/Resource/HLS/HLSDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
//===----------------------------------------------------------------------===//
// StoreEnableOp
//===----------------------------------------------------------------------===//

def StoreEnableOp : HLSOp<"store_enable", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Predicated memref store; commits only when $enable is true.";
  let description = [{
    A flat, region-free conditional store. Writes $value to
    $memref[$indices] iff $enable holds at runtime. Models a BRAM write port
    whose write-enable line is driven by $enable

    Carries an *unconditional* Write memory effect: the predicate is an
    operand, not an effect. This is deliberate and conservative --
      * the scheduler reserves the write port in every cycle the op could
        fire (correct for an enable that may be true any iteration), and
      * the BRAM estimator, walking memory effects, sees exactly one write
        and zero reads, so a bank written only through store_enable infers
        RAM_1P instead of the RAM_2P that the RMW demux wrongly forced.

    Example:
      hls.store_enable %v, %en, %bank[%i] : memref<256xi32>
  }];

  let arguments = (ins AnyType:$value,
                       I1:$enable,
                       AnyMemRef:$memref,
                       Variadic<Index>:$indices);
  let results = (outs);

    let assemblyFormat = [{
      $value `,` $enable `,` $memref `[` $indices `]` attr-dict
      `:` type($value) `,` type($memref)
    }];

  let hasVerifier = 1;
}


def AffineStoreEnableOp : HLSOp<"affine.store_enable", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<AffineWriteOpInterface, ["getMapOperands"]>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Affine-level predicated store; commits iff $condition holds.";
  let description = [{
    Writes $value to $memref at affine $map applied to $mapOperands, but only
    when the integer set $condition (applied to $setOperands) is satisfied.
    The predicate is an affine_set, so the guard stays in the affine domain
    and MemoryDependenceAnalysis treats this as a real affine write -- giving
    it dependence edges automatically (no manual anchoring), and exposing
    loop-carried dependences for predicated accumulation.

    Lowers to hls.store_enable: $condition is materialized to i1, $map is
    expanded to index SSA values.
  }];

  let arguments = (ins AnyType:$value,
                       AnyMemRef:$memref,
                       Variadic<Index>:$mapOperands,
                       Variadic<Index>:$setOperands,
                       AffineMapAttr:$map,
                       IntegerSetAttr:$condition);
  let extraClassDeclaration = [{
      // Primitives the AffineWriteOpInterface defaults invoke.
      unsigned getMemRefOperandIndex()    { return 1; } // operand 1 = memref
      unsigned getStoredValOperandIndex() { return 0; } // operand 0 = value
      ::mlir::AffineMapAttr getAffineMapAttr() { return getMapAttr(); }
  }];                     
                       
  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

#endif // CIRCT_DIALECT_HLS_HLSOPS_TD
