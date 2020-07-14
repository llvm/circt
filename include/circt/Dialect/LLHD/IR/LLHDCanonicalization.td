//===- LLHDCanonicalization.td - LLHD canonicalization -----*- tablegen -*-===//
//
// This file defines LLHD canonicalization patterns.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_IR_CANONICALIZATION_TD
#define CIRCT_DIALECT_LLHD_IR_CANONICALIZATION_TD

include "circt/Dialect/LLHD/IR/LLHD.td"

def XorAllBitsSet : Pat<
  (LLHD_XorOp $lhs, (LLHD_ConstOp AttrConstraint<
    CPred<"$_self.isa<IntegerAttr>() && "
      "$_self.cast<IntegerAttr>().getValue().isAllOnesValue()">,
    "signless integer -1">)),
  (LLHD_NotOp $lhs)>;

// TODO: not(eq(x,y)) -> neq(x,y)
// TODO: not(neq(x,y)) -> eq(x,y)
// TODO: eq(x:i1,y:i1) -> not(xor(x,y))
// TODO: neq(x:i1,y:i1) -> xor(x,y)
// TODO: dexts(target,constant c) -> exts(target, c)
// TODO: dextf(target,constant c) -> extf(target, c)

#endif // CIRCT_DIALECT_LLHD_IR_CANONICALIZATION_TD