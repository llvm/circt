//===- FIRRTL.td - FIRRTL dialect definition ---------------*- tablegen -*-===//
//
// This is the operation definition file for FIRRTL dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef FIRRTL_OPS
#define FIRRTL_OPS

include "mlir/IR/OpBase.td"

def FIRRTLDialect : Dialect {
  let name = "firrtl";

  let summary = "Types and operations for firrtl dialect";
  let description = [{
    This dialect defines the `firrtl` dialect, which is used to lower from
    Chisel code to Verilog.  For more information, see the
    [FIRRTL GitHub page](https://github.com/freechipsproject/firrtl).
  }];

  let cppNamespace = "firrtl";
}

def FIRRTL_UIntType : DialectType<FIRRTLDialect,
    CPred<"$_self.isa<UIntType>()">, "UInt type"> {
  let typeDescription = [{
    `firrtl.UInt` is an unsigned integer of unknown width.
  }];
}


//===----------------------------------------------------------------------===//
// FIRRTL op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class FIRRTLOp<string mnemonic, list<OpTrait> traits = []> :
    Op<FIRRTLDialect, mnemonic, traits>;


// TODO: Should this be split out to addu/adds?
def FIRRTLAddOp : FIRRTLOp<"add", [Commutative]> {
  let summary = "Add operations";
  let description = [{
    Add Operation: x = add (y, z)

    The two options are:
    * add(UInt, UInt) -> UInt
    * add(SInt, SInt) -> SInt
    
    The result width is max(we1, we2)+1
    The add operation result is the sum of e1 and e2 without loss of precision.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

#endif // FIRRTL_OPS
