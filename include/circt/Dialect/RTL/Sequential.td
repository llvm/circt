//===- Sequential.td - RTL sequential ops ------------------*- tablegen -*-===//
//
// Ops to support sequential logic. Clock type and memory essentially.
//
//===----------------------------------------------------------------------===//

// Don't extend 'RTLType' since a "clock" isn't generally considered a data type
// so much as a special wire.
def ClockType : TypeDef<RTLDialect, "Clock"> {
  let summary = "A pulse signal at which registers capture a value.";
  let description = [{ }];

  let mnemonic = "clk";
}
def Clock : TypeConstraint<CPred<"$_self.isa<::circt::rtl::ClockType>()">> {}


def RegisterOp : RTLOp<"reg", [NoSideEffect]> {
  let summary = "Store a value for one cycle.";
  let arguments = (ins AnyType:$in, Clock:$clk);
  let results = (outs AnyType:$out);
}
