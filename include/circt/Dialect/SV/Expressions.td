//===- Expressions.td - SV expression ops ------------------*- tablegen -*-===//
//
// This describes the ops for SystemVerilog expressions.
//
//===----------------------------------------------------------------------===//

def TextualValueOp : SVOp<"textual_value", [NoSideEffect]> {
  let summary = "Expression that expands to a value given SystemVerilog text";
  let description = [{
    This operation produces a typed value expressed by a string of
    SystemVerilog.  This can be used to access macros and other values that are
    only sensible as Verilog text.

    The text string is expected to have the highest precedence, so it should
    be parenthesized if it isn't a single token.
    }];

  let arguments = (ins StrAttr:$string);
  let results = (outs AnySignlessInteger:$result);

  let assemblyFormat = [{
     $string attr-dict `:` type($result)
  }];
}

// A list of things that the extract op knows with how to deal.
def Extractable : Type<Or<[
  CPred<"$_self.isa<::circt::sv::InterfaceType>()">
]>, "Types from which ports or values can be extracted.">;

def ExtractOp : SVOp<"extract", [NoSideEffect]> {
  let summary = "SystemVerilog '.' operator";
  let description = [{
    The SystemVerilog '.' operator can be used to access
    fields/members/modports in types (structs or unions) or interfaces. It is
    not the '.' used in instantiations to connect signals (`.clk(clk)`).

    Get the 'dataflow_in' modport on the 'handshake_example' interface:
    ```
    %ifaceModport = sv.extract @dataflow_in %iface : !sv.interface<@handshake_example>
    ```
  }];

  let arguments = (ins Extractable:$extractable, FlatSymbolRefAttr:$field);
  let results = (outs AnyType:$result);

  let printer = [{ printExtractOp(p, *this); }];
  let parser = [{ return parseExtractOp(parser, result); }];
}
