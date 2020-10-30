//===- ESIIntegrationOps.td - ESI integration --------------*- tablegen -*-===//
//
// Ops to assist in integration with existing RTL.
//
//===----------------------------------------------------------------------===//

def SnippetOp : ESI_Op<"snippet",
      [IsolatedFromAbove, SymbolTable, RegionKindInterface,
       SingleBlockImplicitTerminator<"EndOp">]> {
  let summary = "RTL snippet block";
  let description = [{
    A "snippet" represents a fragment of code which is intended to be
    included in existing code. In the SystemVerilog case, a .sv.inc file
    should be generated to be `included in existing SystemVerilog code.
  }];

  let arguments = (ins StrAttr:$name);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::Graph;
    }

    // TODO(mlir): FunctionLike shouldn't produce a getBody() helper, it is
    // squatting on the name.
    mlir::Block *getBodyBlock() { return &body().front(); }
  }];

  let assemblyFormat = [{
    attr-dict $name $body
  }];
}

def EndOp : ESI_Op<"end", [Terminator]> {
  let summary = "ESI end of region op";
  let description = [{
    No semantic meaning. Just a terminator. Should be ignored by virtually
    everything.
  }];

  let arguments = (ins);
}


def InstantiatedOp: ESI_Op<"instantiated",
                          []> {
  let summary = "A previously instantiated module";
  let description = [{
    This represents an already instantiated instance of a module. Must be
    located in a snippet and the name must match an instance already defined
    in the code above where the snippet is included. The module must be an
    "externmodule" must include a subset of the RTL modules ports which are
    relevant to the snippet.
  }];

  let arguments = (ins StrAttr:$instanceName,
                       FlatSymbolRefAttr:$moduleName,
                       Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    mlir::StringAttr getResultName(size_t i);
  }];

  //  custom<ResultNames>( attr-dict )
  let assemblyFormat = [{
    $instanceName $moduleName `(` $inputs `)` attr-dict
      `:` functional-type($inputs, results)
  }];

  let verifier = [{ return ::verifyInstantiatedOp(*this); }];
}
