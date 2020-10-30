def SnippetOp : ESI_Op<"snippet",
      [IsolatedFromAbove, SymbolTable, RegionKindInterface,
       SingleBlockImplicitTerminator<"EndOp">]> {
  let summary = "ESI RTL Snippet";
  let description = [{
  }];

  let arguments = (ins StrAttr:$name);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static mlir::RegionKind getRegionKind(unsigned index) { return mlir::RegionKind::Graph;}

    // TODO(mlir): FunctionLike shouldn't produce a getBody() helper, it is
    // squatting on the name.
    mlir::Block *getBodyBlock() { return &body().front(); }
  }];

  // let printer = [{ return ::print(p, *this); }];
  // let parser = [{ return ::parse$cppClass(parser, result); }];
  let assemblyFormat = [{
    attr-dict $name $body
  }];
}

def EndOp : ESI_Op<"end", [Terminator]> {
  let summary = "ESI end of region op";
  let description = [{
  }];

  let arguments = (ins);
}
