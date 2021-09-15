#ifndef HIR_OP_INTERFACES
#define HIR_OP_INTERFACES

include "mlir/IR/OpBase.td"

def FuncLike : OpInterface<"FuncLike"> {
let cppNamespace = "circt::hir"; 
  let description = [{
    This is an interface for hir.func like ops.
  }];

  let methods = [
    InterfaceMethod<
      "Get the hir::FuncType for this operation.",
      "hir::FuncType", "getFuncType"
    >,
    InterfaceMethod<
      "Get the Func body .",
      "Region&", "getFuncBody"
    >,
    InterfaceMethod<
      "Update func input args.",
      "void", "updateArguments", (ins "ArrayRef<DictionaryAttr>":$attrs)
    >,
    InterfaceMethod<
      "Get input args names.",
      "ArrayAttr", "getInputNames"
    >,
    InterfaceMethod<
      "Get input args names.",
      "ArrayAttr", "getResultNames"
    >,
  ];
}

#endif
