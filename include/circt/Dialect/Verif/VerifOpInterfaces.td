#ifndef CIRCT_DIALECT_VERIF_VERIFOPINTERFACES_TD
#define CIRCT_DIALECT_VERIF_VERIFOPINTERFACES_TD

include "mlir/IR/OpBase.td"

def RequireLike : OpInterface<"RequireLike"> {
  let cppNamespace = "circt::verif";

  let description = [{
    Common functionality for require-like operations.
  }];

  let methods = [
    InterfaceMethod<"Get the label",
      "std::optional<llvm::StringRef>", "getLabel", (ins)>,
    InterfaceMethod<"Get the label attribute",
      "mlir::StringAttr", "getLabelAttr", (ins)>,
    InterfaceMethod<"Get the property",
      "mlir::Value", "getProperty", (ins)>,
    InterfaceMethod<"Get the enable",
      "mlir::Value", "getEnable", (ins)>,
  ];
}

#endif // CIRCT_DIALECT_VERIF_VERIFOPINTERFACES_TD
