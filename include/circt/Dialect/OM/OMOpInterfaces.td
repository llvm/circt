//===- OMOpInterfaces.td - Object Model dialect op interfaces -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the Object Model dialect operation interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OMOPINTERFACES_TD
#define CIRCT_DIALECT_OM_OMOPINTERFACES_TD

include "mlir/IR/OpBase.td"

def ClassFieldsLike : OpInterface<"ClassFieldsLike"> {
  let cppNamespace = "circt::om";

  let description = [{
    Marks the end of a class-like definition by providing a list of accessible
    fields.
  }];

}

def ClassLike : OpInterface<"ClassLike"> {
  let cppNamespace = "circt::om";

  let description = [{
    Common functionality for class-like operations.
  }];

  let methods = [
    InterfaceMethod<"Get the class-like symbol name",
      "llvm::StringRef", "getSymName", (ins)>,
    InterfaceMethod<"Get the class-like symbol name attribute",
      "mlir::StringAttr", "getSymNameAttr", (ins)>,
    InterfaceMethod<"Get the class-like symbol name attribute name",
      "mlir::StringAttr", "getSymNameAttrName", (ins)>,
    InterfaceMethod<"Get the class-like formal parameter names attribute",
      "mlir::ArrayAttr", "getFormalParamNames", (ins)>,
    InterfaceMethod<"Get the class-like formal parameter names attribute name",
      "mlir::StringAttr", "getFormalParamNamesAttrName", (ins)>,
    InterfaceMethod<"Get the class-like body region",
      "mlir::Region &", "getBody", (ins)>,
    InterfaceMethod<"Get the class-like body block",
      "mlir::Block *", "getBodyBlock", (ins),
      /*methodBody=*/[{ return $_op.getBodyBlock(); }]>,
    InterfaceMethod<"Get the class-like fields op",
      "circt::om::ClassFieldsLike", "getFieldsOp", (ins)>,
    InterfaceMethod<"Get the class-like fields objects",
      "llvm::SmallVector<circt::om::Field>", "getFields", (ins)>,
    InterfaceMethod<"Get the class-like field to type map",
      "std::optional<llvm::MapVector<mlir::StringAttr, mlir::Type>>", "getFieldTypes", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ 
        llvm::MapVector<mlir::StringAttr, mlir::Type> classFields;
        for (auto field : $_op.getFields()) {
          auto result = classFields.insert(
            {field.getName(), field.getType()}
          );
          /// TODO: Can we do this verification on construction?
          if (!result.second) {
            $_op.getFieldsOp().emitOpError() << "field " << field.getName()
            << " is defined twice";
            return std::nullopt;
          }
        }
        return std::optional<llvm::MapVector<mlir::StringAttr, mlir::Type>>(classFields);
      }]>,
  ];
}

def ClassFieldLike : OpInterface<"ClassFieldLike"> {
  let cppNamespace = "circt::om";

  let description = [{
    Common functionality for class-like field operations.
  }];

  let methods = [
    InterfaceMethod<"Get the class-like field's type",
      "mlir::Type", "getType", (ins)>,
    InterfaceMethod<"Get the class-like field's name attribute",
      "mlir::StringAttr", "getNameAttr", (ins)>
  ];
}

def IntegerBinaryArithmeticInterface : OpInterface<"IntegerBinaryArithmeticOp"> {
  let cppNamespace = "circt::om";
  let description = "Common interface for integer binary arithmetic ops.";
  let methods = [
    InterfaceMethod<"Get the lhs Value",
      "mlir::Value", "getLhs", (ins)>,
    InterfaceMethod<"Get the rhs Value",
      "mlir::Value", "getRhs", (ins)>,
    InterfaceMethod<"Get the result Value",
      "mlir::Value", "getResult", (ins)>,
    InterfaceMethod<"Evaluate the integer binary arithmetic operation",
      "mlir::FailureOr<llvm::APSInt>", "evaluateIntegerOperation",
      (ins "const llvm::APSInt &":$lhs, "const llvm::APSInt &":$rhs)>
  ];
}

#endif // CIRCT_DIALECT_OM_OMOPINTERFACES_TD
