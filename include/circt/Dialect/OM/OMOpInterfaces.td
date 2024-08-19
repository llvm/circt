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
  let methods = [
    InterfaceMethod<"Get the class-like field to type map",
      "std::optional<mlir::Type>", "getFieldType", (ins "mlir::StringAttr":$name)>,
    InterfaceMethod<"Get the class-like field to type map",
      "void", "printClassFieldsLike", (ins "mlir::OpAsmPrinter &":$printer,
                                           "llvm::SmallVector<mlir::Value>":$operands),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ 
        mlir::Operation *op = $_op.getOperation();
        printer << "(";
        printer.increaseIndent();
        mlir::ArrayAttr fieldNames = mlir::cast<mlir::ArrayAttr>(op->getAttr("fieldNames"));

        mlir::ArrayRef<mlir::Attribute> perFieldAttrs;
        if (auto attrs = op->getAttr("perFieldAttrs"))
          perFieldAttrs = mlir::cast<mlir::ArrayAttr>(attrs).getValue();

        mlir::ArrayRef<mlir::Attribute> perFieldLocs;
        if (auto locs = op->getAttr("perFieldLocs"))
          perFieldLocs = mlir::cast<mlir::ArrayAttr>(locs).getValue();

        for (unsigned i = 0; i < fieldNames.size(); i++) {
          mlir::StringAttr nameAttr = mlir::cast<mlir::StringAttr>(fieldNames[i]);
          auto name = nameAttr.getValue();
          if (i > 0) {
            printer << ",";
          }
          printer.printNewline();
          printer.printSymbolName(name);
          if (!operands.empty()) {
            printer << " ";
            printer.printOperand(operands[i]);
          }
          printer << " : ";

          std::optional<mlir::Type> type = $_op.getFieldType(nameAttr);
          assert(type.has_value());
          printer.printType(type.value());
          if (!perFieldAttrs.empty())
            if (auto fieldAttr = mlir::dyn_cast<mlir::DictionaryAttr>(perFieldAttrs[i]))
              printer.printOptionalAttrDict(fieldAttr.getValue());
          if (!perFieldLocs.empty()) {
            mlir::Location loc = mlir::cast<mlir::Location>(perFieldLocs[i]);
            if (!mlir::isa<mlir::UnknownLoc>(loc))
              printer.printOptionalLocationSpecifier(loc);
          }
        }
        printer.decreaseIndent();
        if (!fieldNames.empty())
          printer.printNewline();
        printer << ")";
        printer.printOptionalAttrDict(op->getAttrs(),
                                      /*elidedAttrs=*/{"fieldTypes", "fieldNames",
                                                       "fieldIdxs", "perFieldAttrs",
                                                       "perFieldLocs"});
      }]>,
    InterfaceMethod<"Get the class-like field names",
      "llvm::ArrayRef<mlir::Attribute>", "getFieldNames", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ 
        mlir::Attribute fieldNames = this->getOperation()->getAttr("fieldNames");
        // TODO: Should we enforce an invariant that this attribute always
        // exists?
        if (!fieldNames) 
          return {};
        return
          mlir::cast<mlir::ArrayAttr>(fieldNames).getValue();
      }]>
  ];
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
      "circt::om::ClassFieldsLike", "getFieldsOp", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ 
        return mlir::cast<ClassFieldsLike>($_op.getBodyBlock()->getTerminator());
      }]>
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
