//===- OMOps.cpp - Object Model operation definitions ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model operation definitions.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/OMOps.h"

#include "mlir/IR/Builders.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ClassOp
//===----------------------------------------------------------------------===//

ParseResult circt::om::ClassOp::parse(OpAsmParser &parser,
                                      OperationState &state) {
  // Parse the Class symbol name.
  StringAttr symName;
  if (parser.parseSymbolName(symName, mlir::SymbolTable::getSymbolAttrName(),
                             state.attributes))
    return failure();

  // Parse the formal parameters.
  SmallVector<OpAsmParser::Argument> args;
  if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/false))
    return failure();

  // Parse the optional attribute dictionary.
  if (failed(parser.parseOptionalAttrDictWithKeyword(state.attributes)))
    return failure();

  // Parse the body.
  Region *region = state.addRegion();
  if (parser.parseRegion(*region, args))
    return failure();

  // If the region was empty, add an empty block so it's still a SizedRegion<1>.
  if (region->empty())
    region->emplaceBlock();

  // Remember the formal parameter names in an attribute.
  auto argNames = llvm::map_range(args, [&](OpAsmParser::Argument arg) {
    return StringAttr::get(parser.getContext(), arg.ssaName.name.drop_front());
  });
  state.addAttribute(
      "formalParamNames",
      ArrayAttr::get(parser.getContext(), SmallVector<Attribute>(argNames)));

  return success();
}

void circt::om::ClassOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               Twine name,
                               ArrayRef<StringRef> formalParamNames) {
  return build(odsBuilder, odsState, odsBuilder.getStringAttr(name),
               odsBuilder.getStrArrayAttr(formalParamNames));
}

void circt::om::ClassOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               Twine name) {
  return build(odsBuilder, odsState, odsBuilder.getStringAttr(name),
               odsBuilder.getStrArrayAttr({}));
}

void circt::om::ClassOp::print(OpAsmPrinter &printer) {
  // Print the Class symbol name.
  printer << " @";
  printer << getSymName();

  // Retrieve the formal parameter names and values.
  auto argNames = SmallVector<StringRef>(
      getFormalParamNames().getAsValueRange<StringAttr>());
  ArrayRef<BlockArgument> args = getBodyBlock()->getArguments();

  // Print the formal parameters.
  printer << '(';
  for (size_t i = 0, e = args.size(); i < e; ++i) {
    printer << '%' << argNames[i] << ": " << args[i].getType();
    if (i < e - 1)
      printer << ", ";
  }
  printer << ") ";

  // Print the optional attribute dictionary.
  SmallVector<StringRef> elidedAttrs{getSymNameAttrName(),
                                     getFormalParamNamesAttrName()};
  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                           elidedAttrs);

  // Print the body.
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

LogicalResult circt::om::ClassOp::verify() {
  // Verify the formal parameter names match up with the values.
  if (getFormalParamNames().size() != getBodyBlock()->getArguments().size()) {
    auto error = emitOpError(
        "formal parameter name list doesn't match formal parameter value list");
    error.attachNote(getLoc())
        << "formal parameter names: " << getFormalParamNames();
    error.attachNote(getLoc())
        << "formal parameter values: " << getBodyBlock()->getArguments();
    return error;
  }

  return success();
}

void circt::om::ClassOp::getAsmBlockArgumentNames(
    Region &region, OpAsmSetValueNameFn setNameFn) {
  // Retrieve the formal parameter names and values.
  auto argNames = SmallVector<StringRef>(
      getFormalParamNames().getAsValueRange<StringAttr>());
  ArrayRef<BlockArgument> args = getBodyBlock()->getArguments();

  // Use the formal parameter names as the SSA value names.
  for (size_t i = 0, e = args.size(); i < e; ++i)
    setNameFn(args[i], argNames[i]);
}

//===----------------------------------------------------------------------===//
// ObjectOp
//===----------------------------------------------------------------------===//

void circt::om::ObjectOp::build(::mlir::OpBuilder &odsBuilder,
                                ::mlir::OperationState &odsState,
                                om::ClassOp classOp,
                                ::mlir::ValueRange actualParams) {
  return build(odsBuilder, odsState,
               om::ClassType::get(odsBuilder.getContext(),
                                  mlir::FlatSymbolRefAttr::get(classOp)),
               classOp.getNameAttr(), actualParams);
}

LogicalResult
circt::om::ObjectOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Get the containing ModuleOp.
  auto moduleOp = getOperation()->getParentOfType<ModuleOp>();

  // Verify the result type is the same as the referred-to class.
  StringAttr resultClassName = getResult().getType().getClassName().getAttr();
  StringAttr className = getClassNameAttr();
  if (resultClassName != className)
    return emitOpError("result type (")
           << resultClassName << ") does not match referred to class ("
           << className << ')';

  // Verify the referred to ClassOp exists.
  auto classDef = dyn_cast_or_null<ClassOp>(
      symbolTable.lookupSymbolIn(moduleOp, className));
  if (!classDef)
    return emitOpError("refers to non-existant class (") << className << ')';

  auto actualTypes = getActualParams().getTypes();
  auto formalTypes = classDef.getBodyBlock()->getArgumentTypes();

  // Verify the actual parameter list matches the formal parameter list.
  if (actualTypes.size() != formalTypes.size()) {
    auto error = emitOpError(
        "actual parameter list doesn't match formal parameter list");
    error.attachNote(classDef.getLoc())
        << "formal parameters: " << classDef.getBodyBlock()->getArguments();
    error.attachNote(getLoc()) << "actual parameters: " << getActualParams();
    return error;
  }

  // Verify the actual parameter types match the formal parameter types.
  for (size_t i = 0, e = actualTypes.size(); i < e; ++i) {
    if (actualTypes[i] != formalTypes[i]) {
      return emitOpError("actual parameter type (")
             << actualTypes[i] << ") doesn't match formal parameter type ("
             << formalTypes[i] << ')';
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ObjectFieldOp
//===----------------------------------------------------------------------===//

LogicalResult
circt::om::ObjectFieldOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Get the containing ModuleOp.
  auto moduleOp = getOperation()->getParentOfType<ModuleOp>();

  // Get the ObjectInstOp and the ClassOp it is an instance of.
  ObjectOp objectInst = getObject().getDefiningOp<ObjectOp>();
  ClassOp classDef = cast<ClassOp>(
      symbolTable.lookupSymbolIn(moduleOp, objectInst.getClassNameAttr()));

  // Traverse the field path, verifying each field exists.
  Value finalField;
  auto fields = SmallVector<FlatSymbolRefAttr>(
      getFieldPath().getAsRange<FlatSymbolRefAttr>());
  for (size_t i = 0, e = fields.size(); i < e; ++i) {
    // Verify the field exists on the ClassOp.
    auto field = fields[i];
    ClassFieldOp fieldDef =
        cast_or_null<ClassFieldOp>(symbolTable.lookupSymbolIn(classDef, field));
    if (!fieldDef) {
      auto error = emitOpError("referenced non-existant field ") << field;
      error.attachNote(classDef.getLoc()) << "class defined here";
      return error;
    }

    // If there are more fields, verify the current field is of ClassType, and
    // look up the ClassOp for that field.
    if (i < e - 1) {
      auto classType = fieldDef.getValue().getType().dyn_cast<ClassType>();
      if (!classType)
        return emitOpError("nested field access into ")
               << field << " requires a ClassType, but found "
               << fieldDef.getValue().getType();

      // The nested ClassOp must exist, since a field with ClassType must be
      // an ObjectInstOp, which already verifies the class exists.
      classDef = cast<ClassOp>(
          symbolTable.lookupSymbolIn(moduleOp, classType.getClassName()));

      // Proceed to the next field in the path.
      continue;
    }

    // On the last iteration down the path, save the final field being accessed.
    finalField = fieldDef.getValue();
  }

  // Verify the accessed field type matches the result type.
  if (finalField.getType() != getResult().getType())
    return emitOpError("expected type ")
           << getResult().getType() << ", but accessed field has type "
           << finalField.getType();

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void circt::om::ConstantOp::build(::mlir::OpBuilder &odsBuilder,
                                  ::mlir::OperationState &odsState,
                                  ::mlir::TypedAttr constVal) {
  return build(odsBuilder, odsState, constVal.getType(), constVal);
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/OM/OM.cpp.inc"
