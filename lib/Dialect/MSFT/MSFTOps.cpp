//===- MSFTOps.cpp - Implement MSFT dialect operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSFT dialect operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// Misc helper functions
//===----------------------------------------------------------------------===//

static bool hasAttribute(StringRef name, ArrayRef<NamedAttribute> attrs) {
  for (const auto &argAttr : attrs)
    if (argAttr.getName() == name)
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Custom directive parsers/printers
//===----------------------------------------------------------------------===//

static ParseResult parsePhysLoc(OpAsmParser &p, PhysLocationAttr &attr) {
  llvm::SMLoc loc = p.getCurrentLocation();
  StringRef devTypeStr;
  uint64_t x, y, num;

  if (p.parseKeyword(&devTypeStr) || p.parseKeyword("x") || p.parseColon() ||
      p.parseInteger(x) || p.parseKeyword("y") || p.parseColon() ||
      p.parseInteger(y) || p.parseKeyword("n") || p.parseColon() ||
      p.parseInteger(num))
    return failure();

  std::optional<PrimitiveType> devType = symbolizePrimitiveType(devTypeStr);
  if (!devType) {
    p.emitError(loc, "Unknown device type '" + devTypeStr + "'");
    return failure();
  }
  PrimitiveTypeAttr devTypeAttr =
      PrimitiveTypeAttr::get(p.getContext(), *devType);
  attr = PhysLocationAttr::get(p.getContext(), devTypeAttr, x, y, num);
  return success();
}

static void printPhysLoc(OpAsmPrinter &p, Operation *, PhysLocationAttr loc) {
  p << stringifyPrimitiveType(loc.getPrimitiveType().getValue())
    << " x: " << loc.getX() << " y: " << loc.getY() << " n: " << loc.getNum();
}

static ParseResult parseListOptionalRegLocList(OpAsmParser &p,
                                               LocationVectorAttr &locs) {
  SmallVector<PhysLocationAttr, 32> locArr;
  TypeAttr type;
  if (p.parseAttribute(type) || p.parseLSquare() ||
      p.parseCommaSeparatedList(
          [&]() { return parseOptionalRegLoc(locArr, p); }) ||
      p.parseRSquare())
    return failure();

  if (failed(LocationVectorAttr::verify(
          [&p]() { return p.emitError(p.getNameLoc()); }, type, locArr)))
    return failure();
  locs = LocationVectorAttr::get(p.getContext(), type, locArr);
  return success();
}

static void printListOptionalRegLocList(OpAsmPrinter &p, Operation *,
                                        LocationVectorAttr locs) {
  p << locs.getType() << " [";
  llvm::interleaveComma(locs.getLocs(), p, [&p](PhysLocationAttr loc) {
    printOptionalRegLoc(loc, p);
  });
  p << "]";
}

static ParseResult parseImplicitInnerRef(OpAsmParser &p,
                                         hw::InnerRefAttr &innerRef) {
  SymbolRefAttr sym;
  if (p.parseAttribute(sym))
    return failure();
  auto loc = p.getCurrentLocation();
  if (sym.getNestedReferences().size() != 1)
    return p.emitError(loc, "expected <module sym>::<inner name>");
  innerRef = hw::InnerRefAttr::get(
      sym.getRootReference(),
      sym.getNestedReferences().front().getRootReference());
  return success();
}
void printImplicitInnerRef(OpAsmPrinter &p, Operation *,
                           hw::InnerRefAttr innerRef) {
  MLIRContext *ctxt = innerRef.getContext();
  StringRef innerRefNameStr, moduleStr;
  if (innerRef.getName())
    innerRefNameStr = innerRef.getName().getValue();
  if (innerRef.getModule())
    moduleStr = innerRef.getModule().getValue();
  p << SymbolRefAttr::get(ctxt, moduleStr,
                          {FlatSymbolRefAttr::get(ctxt, innerRefNameStr)});
}

// /// Parse an parameter list if present. Same format as HW dialect.
// /// module-parameter-list ::= `<` parameter-decl (`,` parameter-decl)* `>`
// /// parameter-decl ::= identifier `:` type
// /// parameter-decl ::= identifier `:` type `=` attribute
// ///
// static ParseResult parseParameterList(OpAsmParser &parser,
//                                       SmallVector<Attribute> &parameters) {

//   return parser.parseCommaSeparatedList(
//       OpAsmParser::Delimiter::OptionalLessGreater, [&]() {
//         std::string name;
//         Type type;
//         Attribute value;

//         if (parser.parseKeywordOrString(&name) ||
//         parser.parseColonType(type))
//           return failure();

//         // Parse the default value if present.
//         if (succeeded(parser.parseOptionalEqual())) {
//           if (parser.parseAttribute(value, type))
//             return failure();
//         }

//         auto &builder = parser.getBuilder();
//         parameters.push_back(hw::ParamDeclAttr::get(
//             builder.getContext(), builder.getStringAttr(name), type, value));
//         return success();
//       });
// }

// /// Shim to also use this for the InstanceOp custom parser.
// static ParseResult parseParameterList(OpAsmParser &parser,
//                                       ArrayAttr &parameters) {
//   SmallVector<Attribute> parseParameters;
//   if (failed(parseParameterList(parser, parseParameters)))
//     return failure();

//   parameters = ArrayAttr::get(parser.getContext(), parseParameters);

//   return success();
// }

// /// Print a parameter list for a module or instance. Same format as HW
// dialect. static void printParameterList(OpAsmPrinter &p, Operation *op,
//                                ArrayAttr parameters) {
//   if (!parameters || parameters.empty())
//     return;

//   p << '<';
//   llvm::interleaveComma(parameters, p, [&](Attribute param) {
//     auto paramAttr = param.cast<hw::ParamDeclAttr>();
//     p << paramAttr.getName().getValue() << ": " << paramAttr.getType();
//     if (auto value = paramAttr.getValue()) {
//       p << " = ";
//       p.printAttributeWithoutType(value);
//     }
//   });
//   p << '>';
// }

// static ParseResult parseModuleLikeOp(OpAsmParser &parser,
//                                      OperationState &result,
//                                      bool withParameters = false) {
//   using namespace mlir::function_interface_impl;
//   auto loc = parser.getCurrentLocation();

//   // Parse the name as a symbol.
//   StringAttr nameAttr;
//   if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
//                              result.attributes))
//     return failure();

//   if (withParameters) {
//     // Parse the parameters
//     DictionaryAttr paramsAttr;
//     if (parser.parseAttribute(paramsAttr))
//       return failure();
//     result.addAttribute("parameters", paramsAttr);
//   }

//   // Parse the function signature.
//   bool isVariadic = false;
//   SmallVector<OpAsmParser::Argument, 4> entryArgs;
//   SmallVector<Attribute> argNames;
//   SmallVector<Attribute> argLocs;
//   SmallVector<Attribute> resultNames;
//   SmallVector<DictionaryAttr, 4> resultAttrs;
//   SmallVector<Attribute> resultLocs;
//   TypeAttr functionType;
//   if (hw::module_like_impl::parseModuleFunctionSignature(
//           parser, isVariadic, entryArgs, argNames, argLocs, resultNames,
//           resultAttrs, resultLocs, functionType))
//     return failure();

//   // If function attributes are present, parse them.
//   if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
//     return failure();

//   if (hasAttribute("argNames", result.attributes) ||
//       hasAttribute("resultNames", result.attributes)) {
//     parser.emitError(
//         loc, "explicit argNames and resultNames attributes not allowed");
//     return failure();
//   }

//   auto *context = result.getContext();
//   result.addAttribute("argNames", ArrayAttr::get(context, argNames));
//   result.addAttribute("argLocs", ArrayAttr::get(context, argLocs));
//   result.addAttribute("resultNames", ArrayAttr::get(context, resultNames));
//   result.addAttribute("resultLocs", ArrayAttr::get(context, resultLocs));
//   result.addAttribute(MSFTModuleOp::getFunctionTypeAttrName(result.name),
//                       functionType);

//   // Add the attributes to the module arguments.
//   addArgAndResultAttrs(parser.getBuilder(), result, entryArgs, resultAttrs,
//                        MSFTModuleOp::getArgAttrsAttrName(result.name),
//                        MSFTModuleOp::getResAttrsAttrName(result.name));

//   // Parse the optional module body.
//   auto regionSuccess =
//       parser.parseOptionalRegion(*result.addRegion(), entryArgs);
//   if (regionSuccess.has_value() && failed(*regionSuccess))
//     return failure();

//   return success();
// }

// template <typename ModuleTy>
// static void printModuleLikeOp(hw::HWModuleLike moduleLike, OpAsmPrinter &p,
//                               Attribute parameters = nullptr) {
//   using namespace mlir::function_interface_impl;

//   auto argTypes = moduleLike.getInputTypes();
//   auto resultTypes = moduleLike.getOutputTypes();

//   // Print the operation and the function name.
//   p << ' ';
//   p.printSymbolName(SymbolTable::getSymbolName(moduleLike).getValue());

//   if (parameters) {
//     // Print the parameterization.
//     p << ' ';
//     p.printAttribute(parameters);
//   }

//   p << ' ';
//   bool needArgNamesAttr = false;
//   hw::module_like_impl::printModuleSignature(p, moduleLike, argTypes,
//                                              /*isVariadic=*/false,
//                                              resultTypes, needArgNamesAttr);

//   SmallVector<StringRef, 3> omittedAttrs;
//   if (!needArgNamesAttr)
//     omittedAttrs.push_back("argNames");
//   omittedAttrs.push_back("argLocs");
//   omittedAttrs.push_back("resultNames");
//   omittedAttrs.push_back("resultLocs");
//   omittedAttrs.push_back("parameters");
//   omittedAttrs.push_back(
//       ModuleTy::getFunctionTypeAttrName(moduleLike->getName()));
//   omittedAttrs.push_back(ModuleTy::getArgAttrsAttrName(moduleLike->getName()));
//   omittedAttrs.push_back(ModuleTy::getResAttrsAttrName(moduleLike->getName()));

//   printFunctionAttributes(p, moduleLike, omittedAttrs);

//   // Print the body if this is not an external function.
//   Region &mbody = moduleLike.getModuleBody();
//   if (!mbody.empty()) {
//     p << ' ';
//     p.printRegion(mbody, /*printEntryBlockArgs=*/false,
//                   /*printBlockTerminators=*/true);
//   }
// }

//===----------------------------------------------------------------------===//
// DynamicInstanceOp
//===----------------------------------------------------------------------===//

ArrayAttr DynamicInstanceOp::getPath() {
  SmallVector<Attribute, 16> path;
  DynamicInstanceOp next = *this;
  do {
    path.push_back(next.getInstanceRefAttr());
    next = next->getParentOfType<DynamicInstanceOp>();
  } while (next);
  std::reverse(path.begin(), path.end());
  return ArrayAttr::get(getContext(), path);
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

void OutputOp::build(OpBuilder &odsBuilder, OperationState &odsState) {}

//===----------------------------------------------------------------------===//
// MSFT high level design constructs
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SystolicArrayOp
//===----------------------------------------------------------------------===//

ParseResult SystolicArrayOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  uint64_t numRows, numColumns;
  Type rowType, columnType;
  OpAsmParser::UnresolvedOperand rowInputs, columnInputs;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseLSquare() || parser.parseOperand(rowInputs) ||
      parser.parseColon() || parser.parseInteger(numRows) ||
      parser.parseKeyword("x") || parser.parseType(rowType) ||
      parser.parseRSquare() || parser.parseLSquare() ||
      parser.parseOperand(columnInputs) || parser.parseColon() ||
      parser.parseInteger(numColumns) || parser.parseKeyword("x") ||
      parser.parseType(columnType) || parser.parseRSquare())
    return failure();

  hw::ArrayType rowInputType = hw::ArrayType::get(rowType, numRows);
  hw::ArrayType columnInputType = hw::ArrayType::get(columnType, numColumns);
  SmallVector<Value> operands;
  if (parser.resolveOperands({rowInputs, columnInputs},
                             {rowInputType, columnInputType}, loc, operands))
    return failure();
  result.addOperands(operands);

  Type peOutputType;
  SmallVector<OpAsmParser::Argument> peArgs;
  if (parser.parseKeyword("pe")) {
    return failure();
  }
  llvm::SMLoc peLoc = parser.getCurrentLocation();
  if (parser.parseArgumentList(peArgs, AsmParser::Delimiter::Paren)) {
    return failure();
  }
  if (peArgs.size() != 2) {
    return parser.emitError(peLoc, "expected two operands");
  }

  peArgs[0].type = rowType;
  peArgs[1].type = columnType;

  if (parser.parseArrow() || parser.parseLParen() ||
      parser.parseType(peOutputType) || parser.parseRParen())
    return failure();

  result.addTypes({hw::ArrayType::get(
      hw::ArrayType::get(peOutputType, numColumns), numRows)});

  Region *pe = result.addRegion();

  peLoc = parser.getCurrentLocation();

  if (parser.parseRegion(*pe, peArgs))
    return failure();

  if (pe->getBlocks().size() != 1)
    return parser.emitError(peLoc, "expected one block for the PE");
  Operation *peTerm = pe->getBlocks().front().getTerminator();
  if (peTerm->getOperands().size() != 1)
    return peTerm->emitOpError("expected one return value");
  if (peTerm->getOperand(0).getType() != peOutputType)
    return peTerm->emitOpError("expected return type as given in parent: ")
           << peOutputType;

  return success();
}

void SystolicArrayOp::print(OpAsmPrinter &p) {
  hw::ArrayType rowInputType = getRowInputs().getType().cast<hw::ArrayType>();
  hw::ArrayType columnInputType =
      getColInputs().getType().cast<hw::ArrayType>();
  p << " [";
  p.printOperand(getRowInputs());
  p << " : " << rowInputType.getNumElements() << " x ";
  p.printType(rowInputType.getElementType());
  p << "] [";
  p.printOperand(getColInputs());
  p << " : " << columnInputType.getNumElements() << " x ";
  p.printType(columnInputType.getElementType());

  p << "] pe (";
  p.printOperand(getPe().getArgument(0));
  p << ", ";
  p.printOperand(getPe().getArgument(1));
  p << ") -> (";
  p.printType(getPeOutputs()
                  .getType()
                  .cast<hw::ArrayType>()
                  .getElementType()
                  .cast<hw::ArrayType>()
                  .getElementType());
  p << ") ";
  p.printRegion(getPe(), false);
}

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//

LogicalResult LinearOp::verify() {

  for (auto &op : *getBodyBlock()) {
    if (!isa<hw::HWDialect, comb::CombDialect, msft::MSFTDialect>(
            op.getDialect()))
      return emitOpError() << "expected only hw, comb, and msft dialect ops "
                              "inside the datapath.";
  }

  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/MSFT/MSFT.cpp.inc"
