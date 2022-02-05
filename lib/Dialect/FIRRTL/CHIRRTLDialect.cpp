//===- CHIRRTLDialect.cpp - Implement the CHIRRTL dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CHIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace chirrtl;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Parsing and Printing helpers.
//===----------------------------------------------------------------------===//

static ParseResult parseCHIRRTLOp(OpAsmParser &parser,
                                  NamedAttrList &resultAttrs) {
  // Add an empty annotation array if none were parsed.
  auto result = parser.parseOptionalAttrDict(resultAttrs);
  if (!resultAttrs.get("annotations"))
    resultAttrs.append("annotations", parser.getBuilder().getArrayAttr({}));

  // If the attribute dictionary contains no 'name' attribute, infer it from
  // the SSA name (if specified).
  if (resultAttrs.get("name"))
    return success();

  auto resultName = parser.getResultName(0).first;
  if (!resultName.empty() && isdigit(resultName[0]))
    resultName = "";
  auto nameAttr = parser.getBuilder().getStringAttr(resultName);
  auto *context = parser.getBuilder().getContext();
  resultAttrs.push_back({StringAttr::get(context, "name"), nameAttr});
  return result;
}

static void printCHIRRTLOp(OpAsmPrinter &p, Operation *op, DictionaryAttr attr,
                           ArrayRef<StringRef> extraElides = {}) {
  SmallVector<StringRef> elides(extraElides.begin(), extraElides.end());

  // Elide the symbol.
  elides.push_back(hw::InnerName::getInnerNameAttrName());

  // Note that we only need to print the "name" attribute if the asmprinter
  // result name disagrees with it.  This can happen in strange cases, e.g.
  // when there are conflicts.
  SmallString<32> resultNameStr;
  llvm::raw_svector_ostream tmpStream(resultNameStr);
  p.printOperand(op->getResult(0), tmpStream);
  auto actualName = tmpStream.str().drop_front();
  auto expectedName = op->getAttrOfType<StringAttr>("name").getValue();
  // Anonymous names are printed as digits, which is fine.
  if (actualName == expectedName ||
      (expectedName.empty() && isdigit(actualName[0])))
    elides.push_back("name");

  // Elide "annotations" if it is empty.
  if (op->getAttrOfType<ArrayAttr>("annotations").empty())
    elides.push_back("annotations");

  p.printOptionalAttrDict(op->getAttrs(), elides);
}

//===----------------------------------------------------------------------===//
// MemoryPortOp
//===----------------------------------------------------------------------===//

void MemoryPortOp::build(OpBuilder &builder, OperationState &result,
                         Type dataType, Value memory, MemDirAttr direction,
                         StringRef name, ArrayRef<Attribute> annotations) {
  build(builder, result, CMemoryPortType::get(builder.getContext()), dataType,
        memory, direction, name, builder.getArrayAttr(annotations));
}

LogicalResult MemoryPortOp::inferReturnTypes(MLIRContext *context,
                                             Optional<Location> loc,
                                             ValueRange operands,
                                             DictionaryAttr attrs,
                                             mlir::RegionRange regions,
                                             SmallVectorImpl<Type> &results) {
  auto inType = operands[0].getType();
  auto memType = inType.dyn_cast<CMemoryType>();
  if (!memType) {
    if (loc)
      mlir::emitError(*loc, "memory port requires memory operand");
    return failure();
  }
  results.push_back(memType.getElementType());
  results.push_back(CMemoryPortType::get(context));
  return success();
}

static LogicalResult verifyMemoryPortOp(MemoryPortOp memoryPort) {
  // MemoryPorts require exactly 1 access. Right now there are no other
  // operations that could be using that value due to the types.
  if (!memoryPort.port().hasOneUse())
    return memoryPort.emitOpError(
        "port should be used by a chirrtl.memoryport.access");
  return success();
}

MemoryPortAccessOp MemoryPortOp::getAccess() {
  auto uses = port().use_begin();
  if (uses == port().use_end())
    return {};
  return cast<MemoryPortAccessOp>(uses->getOwner());
}

void MemoryPortOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  StringRef base = name();
  if (base.empty())
    base = "memport";
  setNameFn(data(), (base + "_data").str());
  setNameFn(port(), (base + "_port").str());
}

static ParseResult parseMemoryPortOp(OpAsmParser &parser,
                                     NamedAttrList &resultAttrs) {
  // Add an empty annotation array if none were parsed.
  auto result = parser.parseOptionalAttrDict(resultAttrs);
  if (!resultAttrs.get("annotations"))
    resultAttrs.append("annotations", parser.getBuilder().getArrayAttr({}));
  return result;
}

/// Always elide "direction" and elide "annotations" if it exists or
/// if it is empty.
static void printMemoryPortOp(OpAsmPrinter &p, Operation *op,
                              DictionaryAttr attr) {
  // "direction" is always elided.
  SmallVector<StringRef> elides = {"direction"};
  // Annotations elided if empty.
  if (op->getAttrOfType<ArrayAttr>("annotations").empty())
    elides.push_back("annotations");
  p.printOptionalAttrDict(op->getAttrs(), elides);
}

//===----------------------------------------------------------------------===//
// CombMemOp
//===----------------------------------------------------------------------===//

static ParseResult parseCombMemOp(OpAsmParser &parser,
                                  NamedAttrList &resultAttrs) {
  return parseCHIRRTLOp(parser, resultAttrs);
}

static void printCombMemOp(OpAsmPrinter &p, Operation *op,
                           DictionaryAttr attr) {
  printCHIRRTLOp(p, op, attr);
}

void CombMemOp::build(OpBuilder &builder, OperationState &result,
                      FIRRTLType elementType, unsigned numElements,
                      StringRef name, ArrayAttr annotations,
                      StringAttr innerSym) {
  build(builder, result,
        CMemoryType::get(builder.getContext(), elementType, numElements), name,
        annotations, innerSym);
}

//===----------------------------------------------------------------------===//
// SeqMemOp
//===----------------------------------------------------------------------===//

static ParseResult parseSeqMemOp(OpAsmParser &parser,
                                 NamedAttrList &resultAttrs) {
  return parseCHIRRTLOp(parser, resultAttrs);
}

/// Always elide "ruw" and elide "annotations" if it exists or if it is empty.
static void printSeqMemOp(OpAsmPrinter &p, Operation *op, DictionaryAttr attr) {
  printCHIRRTLOp(p, op, attr, {"ruw"});
}

void SeqMemOp::build(OpBuilder &builder, OperationState &result,
                     FIRRTLType elementType, unsigned numElements, RUWAttr ruw,
                     StringRef name, ArrayAttr annotations,
                     StringAttr innerSym) {
  build(builder, result,
        CMemoryType::get(builder.getContext(), elementType, numElements), ruw,
        name, annotations, innerSym);
}

//===----------------------------------------------------------------------===//
// CHIRRTL Dialect
//===----------------------------------------------------------------------===//

// This is used to give custom SSA names which match the "name" attribute of the
// memory operation, which allows us to elide the name attribute.
namespace {
struct CHIRRTLOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const override {
    // Many CHIRRTL dialect operations have an optional 'name' attribute.  If
    // present, use it.
    if (op->getNumResults() == 1)
      if (auto nameAttr = op->getAttrOfType<StringAttr>("name"))
        setNameFn(op->getResult(0), nameAttr.getValue());
  }
};
} // namespace

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/FIRRTL/CHIRRTLTypes.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/FIRRTL/CHIRRTL.cpp.inc"

void CHIRRTLDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/FIRRTL/CHIRRTL.cpp.inc"
      >();

  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/FIRRTL/CHIRRTLTypes.cpp.inc"
      >();

  // Register interface implementations.
  addInterfaces<CHIRRTLOpAsmDialectInterface>();
}

#include "circt/Dialect/FIRRTL/CHIRRTLDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// CMemory Type
//===----------------------------------------------------------------------===//

void CMemoryType::print(AsmPrinter &printer) const {
  printer << "<";
  // Don't print element types with "!firrtl.".
  firrtl::printNestedType(getElementType(), printer);
  printer << ", " << getNumElements() << ">";
}

Type CMemoryType::parse(AsmParser &parser) {
  FIRRTLType elementType;
  unsigned numElements;
  if (parser.parseLess() || firrtl::parseNestedType(elementType, parser) ||
      parser.parseComma() || parser.parseInteger(numElements) ||
      parser.parseGreater())
    return {};
  return parser.getChecked<CMemoryType>(elementType, numElements);
}

LogicalResult CMemoryType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  FIRRTLType elementType,
                                  unsigned numElements) {
  if (!elementType.isPassive()) {
    return emitError() << "behavioral memory element type must be passive";
  }
  return success();
}
