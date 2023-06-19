//===- ModuleImplementation.cpp - Utilities for module-like ops -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include <mlir/IR/BuiltinAttributes.h>

using namespace circt;
using namespace circt::hw;

/// Get the portname from an SSA value string, if said value name is not a
/// number
StringAttr module_like_impl::getPortNameAttr(MLIRContext *context,
                                             StringRef name) {
  if (!name.empty()) {
    // Ignore numeric names like %42
    assert(name.size() > 1 && name[0] == '%' && "Unknown MLIR name");
    if (isdigit(name[1]))
      name = StringRef();
    else
      name = name.drop_front();
  }
  return StringAttr::get(context, name);
}

/// Parse a function result list.
///
///   function-result-list ::= function-result-list-parens
///   function-result-list-parens ::= `(` `)`
///                                 | `(` function-result-list-no-parens `)`
///   function-result-list-no-parens ::= function-result (`,` function-result)*
///   function-result ::= (percent-identifier `:`) type attribute-dict?
///
static ParseResult
parseFunctionResultList(OpAsmParser &parser,
                        SmallVectorImpl<Attribute> &resultNames,
                        SmallVectorImpl<Type> &resultTypes,
                        SmallVectorImpl<DictionaryAttr> &resultAttrs,
                        SmallVectorImpl<Attribute> &resultLocs) {

  auto parseElt = [&]() -> ParseResult {
    // Stash the current location parser location.
    auto irLoc = parser.getCurrentLocation();

    // Parse the result name.
    std::string portName;
    if (parser.parseKeywordOrString(&portName))
      return failure();
    resultNames.push_back(StringAttr::get(parser.getContext(), portName));

    // Parse the results type.
    resultTypes.emplace_back();
    if (parser.parseColonType(resultTypes.back()))
      return failure();

    // Parse the result attributes.
    NamedAttrList attrs;
    if (failed(parser.parseOptionalAttrDict(attrs)))
      return failure();
    resultAttrs.push_back(attrs.getDictionary(parser.getContext()));

    // Parse the result location.
    std::optional<Location> maybeLoc;
    if (failed(parser.parseOptionalLocationSpecifier(maybeLoc)))
      return failure();
    Location loc = maybeLoc ? *maybeLoc : parser.getEncodedSourceLoc(irLoc);
    resultLocs.push_back(loc);

    return success();
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseElt);
}

ParseResult module_like_impl::parseModuleFunctionSignature(
    OpAsmParser &parser, bool &isVariadic,
    SmallVectorImpl<OpAsmParser::Argument> &args,
    SmallVectorImpl<Attribute> &argNames, SmallVectorImpl<Attribute> &argLocs,
    SmallVectorImpl<Attribute> &resultNames,
    SmallVectorImpl<DictionaryAttr> &resultAttrs,
    SmallVectorImpl<Attribute> &resultLocs, TypeAttr &type) {

  using namespace mlir::function_interface_impl;
  auto *context = parser.getContext();

  // Parse the argument list.
  if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowTypes=*/true, /*allowAttrs=*/true))
    return failure();

  // Parse the result list.
  SmallVector<Type> resultTypes;
  if (succeeded(parser.parseOptionalArrow()))
    if (failed(parseFunctionResultList(parser, resultNames, resultTypes,
                                       resultAttrs, resultLocs)))
      return failure();

  // Process the ssa args for the information we're looking for.
  SmallVector<Type> argTypes;
  for (auto &arg : args) {
    argNames.push_back(getPortNameAttr(context, arg.ssaName.name));
    argTypes.push_back(arg.type);
    if (!arg.sourceLoc)
      arg.sourceLoc = parser.getEncodedSourceLoc(arg.ssaName.location);
    argLocs.push_back(*arg.sourceLoc);
  }

  type = TypeAttr::get(FunctionType::get(context, argTypes, resultTypes));

  return success();
}

void module_like_impl::printModuleSignature(OpAsmPrinter &p, Operation *op,
                                            ArrayRef<Type> argTypes,
                                            bool isVariadic,
                                            ArrayRef<Type> resultTypes,
                                            bool &needArgNamesAttr) {
  using namespace mlir::function_interface_impl;

  Region &body = op->getRegion(0);
  bool isExternal = body.empty();
  SmallString<32> resultNameStr;
  mlir::OpPrintingFlags flags;

  auto funcOp = cast<mlir::FunctionOpInterface>(op);

  p << '(';
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    auto argName = getModuleArgumentName(op, i);

    if (!isExternal) {
      // Get the printed format for the argument name.
      resultNameStr.clear();
      llvm::raw_svector_ostream tmpStream(resultNameStr);
      p.printOperand(body.front().getArgument(i), tmpStream);

      // If the name wasn't printable in a way that agreed with argName, make
      // sure to print out an explicit argNames attribute.
      if (tmpStream.str().drop_front() != argName)
        needArgNamesAttr = true;

      p << tmpStream.str() << ": ";
    } else if (!argName.empty()) {
      p << '%' << argName << ": ";
    }

    p.printType(argTypes[i]);
    p.printOptionalAttrDict(getArgAttrs(funcOp, i));

    // TODO: `printOptionalLocationSpecifier` will emit aliases for locations,
    // even if they are not printed.  This will have to be fixed upstream.  For
    // now, use what was specified on the command line.
    if (flags.shouldPrintDebugInfo())
      if (auto loc = getModuleArgumentLocAttr(op, i))
        p.printOptionalLocationSpecifier(loc);
  }

  if (isVariadic) {
    if (!argTypes.empty())
      p << ", ";
    p << "...";
  }

  p << ')';

  // We print result types specially since we support named arguments.
  if (!resultTypes.empty()) {
    p << " -> (";
    for (size_t i = 0, e = resultTypes.size(); i < e; ++i) {
      if (i != 0)
        p << ", ";
      p.printKeywordOrString(getModuleResultNameAttr(op, i).getValue());
      p << ": ";
      p.printType(resultTypes[i]);
      p.printOptionalAttrDict(getResultAttrs(funcOp, i));

      // TODO: `printOptionalLocationSpecifier` will emit aliases for locations,
      // even if they are not printed.  This will have to be fixed upstream. For
      // now, use what was specified on the command line.
      if (flags.shouldPrintDebugInfo())
        if (auto loc = getModuleResultLocAttr(op, i))
          p.printOptionalLocationSpecifier(loc);
    }
    p << ')';
  }
}

static std::optional<DictionaryAttr> toIdxMap(MLIRContext *ctx,
                                              ArrayRef<Attribute> names) {
  DenseSet<StringAttr> seenNames;
  SmallVector<NamedAttribute> idxMap;
  for (auto name : llvm::enumerate(names)) {
    auto nameAttr = name.value().cast<StringAttr>();

    // If the port list has duplicate names, then we cannot create a valid
    // index mapping.
    if (!seenNames.insert(nameAttr).second)
      return std::nullopt;

    // Handle the special case of anonymous ports; these are ports named with
    // empty strings, which are not legal to be used as NamedAttribute keys.
    if (!nameAttr.getValue().empty()) {
      idxMap.push_back(NamedAttribute(
          nameAttr, IntegerAttr::get(IntegerType::get(ctx, 64), name.index())));
    }
  }

  return {DictionaryAttr::get(ctx, idxMap)};
}

LogicalResult module_like_impl::updateModuleIndexMappings(Operation *mod) {
  auto argNames = mod->getAttrOfType<ArrayAttr>("argNames");
  auto resultNames = mod->getAttrOfType<ArrayAttr>("resultNames");

  if (!argNames || !resultNames)
    return mod->emitOpError("missing argNames or resultNames attribute");

  auto argIdxMap = toIdxMap(mod->getContext(), argNames.getValue());
  if (argIdxMap.has_value())
    mod->setAttr("argIdxMap", *argIdxMap);
  else
    mod->removeAttr("argIdxMap");

  auto resultIdxMap = toIdxMap(mod->getContext(), resultNames.getValue());
  if (resultIdxMap.has_value())
    mod->setAttr("resultIdxMap", *resultIdxMap);
  else
    mod->removeAttr("resultIdxMap");

  return success();
}

void module_like_impl::updateModuleIndexMappings(
    OperationState &result, ArrayRef<Attribute> argNames,
    ArrayRef<Attribute> resultNames) {

  auto argIdxMap = toIdxMap(result.getContext(), argNames);
  if (argIdxMap.has_value())
    result.addAttribute("argIdxMap", *argIdxMap);

  auto resultIdxMap = toIdxMap(result.getContext(), resultNames);
  if (resultIdxMap.has_value())
    result.addAttribute("resultIdxMap", *resultIdxMap);
}

static LogicalResult verifyIdxMap(Location loc, DictionaryAttr idxMap,
                                  ArrayAttr names) {
  size_t nonAnonymousPorts = 0;
  for (auto name : names)
    if (!name.cast<StringAttr>().getValue().empty())
      ++nonAnonymousPorts;

  if (idxMap.size() != nonAnonymousPorts)
    return emitError(loc) << "incorrect number of entries in index map";

  // Validate that each port name is present in the index map and that each
  // index is unique.
  llvm::DenseSet<int64_t> seenIndices;
  llvm::DenseSet<StringAttr> seenNames;

  for (auto it : idxMap) {
    auto portName = it.getName();
    Attribute portIndex = it.getValue().dyn_cast<IntegerAttr>();
    if (!portIndex)
      return emitError(loc) << "expected integer attr for port " << portName
                            << " index in index map";

    if (!seenIndices.insert(portIndex.cast<IntegerAttr>().getInt()).second)
      return emitError(loc) << "duplicate index " << portIndex << " for port "
                            << portName << " in index map";

    if (!seenNames.insert(portName).second)
      return emitError(loc) << "duplicate port " << portName << " in index map";
  }

  return success();
}

LogicalResult module_like_impl::verifyModuleIdxMap(Operation *mod) {
  auto argNames = mod->getAttrOfType<ArrayAttr>("argNames");
  if (!argNames)
    return mod->emitOpError("missing argNames attribute");
  auto argIdxMap = mod->getAttrOfType<DictionaryAttr>("argIdxMap");
  if (argIdxMap && failed(verifyIdxMap(mod->getLoc(), argIdxMap, argNames)))

    return failure();

  auto resultNames = mod->getAttrOfType<ArrayAttr>("resultNames");
  if (!resultNames)
    return mod->emitOpError("missing resultNames attribute");
  auto resultIdxMap = mod->getAttrOfType<DictionaryAttr>("resultIdxMap");
  if (resultIdxMap &&
      failed(verifyIdxMap(mod->getLoc(), resultIdxMap, resultNames)))
    return failure();

  return success();
}

static FailureOr<size_t> lookupPortIndexFromMap(Location loc,
                                                DictionaryAttr idxMap,
                                                StringRef portName) {
  auto portIndex = idxMap.getNamed(portName);
  if (!portIndex)
    return emitError(loc) << "missing port " << portName << " in index map";
  return portIndex->getValue().cast<mlir::IntegerAttr>().getInt();
}

FailureOr<size_t> module_like_impl::getModuleArgIndex(Operation *op,
                                                      StringRef name) {
  auto argIdxMap = op->getAttrOfType<DictionaryAttr>("argIdxMap");
  if (!argIdxMap)
    return op->emitOpError("missing argIdxMap attribute - either the module "
                           "was modified without updating the index map, or "
                           "the module has duplicate port names, in which case "
                           "the index map is unavailable");
  return lookupPortIndexFromMap(op->getLoc(), argIdxMap, name);
}

FailureOr<size_t> module_like_impl::getModuleResIndex(Operation *op,
                                                      StringRef name) {
  auto resultIdxMap = op->getAttrOfType<DictionaryAttr>("resultIdxMap");
  if (!resultIdxMap)
    return op->emitOpError(
        "missing resultIdxMap attribute - either the module was modified "
        "without updating the index map, or the module has duplicate port "
        "names, in which case the index map is unavailable");
  return lookupPortIndexFromMap(op->getLoc(), resultIdxMap, name);
}
