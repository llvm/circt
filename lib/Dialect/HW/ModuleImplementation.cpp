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
#include "circt/Support/ParsingUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"

using namespace circt;
using namespace circt::hw;

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

/// Return the port name for the specified argument or result.
static StringRef getModuleArgumentName(Operation *module, size_t argNo) {
  if (auto mod = dyn_cast<HWModuleLike>(module)) {
    if (argNo < mod.getNumInputPorts())
      return mod.getInputName(argNo);
    return StringRef();
  }
  auto argNames = module->getAttrOfType<ArrayAttr>("argNames");
  // Tolerate malformed IR here to enable debug printing etc.
  if (argNames && argNo < argNames.size())
    return argNames[argNo].cast<StringAttr>().getValue();
  return StringRef();
}

static StringRef getModuleResultName(Operation *module, size_t resultNo) {
  if (auto mod = dyn_cast<HWModuleLike>(module)) {
    if (resultNo < mod.getNumOutputPorts())
      return mod.getOutputName(resultNo);
    return StringRef();
  }
  auto resultNames = module->getAttrOfType<ArrayAttr>("resultNames");
  // Tolerate malformed IR here to enable debug printing etc.
  if (resultNames && resultNo < resultNames.size())
    return resultNames[resultNo].cast<StringAttr>().getValue();
  return StringRef();
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

  // Handle either old FunctionOpInterface modules or new-style hwmodulelike
  // This whole thing should be split up into two functions, but the delta is
  // so small, we are leaving this for now.
  auto modOp = dyn_cast<hw::HWModuleLike>(op);
  auto funcOp = dyn_cast<mlir::FunctionOpInterface>(op);
  SmallVector<Attribute> inputAttrs, outputAttrs;
  if (funcOp) {
    if (auto args = funcOp.getAllArgAttrs())
      for (auto a : args.getValue())
        inputAttrs.push_back(a);
    inputAttrs.resize(funcOp.getNumArguments());
    if (auto results = funcOp.getAllResultAttrs())
      for (auto a : results.getValue())
        outputAttrs.push_back(a);
    outputAttrs.resize(funcOp.getNumResults());
  } else {
    inputAttrs = modOp.getAllInputAttrs();
    outputAttrs = modOp.getAllOutputAttrs();
  }

  p << '(';
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    auto argName = modOp ? modOp.getInputName(i) : getModuleArgumentName(op, i);
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
    auto inputAttr = inputAttrs[i];
    p.printOptionalAttrDict(inputAttr
                                ? cast<DictionaryAttr>(inputAttr).getValue()
                                : ArrayRef<NamedAttribute>());

    // TODO: `printOptionalLocationSpecifier` will emit aliases for locations,
    // even if they are not printed.  This will have to be fixed upstream.  For
    // now, use what was specified on the command line.
    if (flags.shouldPrintDebugInfo()) {
      auto loc = modOp.getInputLoc(i);
      if (!isa<UnknownLoc>(loc))
        p.printOptionalLocationSpecifier(loc);
    }
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
      p.printKeywordOrString(getModuleResultName(op, i));
      p << ": ";
      p.printType(resultTypes[i]);
      auto outputAttr = outputAttrs[i];
      p.printOptionalAttrDict(outputAttr
                                  ? cast<DictionaryAttr>(outputAttr).getValue()
                                  : ArrayRef<NamedAttribute>());

      // TODO: `printOptionalLocationSpecifier` will emit aliases for locations,
      // even if they are not printed.  This will have to be fixed upstream. For
      // now, use what was specified on the command line.
      if (flags.shouldPrintDebugInfo()) {
        auto loc = modOp.getOutputLoc(i);
        if (!isa<UnknownLoc>(loc))
          p.printOptionalLocationSpecifier(loc);
      }
    }
    p << ')';
  }
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
                               /*allowType=*/true, /*allowAttrs=*/true))
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
    argNames.push_back(parsing_util::getNameFromSSA(context, arg.ssaName.name));
    argTypes.push_back(arg.type);
    if (!arg.sourceLoc)
      arg.sourceLoc = parser.getEncodedSourceLoc(arg.ssaName.location);
    argLocs.push_back(*arg.sourceLoc);
  }

  type = TypeAttr::get(FunctionType::get(context, argTypes, resultTypes));

  return success();
}

////////////////////////////////////////////////////////////////////////////////
// New Style
////////////////////////////////////////////////////////////////////////////////

/// Parse an optional keyword or string and set instance into 'result'.`
/// Returns failure on a parse issue, but not on not finding the string. 'found'
/// indicates whether the optional value exists.
ParseResult parseOptionalKeywordOrOptionalString(OpAsmParser &p,
                                                 std::string &result,
                                                 bool &found) {
  StringRef keyword;
  if (succeeded(p.parseOptionalKeyword(&keyword))) {
    result = keyword.str();
    found = true;
    return success();
  }

  if (succeeded(p.parseOptionalString(&result)))
    found = true;
  return success();
}

static ParseResult parseDirection(OpAsmParser &p, ModulePort::Direction &dir) {
  StringRef key;
  if (failed(p.parseKeyword(&key)))
    return p.emitError(p.getCurrentLocation(), "expected port direction");
  if (key == "in")
    dir = ModulePort::Direction::Input;
  else if (key == "out")
    dir = ModulePort::Direction::Output;
  else if (key == "inout")
    dir = ModulePort::Direction::InOut;
  else
    return p.emitError(p.getCurrentLocation(), "unknown port direction '")
           << key << "'";
  return success();
}

static ParseResult parseInputPort(OpAsmParser &parser,
                                  module_like_impl::PortParse &result) {
  if (parser.parseOperand(result.ssaName, /*allowResultNumber=*/false))
    return failure();
  NamedAttrList attrs;

  // Parse the result name.
  bool found = false;
  if (parseOptionalKeywordOrOptionalString(parser, result.rawName, found))
    return failure();

  // If there is only a ssa name, use it as the port name.  The ssa name is
  // always required, but if there is the optional arbitrary name, it is used as
  // the port name and the ssa name is just used for parsing the module.
  if (!found)
    result.rawName =
        parsing_util::getNameFromSSA(parser.getContext(), result.ssaName.name)
            .str();

  if (parser.parseColonType(result.type) ||
      parser.parseOptionalAttrDict(attrs) ||
      parser.parseOptionalLocationSpecifier(result.sourceLoc))
    return failure();
  result.attrs = attrs.getDictionary(parser.getContext());
  return success();
}

static ParseResult parseOutputPort(OpAsmParser &parser,
                                   module_like_impl::PortParse &result) {
  // Stash the current location parser location.
  auto irLoc = parser.getCurrentLocation();

  // Parse the result name.
  if (parser.parseKeywordOrString(&result.rawName))
    return failure();

  // Parse the results type.
  if (parser.parseColonType(result.type))
    return failure();

  // Parse the result attributes.
  NamedAttrList attrs;
  if (failed(parser.parseOptionalAttrDict(attrs)))
    return failure();
  result.attrs = attrs.getDictionary(parser.getContext());

  // Parse the result location.
  std::optional<Location> maybeLoc;
  if (failed(parser.parseOptionalLocationSpecifier(maybeLoc)))
    return failure();
  result.sourceLoc = maybeLoc ? *maybeLoc : parser.getEncodedSourceLoc(irLoc);

  return success();
}

/// Parse a single argument with the following syntax:
///
///   output (id|string) : !type { optionalAttrDict} loc(optionalSourceLoc)`
///   (input|inout) %ssaname : !type { optionalAttrDict} loc(optionalSourceLoc)`
///
static ParseResult parsePort(OpAsmParser &p,
                             module_like_impl::PortParse &result) {
  NamedAttrList attrs;
  if (parseDirection(p, result.direction))
    return failure();
  if (result.direction == ModulePort::Direction::Output)
    return parseOutputPort(p, result);
  return parseInputPort(p, result);
}

static ParseResult
parsePortList(OpAsmParser &p,
              SmallVectorImpl<module_like_impl::PortParse> &result) {
  auto parseOnePort = [&]() -> ParseResult {
    return parsePort(p, result.emplace_back());
  };
  return p.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseOnePort,
                                   " in port list");
}

ParseResult module_like_impl::parseModuleSignature(
    OpAsmParser &parser, SmallVectorImpl<PortParse> &args, TypeAttr &modType) {

  auto *context = parser.getContext();

  // Parse the port list.
  if (parsePortList(parser, args))
    return failure();

  // Process the ssa args for the information we're looking for.
  SmallVector<ModulePort> ports;
  for (auto &arg : args) {
    ports.push_back(
        {StringAttr::get(context, arg.rawName), arg.type, arg.direction});
    // rewrite type AFTER constructing ports.  This will be used in block args.
    if (arg.direction == ModulePort::InOut)
      arg.type = InOutType::get(arg.type);
    if (!arg.sourceLoc)
      arg.sourceLoc = parser.getEncodedSourceLoc(arg.ssaName.location);
  }
  modType = TypeAttr::get(ModuleType::get(context, ports));

  return success();
}

static const char *directionAsString(ModulePort::Direction dir) {
  if (dir == ModulePort::Direction::Input)
    return "in";
  if (dir == ModulePort::Direction::Output)
    return "out";
  if (dir == ModulePort::Direction::InOut)
    return "inout";
  assert(0 && "Unknown port direction");
  abort();
  return "unknown";
}

void module_like_impl::printModuleSignatureNew(OpAsmPrinter &p, Operation *op) {

  Region &body = op->getRegion(0);
  bool isExternal = body.empty();
  SmallString<32> resultNameStr;
  mlir::OpPrintingFlags flags;
  unsigned curArg = 0;

  auto typeAttr = op->getAttrOfType<TypeAttr>("module_type");
  auto modType = cast<ModuleType>(typeAttr.getValue());
  auto portAttrs = op->getAttrOfType<ArrayAttr>("per_port_attrs");
  auto locAttrs = op->getAttrOfType<ArrayAttr>("port_locs");

  p << '(';
  for (auto [i, port] : llvm::enumerate(modType.getPorts())) {
    if (i > 0)
      p << ", ";
    p.printKeywordOrString(directionAsString(port.dir));
    if (port.dir == ModulePort::Direction::Output) {
      p << " ";
      p.printKeywordOrString(port.name);
    } else {
      if (!isExternal) {
        // Get the printed format for the argument name.
        resultNameStr.clear();
        llvm::raw_svector_ostream tmpStream(resultNameStr);
        p.printOperand(body.front().getArgument(curArg), tmpStream);
        p << " " << tmpStream.str();
        // If the name wasn't printable in a way that agreed with argName, make
        // sure to print out an explicit argNames attribute.
        if (tmpStream.str().drop_front() != port.name) {
          p << " ";
          p.printKeywordOrString(port.name);
        }
      } else {
        p << " %" << port.name.getValue();
      }
      ++curArg;
    }
    p << " : ";
    p.printType(port.type);
    if (portAttrs && !portAttrs.empty())
      if (auto attr = dyn_cast<DictionaryAttr>(portAttrs[i]))
        p.printOptionalAttrDict(attr.getValue());

    // TODO: `printOptionalLocationSpecifier` will emit aliases for locations,
    // even if they are not printed.  This will have to be fixed upstream.  For
    // now, use what was specified on the command line.
    if (flags.shouldPrintDebugInfo() && locAttrs)
      if (auto loc = locAttrs[i])
        if (!isa<UnknownLoc>(loc))
          p.printOptionalLocationSpecifier(cast<Location>(loc));
  }

  p << ')';
}
