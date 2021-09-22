//===- HWAttributes.cpp - Implement HW attributes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

using namespace circt;
using namespace circt::hw;

//===----------------------------------------------------------------------===//
// ODS Boilerplate
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/HW/HWAttributes.cpp.inc"

void HWDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/HW/HWAttributes.cpp.inc"
      >();
}

Attribute HWDialect::parseAttribute(DialectAsmParser &p, Type type) const {
  StringRef attrName;
  Attribute attr;
  if (p.parseKeyword(&attrName))
    return Attribute();
  auto parseResult =
      generatedAttributeParser(getContext(), p, attrName, type, attr);
  if (parseResult.hasValue())
    return attr;
  p.emitError(p.getNameLoc(), "Unexpected hw attribute '" + attrName + "'");
  return {};
}

void HWDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  if (succeeded(generatedAttributePrinter(attr, p)))
    return;
  llvm_unreachable("Unexpected attribute");
}

//===----------------------------------------------------------------------===//
// OutputFileAttr
//===----------------------------------------------------------------------===//

static std::string canonicalizeFilename(const Twine &directory,
                                        const Twine &filename) {
  SmallString<128> fullPath;

  // If the filename is an absolute path, we don't need the directory.
  if (llvm::sys::path::is_absolute(filename))
    filename.toVector(fullPath);
  else
    llvm::sys::path::append(fullPath, directory, filename);

  // If this is a directory target, we need to ensure it ends with a `/`
  if (filename.isTriviallyEmpty() && !fullPath.endswith("/"))
    fullPath += "/";

  return std::string(fullPath);
}

OutputFileAttr OutputFileAttr::getFromFilename(MLIRContext *context,
                                               const Twine &filename,
                                               bool excludeFromFileList,
                                               bool includeReplicatedOps) {
  return OutputFileAttr::getFromDirectoryAndFilename(
      context, "", filename, excludeFromFileList, includeReplicatedOps);
}

OutputFileAttr OutputFileAttr::getFromDirectoryAndFilename(
    MLIRContext *context, const Twine &directory, const Twine &filename,
    bool excludeFromFileList, bool includeReplicatedOps) {
  auto canonicalized = canonicalizeFilename(directory, filename);
  return OutputFileAttr::get(StringAttr::get(context, canonicalized),
                             BoolAttr::get(context, excludeFromFileList),
                             BoolAttr::get(context, includeReplicatedOps));
}

OutputFileAttr OutputFileAttr::getAsDirectory(MLIRContext *context,
                                              const Twine &directory,
                                              bool excludeFromFileList,
                                              bool includeReplicatedOps) {
  return getFromDirectoryAndFilename(context, directory, "",
                                     excludeFromFileList, includeReplicatedOps);
}

bool OutputFileAttr::isDirectory() {
  return getFilename().getValue().endswith("/");
}

/// Option         ::= 'excludeFromFileList' | 'includeReplicatedOp'
/// OutputFileAttr ::= 'output_file<' directory ',' name (',' Option)* '>'
Attribute OutputFileAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                Type type) {
  StringAttr filename;
  if (p.parseLess() || p.parseAttribute<StringAttr>(filename))
    return Attribute();

  // Parse the additional keyword attributes.  Its easier to let people specify
  // these more than once than to detect the problem and do something about it.
  bool excludeFromFileList = false;
  bool includeReplicatedOps = false;
  while (true) {
    if (p.parseOptionalComma())
      break;
    if (!p.parseOptionalKeyword("excludeFromFileList"))
      excludeFromFileList = true;
    else if (!p.parseKeyword("includeReplicatedOps",
                             "or 'excludeFromFileList'"))
      includeReplicatedOps = true;
    else
      return Attribute();
  }

  if (p.parseGreater())
    return Attribute();

  return OutputFileAttr::get(context, filename,
                             BoolAttr::get(context, excludeFromFileList),
                             BoolAttr::get(context, includeReplicatedOps));
}

void OutputFileAttr::print(DialectAsmPrinter &p) const {
  p << "output_file<" << getFilename();
  if (getExcludeFromFilelist().getValue())
    p << ", excludeFromFileList";
  if (getIncludeReplicatedOps().getValue())
    p << ", includeReplicatedOps";
  p << ">";
}

//===----------------------------------------------------------------------===//
// ParamDeclAttr
//===----------------------------------------------------------------------===//

Attribute ParamDeclAttr::parse(MLIRContext *context, DialectAsmParser &p,
                               Type type) {
  llvm::errs() << "Should never parse raw\n";
  abort();
}

void ParamDeclAttr::print(DialectAsmPrinter &p) const {
  p << "param.decl<" << getName() << ": " << getType();
  if (getValue())
    p << " = " << getValue();
  p << ">";
}

//===----------------------------------------------------------------------===//
// ParamDeclRefAttr
//===----------------------------------------------------------------------===//

Attribute ParamDeclRefAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                  Type type) {
  StringAttr name;
  if (p.parseLess() || p.parseAttribute(name) || p.parseGreater())
    return Attribute();

  return ParamDeclRefAttr::get(context, name, type);
}

void ParamDeclRefAttr::print(DialectAsmPrinter &p) const {
  p << "param.decl.ref<" << getName() << ">";
}

//===----------------------------------------------------------------------===//
// ParamVerbatimAttr
//===----------------------------------------------------------------------===//

Attribute ParamVerbatimAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                   Type type) {
  StringAttr text;
  if (p.parseLess() || p.parseAttribute(text) || p.parseGreater())
    return Attribute();

  return ParamVerbatimAttr::get(context, text, type);
}

void ParamVerbatimAttr::print(DialectAsmPrinter &p) const {
  p << "param.verbatim<" << getValue() << ">";
}

//===----------------------------------------------------------------------===//
// ParamBinaryAttr
//===----------------------------------------------------------------------===//

Attribute ParamBinaryAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                 Type type) {
  Attribute lhs, rhs;
  StringRef opcodeStr;
  auto loc = p.getCurrentLocation();
  if (p.parseLess() || p.parseKeyword(&opcodeStr) ||
      p.parseAttribute(lhs, type) || p.parseComma() ||
      p.parseAttribute(rhs, type) || p.parseGreater())
    return Attribute();

  Optional<PBO> opcode = symbolizePBO(opcodeStr);
  if (!opcode.hasValue()) {
    p.emitError(loc, "unknown binary operator name");
    return {};
  }

  return ParamBinaryAttr::get(context, *opcode, lhs, rhs, type);
}

void ParamBinaryAttr::print(DialectAsmPrinter &p) const {
  p << "param.binary<" << stringifyPBO(getOpcode()) << " ";
  p.printAttributeWithoutType(getLhs());
  p << ", ";
  p.printAttributeWithoutType(getRhs());
  p << ">";
}
