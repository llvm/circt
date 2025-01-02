//===- VCD.cpp - VCD parser/printer implementation -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/VCD.h"

using namespace circt;
using namespace vcd;

// VCDToken implementation
VCDToken::VCDToken(Kind kind, StringRef spelling) 
  : kind(kind), spelling(spelling) {}

StringRef VCDToken::getSpelling() const { return spelling; }
VCDToken::Kind VCDToken::getKind() const { return kind; }
bool VCDToken::is(Kind K) const { return kind == K; }
bool VCDToken::isCommand() const { return spelling.starts_with("$"); }

llvm::SMLoc VCDToken::getLoc() const { 
  return llvm::SMLoc::getFromPointer(spelling.data()); 
}

llvm::SMLoc VCDToken::getEndLoc() const {
  return llvm::SMLoc::getFromPointer(spelling.data() + spelling.size());
}

llvm::SMRange VCDToken::getLocRange() const { 
  return llvm::SMRange(getLoc(), getEndLoc()); 
}

static StringAttr getMainBufferNameIdentifier(const llvm::SourceMgr &sourceMgr,
                                              MLIRContext *context) {
  auto mainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "<unknown>";
  return StringAttr::get(context, bufferName);
}

// VCDLexer implementation 
VCDLexer::VCDLexer(const llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context)
    : sourceMgr(sourceMgr),
      bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)),
      curBuffer(sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer()),
      curPtr(curBuffer.begin()),
      curToken(lexTokenImpl()) {}

// VCDParser implementation
VCDParser::VCDParser(mlir::MLIRContext *context, VCDLexer &lexer)
    : context(context), lexer(lexer) {}

// SignalMapping implementation
SignalMapping::SignalMapping(mlir::ModuleOp moduleOp, const VCDFile &file,
                            igraph::InstanceGraph &instanceGraph,
                            igraph::InstancePathCache &instancePathCache,
                            ArrayRef<StringAttr> pathToTop, 
                            VCDFile::Scope *topScope,
                            FlatSymbolRefAttr topModuleName)
    : moduleOp(moduleOp), topScope(topScope), topModuleName(topModuleName),
      file(file), instanceGraph(instanceGraph),
      instancePathCache(instancePathCache) {
}

std::unique_ptr<VCDFile> importVCDFile(llvm::SourceMgr &sourceMgr,
                                       MLIRContext *context) {
  VCDLexer lexer(sourceMgr, context);
  VCDParser parser(context, lexer);
  std::unique_ptr<VCDFile> file;
  if (parser.parseVCDFile(file))
    return nullptr;
  return file;
}
