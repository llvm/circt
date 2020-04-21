//===- FIRParser.h - .fir to FIRRTL dialect parser --------------*- C++ -*-===//
//
// Defines the interface to the .fir file parser.
//
//===----------------------------------------------------------------------===//

#ifndef SPT_DIALECT_FIRPARSER_H
#define SPT_DIALECT_FIRPARSER_H

namespace llvm {
class SourceMgr;
}

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace spt {

mlir::OwningModuleRef parseFIRFile(llvm::SourceMgr &sourceMgr,
                                   mlir::MLIRContext *context);
void registerFIRParserTranslation();

} // namespace spt

#endif // SPT_DIALECT_FIRPARSER_H
