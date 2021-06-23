#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

using namespace circt;
using namespace hir;
using namespace llvm;

// parser and printer for Time and offset.
ParseResult
parseTimeAndOffset(mlir::OpAsmParser &parser, OpAsmParser::OperandType &tstart,
                   llvm::Optional<OpAsmParser::OperandType> &varOffset,
                   IntegerAttr &constOffset);
void printTimeAndOffset(OpAsmPrinter &printer, Operation *op, Value tstart,
                        Value varOffset, IntegerAttr constOffset);

// parser and printer for array address types.

ParseResult parseOptionalArrayAccessTypes(mlir::OpAsmParser &parser,
                                          ArrayAttr &constAddrs,
                                          SmallVectorImpl<Type> &varAddrTypes);
void printOptionalArrayAccessTypes(OpAsmPrinter &printer, Operation *op,
                                   ArrayAttr constAddrs,
                                   TypeRange varAddrTypes);

// parser and printer for array address indices.
ParseResult
parseOptionalArrayAccess(OpAsmParser &parser,
                         SmallVectorImpl<OpAsmParser::OperandType> &varAddrs,
                         ArrayAttr &constAddrs);
void printOptionalArrayAccess(OpAsmPrinter &printer, Operation *op,
                              OperandRange varAddrs, ArrayAttr constAddrs);

ParseResult parseMemrefAndElementType(OpAsmParser &, Type &,
                                      SmallVectorImpl<Type> &, Type &);

void printMemrefAndElementType(OpAsmPrinter &, Operation *, Type, TypeRange,
                               Type);
