// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef __ESI_DIALECT_HPP__
#define __ESI_DIALECT_HPP__

#include <mlir/IR/Dialect.h>

namespace mlir {
namespace esi {

class ESIDialect : public ::mlir::Dialect {
public:
  explicit ESIDialect(MLIRContext *context);

  /// Returns the prefix used in the textual IR to refer to LLHD operations
  static StringRef getDialectNamespace() { return "esi"; }

  /// Parses a type registered to this dialect
  Type parseType(DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect
  void printType(Type type, DialectAsmPrinter &printer) const override;

  // /// Parse an attribute regustered to this dialect
  // Attribute parseAttribute(DialectAsmParser &parser, Type type) const
  // override;

  // /// Print an attribute registered to this dialect
  // void printAttribute(Attribute attr,
  //                     DialectAsmPrinter &printer) const override;

  // Operation *materializeConstant(OpBuilder &builder, Attribute value, Type
  // type,
  //                                 Location loc);
};

} // namespace esi
} // namespace mlir

#endif
