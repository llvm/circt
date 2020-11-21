//===- ESIDialect.h - ESI dialect Dialect class -----------------*- C++ -*-===//
//
// The Elastic Silicon Interconnect (ESI) dialect
//
// ESI is a system interconnect generator. It is type safe and
// latency-insensitive. It can be used for on-chip, inter-chip, and host-chip
// communication. It is also intended to help with incremental adoption and
// integration with existing RTL as it provides a standardized, typed interface
// to the outside world.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESIDIALECT_H
#define CIRCT_DIALECT_ESI_ESIDIALECT_H

#include "mlir/IR/Dialect.h"

namespace circt {
namespace esi {

class ESIDialect : public ::mlir::Dialect {
public:
  explicit ESIDialect(mlir::MLIRContext *context);

  /// Returns the prefix used in the textual IR to refer to ESI operations
  static llvm::StringRef getDialectNamespace() { return "esi"; }

  /// Parses a type registered to this dialect
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
};

void registerESIPasses();

} // namespace esi
} // namespace circt

#include "circt/Dialect/ESI/ESIAttrs.h.inc"

#endif
