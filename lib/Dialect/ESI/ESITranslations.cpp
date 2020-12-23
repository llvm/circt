//===- ESITranslations.cpp - ESI translations -------------------*- C++ -*-===//
//
// ESI translations:
// - Cap'nProto schema generation
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"

#include "circt/Dialect/RTL/Dialect.h"
#include "circt/Dialect/SV/Dialect.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Translation.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace circt::esi;

//===----------------------------------------------------------------------===//
// ESI Cosim Cap'nProto schema generation.
//===----------------------------------------------------------------------===//

namespace {
struct ExportCosimSchema {
  ExportCosimSchema(ModuleOp module, llvm::raw_ostream &os)
      : module(module), os(os), diag(module.getContext()->getDiagEngine()) {
    diag.registerHandler([this](Diagnostic &diag) -> LogicalResult {
      if (diag.getSeverity() == DiagnosticSeverity::Error)
        ++errorCount;
      return failure();
    });
  }

  LogicalResult emit();

  /// Collect the types for which we need to emit a schema.
  void collectTypes(CosimEndpoint);

  /// Compute a hash which doesn't change from run-to-run for a given type.
  static llvm::hash_code getDeterministicHash(Type t);

private:
  ModuleOp module;
  llvm::raw_ostream &os;
  DiagnosticEngine &diag;
  size_t errorCount = 0;

  llvm::hash_code fileHash = 0;
  llvm::DenseSet<Type> types;
};
} // anonymous namespace

llvm::hash_code ExportCosimSchema::getDeterministicHash(Type t) {
  IntegerType i = t.dyn_cast<IntegerType>();
  if (i)
    return llvm::hash_combine(i.getWidth(), i.getSignedness());
  return 0;
}

void ExportCosimSchema::collectTypes(CosimEndpoint ep) {
  ChannelPort inputPort = ep.input().getType().dyn_cast<ChannelPort>();
  // if (!inputPort)
  ChannelPort outputPort = ep.output().getType().dyn_cast<ChannelPort>();
  types.insert(inputPort.getInner());
  types.insert(outputPort.getInner());
  fileHash =
      llvm::hash_combine(getDeterministicHash(inputPort.getInner()),
                         getDeterministicHash(outputPort.getInner()), fileHash);
}

LogicalResult ExportCosimSchema::emit() {
  module.walk([this](CosimEndpoint ep) { collectTypes(ep); });

  os << "## ESI generated schema\n";

  // Emit the capnp file id.
  os << "@" << llvm::format_hex(fileHash, /*width=*/16 + 2) << ";\n\n";

  // diag.emit(module.getLoc(), DiagnosticSeverity::Error)
  // << "Incomplete translation";
  return errorCount == 0 ? success() : failure();
}

static LogicalResult exportCosimSchema(ModuleOp module, llvm::raw_ostream &os) {
  ExportCosimSchema schema(module, os);
  return schema.emit();
}

//===----------------------------------------------------------------------===//
// Register all ESI translations.
//===----------------------------------------------------------------------===//

void circt::esi::registerESITranslations() {
  TranslateFromMLIRRegistration cosimToCapnp(
      "emit-esi-capnp", exportCosimSchema, [](DialectRegistry &registry) {
        registry
            .insert<ESIDialect, circt::rtl::RTLDialect, circt::sv::SVDialect,
                    mlir::StandardOpsDialect, mlir::BuiltinDialect>();
      });
}
