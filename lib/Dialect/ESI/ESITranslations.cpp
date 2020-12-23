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
#include "llvm/ADT/TypeSwitch.h"

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

  llvm::raw_ostream &emitId(llvm::hash_code id) {
    return os << "@" << llvm::format_hex(id, /*width=*/16 + 2);
  }

  /// Emit the whole schema.
  LogicalResult emit();

  /// Collect the types for which we need to emit a schema.
  LogicalResult collectTypes(CosimEndpoint);

  /// Do we support emitting a schema for 'type'?
  static bool isTypeSupported(Type type);

  /// Emit a schema for a single int.
  LogicalResult emitSchemaFor(IntegerType type);

  /// Compute a hash which doesn't change from run-to-run for a given type.
  static llvm::hash_code getDeterministicHash(Type t);

private:
  /// Intentation utils.
  raw_ostream &indent() { return os.indent(currentIndent); }
  void addIndent() { currentIndent += 2; }
  void reduceIndent() { currentIndent -= 2; }
  size_t currentIndent = 0;

  ModuleOp module;
  llvm::raw_ostream &os;
  DiagnosticEngine &diag;
  size_t errorCount = 0;

  llvm::hash_code fileHash = 0;
  llvm::DenseSet<Type> types;
};
} // anonymous namespace

bool ExportCosimSchema::isTypeSupported(Type type) {
  auto i = type.dyn_cast<IntegerType>();
  if (i)
    return i.getWidth() <= 64;
  return false;
}

LogicalResult ExportCosimSchema::emitSchemaFor(IntegerType type) {
  SmallString<16> typeStr;
  if (type.isSigned())
    typeStr = "Int";
  else
    typeStr = "UInt";

  auto w = type.getWidth();
  if (w == 1)
    typeStr = "Bool";
  else if (w <= 8)
    typeStr += "8";
  else if (w <= 16)
    typeStr += "16";
  else if (w <= 32)
    typeStr += "32";
  else if (w <= 64)
    typeStr += "64";
  else
    return diag.emit(nullptr, DiagnosticSeverity::Error)
           << "Capnp does not support ints larger than 64-bit";

  indent() << "struct ESI" << type << " ";
  emitId(getDeterministicHash(type)) << " {\n";
  addIndent();

  indent() << "// Actual type is " << type << ".\n";
  indent() << "i @0 :" << typeStr << ";\n";

  reduceIndent();
  indent() << "}\n\n";
  return success();
}

llvm::hash_code ExportCosimSchema::getDeterministicHash(Type t) {
  // This is temporary until I figure a way to access a deterministic hash.
  // TODO: replace me!
  IntegerType i = t.dyn_cast<IntegerType>();
  if (i)
    return llvm::hash_combine(i.getWidth(), i.getSignedness());
  return 0;
}

LogicalResult ExportCosimSchema::collectTypes(CosimEndpoint ep) {
  ChannelPort inputPort = ep.input().getType().dyn_cast<ChannelPort>();
  if (!inputPort)
    return ep.emitOpError("Expected ChannelPort type for input. Got ")
           << inputPort;
  Type inputTy = inputPort.getInner();
  if (!isTypeSupported(inputTy))
    return ep.emitOpError("Type '") << inputTy << "' not supported.";
  types.insert(inputTy);

  ChannelPort outputPort = ep.output().getType().dyn_cast<ChannelPort>();
  if (!outputPort)
    ep.emitOpError("Expected ChannelPort type for output. Got ") << outputPort;
  Type outputTy = outputPort.getInner();
  if (!isTypeSupported(outputTy))
    return ep.emitOpError("Type '") << outputTy << "' not supported.";
  types.insert(outputTy);

  fileHash = llvm::hash_combine(getDeterministicHash(inputTy),
                                getDeterministicHash(outputTy), fileHash);
  return success();
}

LogicalResult ExportCosimSchema::emit() {
  module.walk([this](CosimEndpoint ep) { collectTypes(ep); });

  os << "## ESI generated schema\n";

  // Emit the capnp file id.
  emitId(fileHash) << ";\n\n";

  for (auto type : types) {
    LogicalResult rc =
        TypeSwitch<Type, LogicalResult>(type)
            .Case([this](IntegerType it) { return emitSchemaFor(it); })
            .Default([this](Type type) {
              return diag.emit(nullptr, DiagnosticSeverity::Error)
                     << "Type '" << type << "' not supported.";
            });
    if (failed(rc))
      // If we fail during an emission, dump out early since the output may be
      // corrupted.
      return failure();
  }

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
