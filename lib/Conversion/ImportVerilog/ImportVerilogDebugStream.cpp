#include "ImportVerilogDebugStream.h"
#include "ImportVerilogInternals.h"

ImportVerilogDebugStream
dbgs(const std::optional<mlir::Location> sourceLocation,
     const std::optional<std::source_location> cl) {

  return ImportVerilogDebugStream(sourceLocation, cl);
}

ImportVerilogDebugStream circt::ImportVerilog::Context::dbgs(
    const std::optional<std::variant<mlir::Location, slang::SourceLocation>>
        sourceLocation,
    const std::optional<std::source_location> cl) {

  std::optional<mlir::Location> mlirLoc = {};

  if (sourceLocation) {
    if (auto *ml = std::get_if<mlir::Location>(&*sourceLocation)) {
      mlirLoc = *ml;
    } else if (auto *sl =
                   std::get_if<slang::SourceLocation>(&*sourceLocation)) {
      mlirLoc = convertLocation(*sl);
    }
  }

  return ImportVerilogDebugStream(mlirLoc, cl);
}
