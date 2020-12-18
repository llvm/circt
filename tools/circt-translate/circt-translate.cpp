//===- circt-translate.cpp - CIRCT Translation Driver ---------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "circt/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"

int main(int argc, char **argv) {
  circt::registerAllTranslations();
  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "CIRCT Translation Testing Tool"));
}
