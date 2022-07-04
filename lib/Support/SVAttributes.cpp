#include "circt/Support/SVAttributes.h"
#include "mlir/IR/Operation.h"

llvm::StringRef svAttr = "sv.attributes";

bool circt::hasSVAttributes(mlir::Operation *op) { return op->hasAttr(svAttr); }

mlir::Attribute circt::getSVAttributes(mlir::Operation *op) {
  return op->getAttr(svAttr);
}

void circt::setSVAttributes(mlir::Operation *op, mlir::Attribute attr) {
  return op->setAttr(svAttr, attr);
}
