//===- EmissionPrinter.cpp - EmissionPrinter implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the EmissionPrinter class.
//
//===----------------------------------------------------------------------===//

#include "EmissionPrinter.h"

using namespace circt;
using namespace circt::ExportSystemC;

static std::string pathToMacroName(StringRef path) {
  std::string macroname = path.upper();
  std::replace(macroname.begin(), macroname.end(), '.', '_');
  std::replace(macroname.begin(), macroname.end(), '/', '_');
  return macroname;
}

EmissionPrinter::EmissionPrinter(raw_ostream &os, EmissionConfig config)
    : config(config), os(os) {}

void EmissionPrinter::emitFileHeader(StringRef filename) {
  os << "// " << filename << "\n";
  std::string macroname = pathToMacroName(filename);
  os << "#ifndef " << macroname << "\n";
  os << "#define " << macroname << "\n\n";
  os << "#include <systemc.h>\n\n";
}

void EmissionPrinter::emitFileFooter(StringRef filename) {
  std::string macroname = pathToMacroName(filename);
  os << "\n#endif // " << macroname << "\n\n";
}

LogicalResult EmissionPrinter::emitOp(Operation *op) {
  for (size_t i = 0; i < patterns.size(); ++i) {
    if (patterns[i]->match(op, config))
      return patterns[i]->emitStatement(op, config, *this);
  }
  return mlir::emitError(op->getLoc(), "no emission pattern found for ")
         << op->getName();
}

EmissionResult EmissionPrinter::getExpression(Value value) {
  Operation *op = value.getDefiningOp();
  if (op == nullptr)
    op = value.cast<BlockArgument>().getParentRegion()->getParentOp();

  for (size_t i = 0; i < patterns.size(); ++i) {
    if (patterns[i]->match(op, config)) {
      return patterns[i]->getExpression(value, config, *this);
    }
  }
  return EmissionResult();
}

LogicalResult EmissionPrinter::emitRegion(Region &region, bool increaseIndent) {
  assert(region.hasOneBlock() &&
         "only regions with exactly one block are supported for now");

  if (increaseIndent)
    indentLevel++;

  for (Operation &op : region.getBlocks().front()) {
    if (failed(emitOp(&op)))
      return failure();
  }

  if (increaseIndent)
    indentLevel--;

  return success();
}

void EmissionPrinter::emitIndent() {
  const StringRef indent = "  ";
  for (uint32_t i = 0; i < indentLevel; ++i) {
    os << indent;
  }
}
