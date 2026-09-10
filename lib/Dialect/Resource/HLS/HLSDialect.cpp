//===- HLSDialect.cpp - HLS dialect implementation ------------------------===//

#include "circt/Dialect/Resource/HLS/HLSOps.h"
#include "circt/Dialect/Resource/HLS/HLSDialect.cpp.inc"

#include <circt/Conversion/Passes.h.inc>

using namespace circt;
using namespace circt::hls_analysis;

void HLSDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Resource/HLS/HLS.cpp.inc"
      >();
  // When you add attributes/types later, register them here too.
}
