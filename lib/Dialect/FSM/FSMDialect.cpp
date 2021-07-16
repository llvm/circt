//===- FSMDialect.cpp - Implement the FSM dialect -------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FSM/FSMDialect.h"
#include "circt/Dialect/FSM/FSMOps.h"

using namespace circt;
using namespace fsm;

void FSMDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/FSM/FSM.cpp.inc"
      >();
}
