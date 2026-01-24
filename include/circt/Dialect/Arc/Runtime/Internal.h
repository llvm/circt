// TODO: Header
#ifndef CIRCT_DIALECT_ARC_RUNTIME_INTERNAL_H
#define CIRCT_DIALECT_ARC_RUNTIME_INTERNAL_H

#include <cassert>
#include <iostream>

namespace circt::arc::runtime::impl {

[[noreturn]] inline static void fatalError(const char *message) {
  std::cerr << "[ArcRuntime] Internal Error: " << message << std::endl;
  assert(false && "ArcRuntime Internal Error");
  abort();
}

} // namespace circt::arc::runtime::impl

#endif // CIRCT_DIALECT_ARC_RUNTIME_INTERNAL_H
