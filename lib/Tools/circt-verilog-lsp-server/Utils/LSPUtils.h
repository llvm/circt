#ifndef CIRCT_SUPPORT_LSPUTILS_H
#define CIRCT_SUPPORT_LSPUTILS_H

#include "circt/Support/LLVM.h"
namespace circt {

namespace lsp {
namespace Logger {
// These are specialization of logger functions. Slang requires RTTI but
// usually LLVM is not built with RTTI. So it causes compilation error when
// these functions are used in `VerilogServerImpl`.
void error(Twine message);
void info(Twine message);
void debug(Twine message);
} // namespace Logger
} // namespace lsp
} // namespace circt
#endif // CIRCT_SUPPORT_LSPUTILS_H
