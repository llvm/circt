#ifndef CIRCT_CONVERSION_EXPORTYOSYS_RTLILCONVERTERINTERNAL
#define CIRCT_CONVERSION_EXPORTYOSYS_RTLILCONVERTERINTERNAL

#include "mlir/Support/LogicalResult.h"
#include <string>

namespace Yosys::RTLIL {
class Design;
}

namespace mlir {
class ModuleOp;
}

namespace circt {
namespace rtlil {
mlir::LogicalResult importRTLILDesign(Yosys::RTLIL::Design *design, mlir::ModuleOp module);
std::string getEscapedName(llvm::StringRef name);
}
} // namespace circt

#endif