#ifndef CIRCT_CONVERSION_EXPORTYOSYS_RTLILCONVERTERINTERNAL
#define CIRCT_CONVERSION_EXPORTYOSYS_RTLILCONVERTERINTERNAL

#include "circt/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include <string>

namespace Yosys::RTLIL {
class Design;
class Module;
} // namespace Yosys::RTLIL

namespace mlir {
class ModuleOp;
}

namespace circt {
namespace hw {
class HWModuleLike;
class InstanceGraph;
} // namespace hw
namespace rtlil {

mlir::LogicalResult importRTLILDesign(Yosys::RTLIL::Design *design,
                                      mlir::ModuleOp module);
std::string getEscapedName(llvm::StringRef name);
void init_yosys(bool redirectLog = true);
} // namespace rtlil
} // namespace circt

#endif