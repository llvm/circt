#include "Evaluator.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include <pybind11/embed.h>

using namespace mlir;
using namespace circt;

void rtg::evaluate(ModuleOp moduleOp, const std::string &filename) {
  pybind11::scoped_interpreter python;
  auto rtgentry = pybind11::module::import("rtgentry");
  MlirModule mod;
  mod.ptr = moduleOp.getAsOpaquePointer();
  pybind11::object m = pybind11::cast(mod);
  rtgentry.attr("codegen")(m, filename);
}

mlir::OwningOpRef<ModuleOp> rtg::evaluate(const std::string &packageName,
                                          const std::string &filename) {
  pybind11::scoped_interpreter python;
  auto rtgentry = pybind11::module::import("rtgentry");
  pybind11::object obj = rtgentry.attr("codegen2")(packageName, filename);
  MlirModule mod = pybind11::cast<MlirModule>(obj);
  auto m = ModuleOp::getFromOpaquePointer(mod.ptr);
  llvm::errs() << "ID from python: "
               << m->getName().getTypeID().getAsOpaquePointer() << "\n";
  auto otherID = TypeID::get<ModuleOp>();
  llvm::errs() << "ID from CPP: " << otherID.getAsOpaquePointer() << "\n";
  mlir::OpBuilder builder(m);
  return cast<ModuleOp>(builder.clone(*m.getOperation()));
}
