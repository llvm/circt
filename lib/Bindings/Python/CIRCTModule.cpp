//===- CIRCTModule.cpp - Main nanobind module -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIRCTModules.h"

#include "circt-c/Conversion.h"
#include "circt-c/Dialect/AIG.h"
#include "circt-c/Dialect/Arc.h"
#include "circt-c/Dialect/Comb.h"
#include "circt-c/Dialect/DC.h"
#include "circt-c/Dialect/Debug.h"
#include "circt-c/Dialect/ESI.h"
#include "circt-c/Dialect/Emit.h"
#include "circt-c/Dialect/FSM.h"
#include "circt-c/Dialect/HW.h"
#include "circt-c/Dialect/HWArith.h"
#include "circt-c/Dialect/Handshake.h"
#include "circt-c/Dialect/Kanagawa.h"
#include "circt-c/Dialect/LTL.h"
#include "circt-c/Dialect/MSFT.h"
#include "circt-c/Dialect/OM.h"
#include "circt-c/Dialect/Pipeline.h"
#include "circt-c/Dialect/RTG.h"
#include "circt-c/Transforms.h"
#ifdef CIRCT_INCLUDE_TESTS
#include "circt-c/Dialect/RTGTest.h"
#endif
#include "circt-c/Dialect/SV.h"
#include "circt-c/Dialect/Seq.h"
#include "circt-c/Dialect/Verif.h"
#include "circt-c/ExportVerilog.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Dialect/Index.h"
#include "mlir-c/Dialect/SCF.h"
#include "mlir-c/Dialect/SMT.h"
#include "mlir-c/IR.h"
#include "mlir-c/Transforms.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "llvm-c/ErrorHandling.h"
#include "llvm/Support/Signals.h"

#include "NanobindUtils.h"
#include <nanobind/nanobind.h>
namespace nb = nanobind;

static void registerPasses() {
  registerAIGPasses();
  registerArcPasses();
  registerCombPasses();
  registerDCPasses();
  registerSeqPasses();
  registerSVPasses();
  registerFSMPasses();
  registerHWArithPasses();
  registerHWPasses();
  registerRTGPasses();
  registerHandshakePasses();
  registerKanagawaPasses();
  registerPipelinePasses();
  mlirRegisterCIRCTConversionPasses();
  mlirRegisterCIRCTTransformsPasses();
  mlirRegisterTransformsCSE();
}

NB_MODULE(_circt, m) {
  m.doc() = "CIRCT Python Native Extension";
  registerPasses();
  llvm::sys::PrintStackTraceOnErrorSignal(/*argv=*/"");
  LLVMEnablePrettyStackTrace();

  m.def(
      "register_dialects",
      [](nb::object capsule) {
        // Get the MlirContext capsule from PyMlirContext capsule.
        auto wrappedCapsule = capsule.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
        MlirContext context = mlirPythonCapsuleToContext(wrappedCapsule.ptr());

        // Collect CIRCT dialects to register.
        MlirDialectHandle aig = mlirGetDialectHandle__aig__();
        mlirDialectHandleRegisterDialect(aig, context);
        mlirDialectHandleLoadDialect(aig, context);

        MlirDialectHandle comb = mlirGetDialectHandle__comb__();
        mlirDialectHandleRegisterDialect(comb, context);
        mlirDialectHandleLoadDialect(comb, context);

        MlirDialectHandle debug = mlirGetDialectHandle__debug__();
        mlirDialectHandleRegisterDialect(debug, context);
        mlirDialectHandleLoadDialect(debug, context);

        MlirDialectHandle emit = mlirGetDialectHandle__emit__();
        mlirDialectHandleRegisterDialect(emit, context);
        mlirDialectHandleLoadDialect(emit, context);

        MlirDialectHandle esi = mlirGetDialectHandle__esi__();
        mlirDialectHandleRegisterDialect(esi, context);
        mlirDialectHandleLoadDialect(esi, context);

        MlirDialectHandle msft = mlirGetDialectHandle__msft__();
        mlirDialectHandleRegisterDialect(msft, context);
        mlirDialectHandleLoadDialect(msft, context);

        MlirDialectHandle hw = mlirGetDialectHandle__hw__();
        mlirDialectHandleRegisterDialect(hw, context);
        mlirDialectHandleLoadDialect(hw, context);

        MlirDialectHandle hwarith = mlirGetDialectHandle__hwarith__();
        mlirDialectHandleRegisterDialect(hwarith, context);
        mlirDialectHandleLoadDialect(hwarith, context);

        MlirDialectHandle index = mlirGetDialectHandle__index__();
        mlirDialectHandleRegisterDialect(index, context);
        mlirDialectHandleLoadDialect(index, context);

        MlirDialectHandle scf = mlirGetDialectHandle__scf__();
        mlirDialectHandleRegisterDialect(scf, context);
        mlirDialectHandleLoadDialect(scf, context);

        MlirDialectHandle om = mlirGetDialectHandle__om__();
        mlirDialectHandleRegisterDialect(om, context);
        mlirDialectHandleLoadDialect(om, context);

        MlirDialectHandle pipeline = mlirGetDialectHandle__pipeline__();
        mlirDialectHandleRegisterDialect(pipeline, context);
        mlirDialectHandleLoadDialect(pipeline, context);

        MlirDialectHandle rtg = mlirGetDialectHandle__rtg__();
        mlirDialectHandleRegisterDialect(rtg, context);
        mlirDialectHandleLoadDialect(rtg, context);

#ifdef CIRCT_INCLUDE_TESTS
        MlirDialectHandle rtgtest = mlirGetDialectHandle__rtgtest__();
        mlirDialectHandleRegisterDialect(rtgtest, context);
        mlirDialectHandleLoadDialect(rtgtest, context);
#endif

        MlirDialectHandle seq = mlirGetDialectHandle__seq__();
        mlirDialectHandleRegisterDialect(seq, context);
        mlirDialectHandleLoadDialect(seq, context);

        MlirDialectHandle sv = mlirGetDialectHandle__sv__();
        mlirDialectHandleRegisterDialect(sv, context);
        mlirDialectHandleLoadDialect(sv, context);

        MlirDialectHandle fsm = mlirGetDialectHandle__fsm__();
        mlirDialectHandleRegisterDialect(fsm, context);
        mlirDialectHandleLoadDialect(fsm, context);

        MlirDialectHandle handshake = mlirGetDialectHandle__handshake__();
        mlirDialectHandleRegisterDialect(handshake, context);
        mlirDialectHandleLoadDialect(handshake, context);

        MlirDialectHandle kanagawa = mlirGetDialectHandle__kanagawa__();
        mlirDialectHandleRegisterDialect(kanagawa, context);
        mlirDialectHandleLoadDialect(kanagawa, context);

        MlirDialectHandle ltl = mlirGetDialectHandle__ltl__();
        mlirDialectHandleRegisterDialect(ltl, context);
        mlirDialectHandleLoadDialect(ltl, context);

        MlirDialectHandle verif = mlirGetDialectHandle__verif__();
        mlirDialectHandleRegisterDialect(verif, context);
        mlirDialectHandleLoadDialect(verif, context);

        MlirDialectHandle smt = mlirGetDialectHandle__smt__();
        mlirDialectHandleRegisterDialect(smt, context);
        mlirDialectHandleLoadDialect(smt, context);
      },
      "Register CIRCT dialects on a PyMlirContext.");

  m.def("export_verilog", [](MlirModule mod, nb::object fileObject) {
    circt::python::PyFileAccumulator accum(fileObject, false);
    nb::gil_scoped_release();
    mlirExportVerilog(mod, accum.getCallback(), accum.getUserData());
  });

  m.def("export_split_verilog", [](MlirModule mod, std::string directory) {
    auto cDirectory = mlirStringRefCreateFromCString(directory.c_str());
    mlirExportSplitVerilog(mod, cDirectory);
  });

  nb::module_ aig = m.def_submodule("_aig", "AIG API");
  circt::python::populateDialectAIGSubmodule(aig);
  nb::module_ esi = m.def_submodule("_esi", "ESI API");
  circt::python::populateDialectESISubmodule(esi);
  nb::module_ msft = m.def_submodule("_msft", "MSFT API");
  circt::python::populateDialectMSFTSubmodule(msft);
  nb::module_ hw = m.def_submodule("_hw", "HW API");
  circt::python::populateDialectHWSubmodule(hw);
  nb::module_ seq = m.def_submodule("_seq", "Seq API");
  circt::python::populateDialectSeqSubmodule(seq);
  nb::module_ om = m.def_submodule("_om", "OM API");
  circt::python::populateDialectOMSubmodule(om);
  nb::module_ rtg = m.def_submodule("_rtg", "RTG API");
  circt::python::populateDialectRTGSubmodule(rtg);
  nb::module_ rtgtool = m.def_submodule("_rtgtool", "RTGTool API");
  circt::python::populateDialectRTGToolSubmodule(rtgtool);
#ifdef CIRCT_INCLUDE_TESTS
  nb::module_ rtgtest = m.def_submodule("_rtgtest", "RTGTest API");
  circt::python::populateDialectRTGTestSubmodule(rtgtest);
#endif
  nb::module_ sv = m.def_submodule("_sv", "SV API");
  circt::python::populateDialectSVSubmodule(sv);
  nb::module_ support = m.def_submodule("_support", "CIRCT support");
  circt::python::populateSupportSubmodule(support);
}
