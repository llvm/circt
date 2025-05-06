//===-- Passes.td - Conversion pass definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for all dialect conversions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_PASSES_TD
#define CIRCT_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// AffineToPipeline
//===----------------------------------------------------------------------===//

def AffineToLoopSchedule : Pass<"convert-affine-to-loopschedule", "mlir::func::FuncOp"> {
  let summary = "Convert Affine dialect to LoopSchedule scheduled loops";
  let description = [{
    This pass analyzes Affine loops and control flow, creates a Scheduling
    problem using the Calyx operator library, solves the problem, and lowers
    the loops to a LoopSchedule.
  }];
  let constructor = "circt::createAffineToLoopSchedule()";
  let dependentDialects = [
    "circt::loopschedule::LoopScheduleDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ExportChiselInterface and ExportSplitChiselInterface
//===----------------------------------------------------------------------===//

def ExportChiselInterface : Pass<"export-chisel-interface", "firrtl::CircuitOp"> {
  let summary = "Emit a Chisel interface to a FIRRTL circuit";
  let description = [{
    This pass generates a Scala Chisel interface for the top level module of
    a FIRRTL circuit.
  }];

  let constructor = "createExportChiselInterfacePass()";
  let dependentDialects = [
    "circt::firrtl::FIRRTLDialect"
  ];
}

def ExportSplitChiselInterface : Pass<"export-split-chisel-interface", "firrtl::CircuitOp"> {
  let summary = "Emit a Chisel interface to a FIRRTL circuit to a directory of files";
  let description = [{
    This pass generates a Scala Chisel interface for the top level module of
    a FIRRTL circuit.
  }];

  let constructor = "createExportSplitChiselInterfacePass()";
  let dependentDialects = [
    "circt::firrtl::FIRRTLDialect"
  ];
  let options = [
    Option<"directoryName", "dir-name", "std::string",
            "", "Directory to emit into">
   ];
}

//===----------------------------------------------------------------------===//
// ExportVerilog and ExportSplitVerilog
//===----------------------------------------------------------------------===//

def TestApplyLoweringOption : Pass<"test-apply-lowering-options",
                                   "mlir::ModuleOp"> {
  let summary = "Apply lowering options";
  let description = [{
    This pass allows overriding lowering options.  It is intended for test
    construction.
  }];

  let constructor = "createTestApplyLoweringOptionPass()";
  let dependentDialects = [
    "circt::sv::SVDialect", "circt::comb::CombDialect", "circt::hw::HWDialect"
  ];
  let options = [
    Option<"optionsString", "options", "std::string", "", "Lowering Options">
  ];
}

def LegalizeAnonEnums : Pass<"legalize-anon-enums", "mlir::ModuleOp"> {
  let summary = "Prepare anonymous enumeration types for ExportVerilog";
  let description = [{
    This pass transforms all anonymous enumeration types into typedecls to work
    around difference in how anonymous enumerations work in SystemVerilog.
  }];
  let constructor = "createLegalizeAnonEnumsPass()";
  let dependentDialects = [
    "circt::sv::SVDialect", "circt::comb::CombDialect", "circt::hw::HWDialect"
  ];
}

def HWLowerInstanceChoices : Pass<"hw-lower-instance-choices",
                                  "mlir::ModuleOp"> {
  let summary = "Prepare the collateral for instance choice emission";
  let description = [{
    This pass runs as part of verilog emission.
    It introduces the macros & file lists to which instance choices lower to.
  }];

  let constructor = "createHWLowerInstanceChoicesPass()";
  let dependentDialects = [
    "circt::sv::SVDialect", "circt::hw::HWDialect"
  ];
}

def PrepareForEmission : Pass<"prepare-for-emission"> {
  let summary = "Prepare IR for ExportVerilog";
  let description = [{
    This pass runs only PrepareForEmission.
    It is not necessary for users to run this pass explicitly since
    ExportVerilog internally runs PrepareForEmission.
  }];

  let constructor = "createPrepareForEmissionPass()";
  let dependentDialects = [
    "circt::sv::SVDialect", "circt::comb::CombDialect", "circt::hw::HWDialect"
  ];
}

def ExportVerilog : Pass<"export-verilog", "mlir::ModuleOp"> {
  let summary = "Emit the IR to a (System)Verilog file";
  let description = [{
      This pass creates empty module bodies for external modules.  This is
      useful for linting to eliminate missing file errors.
  }];

  let constructor = "createExportVerilogPass()";
  let dependentDialects = [
    "circt::sv::SVDialect", "circt::comb::CombDialect", "circt::hw::HWDialect"
  ];
}

def ExportSplitVerilog : Pass<"export-split-verilog", "mlir::ModuleOp"> {
  let summary = "Emit the IR to a (System)Verilog directory of files";
  let description = [{
    This pass generates (System)Verilog for the current design, mutating it
    where necessary to be valid Verilog.
  }];

  let constructor = "createExportSplitVerilogPass()";
  let dependentDialects = [
    "circt::sv::SVDialect", "circt::comb::CombDialect", "circt::hw::HWDialect"
  ];

  let options = [
    Option<"directoryName", "dir-name", "std::string",
            "", "Directory to emit into">
   ];
}

//===----------------------------------------------------------------------===//
// SCFToCalyx
//===----------------------------------------------------------------------===//

def SCFToCalyx : Pass<"lower-scf-to-calyx", "mlir::ModuleOp"> {
  let summary = "Lower SCF/Standard to Calyx";
  let description = [{
    This pass lowers SCF / standard to Calyx.
  }];
  let constructor = "circt::createSCFToCalyxPass()";
  let dependentDialects = [
    "calyx::CalyxDialect", "::mlir::scf::SCFDialect", "hw::HWDialect",
    "comb::CombDialect"
  ];
  let options = [
    Option<"topLevelFunctionOpt", "top-level-function", "std::string", "",
            "Identifier of top-level function to be the entry-point component"
            " of the Calyx program.">,
    Option<"ciderSourceLocationMetadata", "cider-source-location-metadata", "bool", "",
            "Whether to track source location for the Cider debugger.">,
    Option<"writeJsonOpt", "write-json", "std::string", "",
            "Whether to write memory contents to the json file.">
  ];
}

//===----------------------------------------------------------------------===//
// LoopScheduleToCalyx
//===----------------------------------------------------------------------===//

def LoopScheduleToCalyx : Pass<"lower-loopschedule-to-calyx", "mlir::ModuleOp"> {
  let summary = "Lower LoopSchedule to Calyx";
  let description = [{
    This pass lowers LoopSchedule to Calyx.
  }];
  let constructor = "circt::createLoopScheduleToCalyxPass()";
  let dependentDialects = [
    "calyx::CalyxDialect", "loopschedule::LoopScheduleDialect", "hw::HWDialect",
    "comb::CombDialect"
  ];
  let options = [
    Option<"topLevelFunctionOpt", "top-level-function", "std::string", "",
            "Identifier of top-level function to be the entry-point component"
            " of the Calyx program.">,
    Option<"ciderSourceLocationMetadata", "cider-source-location-metadata", "bool", "",
            "Whether to track source location for the Cider debugger.">
  ];
}

//===----------------------------------------------------------------------===//
// PipelineToHW
//===----------------------------------------------------------------------===//

// TODO: @mortbopet: There is a possible non-neglible speedup that can be achieved
// here by allowing this pass to run on a per-hwmodule/whatever container the
// pipeline is nested within-granularity. However, this conversion adds (and removes)
// new modules to the top-level mlir::ModuleOp scope, which technically violates
// hw::HWModuleLike's IsolatedFromAbove (and thus has previously caused
// concurrency issues via. concurrent additions and removals to the mlir::ModuleOp
// symboltable).
def PipelineToHW : Pass<"lower-pipeline-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower Pipeline to HW";
  let description = [{
    This pass lowers `pipeline.rtp` operations to HW.
  }];
  let constructor = "circt::createPipelineToHWPass()";
  let dependentDialects = [
    "hw::HWDialect", "comb::CombDialect", "seq::SeqDialect"
  ];
  let options = [
    Option<"clockGateRegs", "clock-gate-regs", "bool", "false",
      "Clock gate each register instead of (default) input muxing  (ASIC optimization).">,
    Option<"enablePowerOnValues", "enable-poweron-values", "bool", "false",
      "Add power-on values to the pipeline control registers">
  ];
}

//===----------------------------------------------------------------------===//
// CalyxToHW
//===----------------------------------------------------------------------===//

def CalyxToHW : Pass<"lower-calyx-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower Calyx to HW";
  let description = [{
    This pass lowers Calyx to HW.
  }];
  let constructor = "circt::createCalyxToHWPass()";
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect",
                           "seq::SeqDialect", "sv::SVDialect"];
}

//===----------------------------------------------------------------------===//
// CalyxNative
//===----------------------------------------------------------------------===//
def CalyxNative : Pass<"calyx-native", "mlir::ModuleOp"> {
  let summary = "Callout to the Calyx native compiler and run a pass pipeline";
  let description = [{
    This pass calls out to the native, Rust-based Calyx compiler to run passes
    with it and generate a new, valid, calyx dialect program.
  }];
  let constructor = "circt::createCalyxNativePass()";
  let dependentDialects = ["calyx::CalyxDialect"];
  let options = [
    Option<"passPipeline", "pass-pipeline", "std::string",
            "", "Passes to run with the native compiler">,
  ];
}



//===----------------------------------------------------------------------===//
// CalyxToFSM
//===----------------------------------------------------------------------===//

def CalyxToFSM : Pass<"lower-calyx-to-fsm", "calyx::ComponentOp"> {
  let summary = "Lower Calyx to FSM";
  let description = [{
    This pass lowers a Calyx control schedule to an FSM representation.
    An `fsm.machine` operation is nested within the `control` region of the Calyx
    component. This machine is itself in an intermediate format wherein it has
    no I/O ports and solely contains output statements with `calyx.enable`s
    referencing `calyx.group` and transition logic guarded by the SSA values
    specified in the source control schedule.
    This intermediate state facilitates transformation of the FSM, given that
    top-level I/O has yet to be materialized (one input and output per activated
    group) as well as guard transition logic (every transition must be guarded
    on all groups active within the state having finished). As such, `calyx.enable`
    operations can easily be moved between states without worrying about updating
    transition guards while doing so.

    Eventually, the FSM must be materialized (materialize I/O ports, remove
    `calyx.enable` operations in favor of asserting output ports, guarding
    transitions by input `done` ports) and outlined to a separate module.
  }];
  let constructor = "circt::createCalyxToFSMPass()";
  let dependentDialects = ["fsm::FSMDialect", "comb::CombDialect"];
}

def MaterializeCalyxToFSM : Pass<"materialize-calyx-to-fsm", "calyx::ComponentOp"> {
  let summary = "Materializes an FSM embedded inside the control of this Calyx component.";
  let description = [{
    Materializes the FSM in the control of the component. This materializes the
    top-level I/O of the FSM to receive `group_done` signals as input and
    `group_go` signals as output, based on the `calyx.enable` operations
    used within the states of the FSM.
    Each transition of the FSM is predicated on the enabled groups within a
    state being done, or, for static groups, a separate sub-FSM is instantiated
    to await the group finishing.

    Given an FSM that enables N unique groups, the top-level FSM will have N+1
    in- and output ports, wherein:
    * Input # 0 to N-1 are `group_done` signals
    * Input N is the top-level `go` port
    * Output 0 to N-1 are `group_go` signals
    * Output N is the top-level `done` port
  }];
  let dependentDialects = ["comb::CombDialect", "hw::HWDialect", "fsm::FSMDialect"];
  let constructor = "circt::createMaterializeCalyxToFSMPass()";
}


def CalyxRemoveGroupsFromFSM : Pass<"calyx-remove-groups-fsm", "calyx::ComponentOp"> {
  let summary = "Perform FSM outlining and group removal";
  let description = [{
    This pass will outline the FSM into module scope and replace any SSA value references
    from within the FSM body with additional inputs. Given this, the FSM
    is instantiated as a `fsm.hw_module` operation within the Calyx component.
    Using the FSM I/O (which is the group go/done signals), the `calyx.group`
    operations are removed from the component, with the group go and done signals
    being wired up to the FSM instance.
    Example:
    ```mlir
    calyx.component {
        %reg, ... = calyx.register ... : i1
        calyx.wires {
            // Groups have explicit done signals, and assignments are not guarded
            // by a group go signal.
            calyx.group @A {
                ...
                calyx.assign %reg = ...
                ...
                calyx.group_done %foo ? %bar
            }
        }
        calyx.control {
            // Machine is defined inside the `calyx.control` operation and references
            // SSA values defined outside the machine.
            fsm.machine @control(%A_done : i1) -> (%A_go : i1) {
                ...
                %0 = comb.not %reg // reference some SSA value defined outside the machine
                ...
            }
        }
    }
    ```

    into
    ```mlir
    // The machine has been outlined into module scope, and no longer references
    // any SSA values defined outside the machine. It is now fully independent
    // from any notion of Calyx.
    fsm.machine @control(%A_done : i1, %reg : i1) -> (%A_go : i1) {
        ...
        %0 = comb.not %reg // reference some SSA value defined outside the machine
        ...
    }

    calyx.component {
        %reg, ... = calyx.register ...
        // Done signals are now wires
        %A_done_in, %A_done_out = calyx.wire : i1
        // The FSM is now instantiated as an `fsm.hwinstance` module
        %A_go = fsm.hwinstance @control(%A_done_out, %reg) : ...
        calyx.wires {
            // Groups have been inlined, the group go signal is now a guard for
            // all assignments, and `calyx.group_done` operations have been
            // replaced by wire assignments.
            ...
            calyx.assign %reg = %A_go ? ...
            ...
            calyx.assign %A_done_in = %foo ? %bar
        }
        calyx.control {
        }
    }
    ```
  }];
  let dependentDialects = [
      "fsm::FSMDialect",
      "comb::CombDialect",
      "hw::HWDialect",
      "seq::SeqDialect"
  ];
  let constructor = "circt::createRemoveGroupsFromFSMPass()";
}

//===----------------------------------------------------------------------===//
// FSMToSV
//===----------------------------------------------------------------------===//

def ConvertFSMToSV : Pass<"convert-fsm-to-sv", "mlir::ModuleOp"> {
  let summary = "Convert FSM to SV and HW";
  let constructor = "circt::createConvertFSMToSVPass()";
  let dependentDialects = [
      "circt::comb::CombDialect",
      "circt::emit::EmitDialect",
      "circt::hw::HWDialect",
      "circt::seq::SeqDialect",
      "circt::sv::SVDialect",
  ];
}

//===----------------------------------------------------------------------===//
// FIRRTLToHW
//===----------------------------------------------------------------------===//

def LowerFIRRTLToHW : Pass<"lower-firrtl-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower FIRRTL to HW";
  let description = [{
    Lower a module of FIRRTL dialect to the HW dialect family.
  }];
  let constructor = "circt::createLowerFIRRTLToHWPass()";

  let dependentDialects = [
    "comb::CombDialect",
    "emit::EmitDialect",
    "hw::HWDialect",
    "ltl::LTLDialect",
    "seq::SeqDialect",
    "sim::SimDialect",
    "sv::SVDialect",
    "verif::VerifDialect",
  ];
  let options = [
    Option<"enableAnnotationWarning", "warn-on-unprocessed-annotations",
           "bool", "false",
    "Emit warnings on unprocessed annotations during lower-to-hw pass">,
    Option<"verificationFlavor", "verification-flavor",
           "circt::firrtl::VerificationFlavor",
           "circt::firrtl::VerificationFlavor::None",
           "Specify a verification flavor used in the lowering",
           [{::llvm::cl::values(
            clEnumValN(circt::firrtl::VerificationFlavor::None,
              "none", "Use the flavor specified by the op"),
            clEnumValN(circt::firrtl::VerificationFlavor::IfElseFatal,
              "if-else-fatal", "Use Use `if(cond) else $fatal(..)` format"),
            clEnumValN(circt::firrtl::VerificationFlavor::Immediate,
              "immediate", "Use immediate verif statements"),
            clEnumValN(circt::firrtl::VerificationFlavor::SVA, "sva", "Use SVA")
          )}]
           >
  ];
}

//===----------------------------------------------------------------------===//
// HandshakeToDC
//===----------------------------------------------------------------------===//

def HandshakeToDC : Pass<"lower-handshake-to-dc", "mlir::ModuleOp"> {
  let summary = "Lower Handshake to DC";
  let description = [{
    Lower Handshake to DC operations.
    Currently, a `handshake.func` will be converted into a `hw.module`. This
    is principally an incorrect jump of abstraction - DC does not imply any
    RTL/hardware semantics. However, DC does not define a container operation,
    and there does not exist an e.g. `func.graph_func` which would be a generic
    function with graph region behaviour. Thus, for now, we just use `hw.module`
    as a container operation.
  }];
  let dependentDialects = [
    "dc::DCDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "hw::HWDialect"
  ];
  let options = [
    Option<"clkName", "clk-name", "std::string", "\"clk\"",
           "Name of the clock signal to use in the generated DC module">,
    Option<"rstName", "rst-name", "std::string", "\"rst\"",
           "Name of the reset signal to use in the generated DC module">,
  ];
}

//===----------------------------------------------------------------------===//
// DCToHW
//===----------------------------------------------------------------------===//

def DCToHW : Pass<"lower-dc-to-hw"> {
  let summary = "Lower DC to HW";
  let description = [{
    Lower DC to ESI/hw/comb/seq operations.
    In case the IR contains DC operations that need to be clocked (fork, buffer),
    there must exist a clock and reset signal in the parent `FunctionLike`
    operation. These arguments are to be marked with a `dc.clock` and `dc.reset`
    attribute, respectively.
  }];
  let constructor = "circt::createDCToHWPass()";
  let dependentDialects = ["dc::DCDialect", "esi::ESIDialect", "hw::HWDialect",
      "comb::CombDialect", "seq::SeqDialect"];
}

//===----------------------------------------------------------------------===//
// HandshakeToHW
//===----------------------------------------------------------------------===//

def HandshakeToHW : Pass<"lower-handshake-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower Handshake to ESI/HW/Comb/Seq";
  let description = [{
    Lower Handshake to ESI/HW/Comb/Seq.
  }];
  let constructor = "circt::createHandshakeToHWPass()";
  let dependentDialects = ["hw::HWDialect", "esi::ESIDialect", "comb::CombDialect",
                           "seq::SeqDialect"];
}

//===----------------------------------------------------------------------===//
// Moore to Comb/HW/LLHD
//===----------------------------------------------------------------------===//

def ConvertMooreToCore : Pass<"convert-moore-to-core", "mlir::ModuleOp"> {
  let summary = "Convert Moore to Core";
  let description = [{
    This pass translates Moore to the core dialects.
  }];
  let constructor = "circt::createConvertMooreToCorePass()";
  let dependentDialects = [
    "comb::CombDialect",
    "hw::HWDialect",
    "llhd::LLHDDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::scf::SCFDialect",
    "sim::SimDialect",
    "verif::VerifDialect",
  ];
}

//===----------------------------------------------------------------------===//
// HWToLLVM
//===----------------------------------------------------------------------===//

def ConvertHWToLLVM : Pass<"convert-hw-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert HW to LLVM";
  let description = [{
    This pass translates HW to LLVM.
  }];
  let constructor = "circt::createConvertHWToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// HWArithToHW
//===----------------------------------------------------------------------===//

def HWArithToHW : Pass<"lower-hwarith-to-hw", "mlir::ModuleOp"> {
  let summary = "Lower HWArith to HW/Comb";
  let description = [{
    This pass lowers HWArith to HW/Comb.
  }];
  let constructor = "circt::createHWArithToHWPass()";
  let dependentDialects = ["hw::HWDialect", "comb::CombDialect", "sv::SVDialect"];
}

//===----------------------------------------------------------------------===//
// HWToSystemC
//===----------------------------------------------------------------------===//

def ConvertHWToSystemC : Pass<"convert-hw-to-systemc", "mlir::ModuleOp"> {
  let summary = "Convert HW to SystemC";
  let description = [{
    This pass translates a HW design into an equivalent SystemC design.
  }];
  let constructor = "circt::createConvertHWToSystemCPass()";
  let dependentDialects = ["systemc::SystemCDialect", "mlir::emitc::EmitCDialect"];
}

//===----------------------------------------------------------------------===//
// HWToSV
//===----------------------------------------------------------------------===//

def LowerHWToSV : Pass<"lower-hw-to-sv", "hw::HWModuleOp"> {
  let summary = "Convert HW to SV";
  let description = [{
    This pass converts various HW contructs to SV.
  }];
  let constructor = "circt::createLowerHWToSVPass()";
  let dependentDialects = ["sv::SVDialect"];
}

//===----------------------------------------------------------------------===//
// HWToBTOR2
//===----------------------------------------------------------------------===//

def ConvertHWToBTOR2 : Pass<"convert-hw-to-btor2", "mlir::ModuleOp"> {
  let summary = "Convert HW to BTOR2";
  let description = [{
    This pass converts a HW module into a state transition system that is then
    directly used to emit btor2. The output of this pass is thus a btor2 string.
  }];
  let constructor = "circt::createConvertHWToBTOR2Pass()";
  let dependentDialects = ["hw::HWDialect", "sv::SVDialect", "comb::CombDialect",
                           "seq::SeqDialect"];
}

//===----------------------------------------------------------------------===//
// LTLToCore
//===----------------------------------------------------------------------===//

def LowerLTLToCore : Pass<"lower-ltl-to-core", "hw::HWModuleOp"> {
  let summary = "Convert LTL and Verif to Core";
  let description = [{
    This pass converts ltl and verif operations to core ones. This can be done directly
    without going through FSM if we're only working with overlapping properties (no delays).
  }];
  let constructor = "circt::createLowerLTLToCorePass()";
  let dependentDialects = [
    "hw::HWDialect", "sv::SVDialect", "comb::CombDialect",
    "seq::SeqDialect"
  ];
}

//===----------------------------------------------------------------------===//
// VerifToSV
//===----------------------------------------------------------------------===//

def LowerVerifToSV : Pass<"lower-verif-to-sv", "hw::HWModuleOp"> {
  let summary = "Convert Verif to SV";
  let description = [{
    This pass converts various Verif contructs to SV.
  }];
  let constructor = "circt::createLowerVerifToSVPass()";
  let dependentDialects = [
    "sv::SVDialect", "hw::HWDialect", "comb::CombDialect"];
}

//===----------------------------------------------------------------------===//
// CFToHandshake
//===----------------------------------------------------------------------===//

def CFToHandshake : Pass<"lower-cf-to-handshake", "mlir::ModuleOp"> {
  let summary = "Lower func and CF into Handshake IR";
  let constructor = "circt::createCFToHandshakePass()";
  let dependentDialects = ["handshake::HandshakeDialect"];
  let options =
    [Option<"sourceConstants", "source-constants", "bool", "false",
            "If true, will connect constants to source operations instead of "
            "to the control network. May reduce the size of the final circuit.">,
     Option<"disableTaskPipelining", "disable-task-pipelining", "bool", "false",
            "If true, will disable support for task pipelining. This relaxes the "
            "restrictions put on the structure of the input CDFG. Disabling "
            "task pipelining may severely reduce kernel II.">];
}

def HandshakeRemoveBlock : Pass<"handshake-remove-block-structure", "handshake::FuncOp"> {
  let summary = "Remove block structure in Handshake IR";
  let constructor = "circt::createHandshakeRemoveBlockPass()";
}

//===----------------------------------------------------------------------===//
// ConvertToArcs
//===----------------------------------------------------------------------===//

def ConvertToArcs : Pass<"convert-to-arcs", "mlir::ModuleOp"> {
  let summary = "Outline logic between registers into state transfer arcs";
  let description = [{
    This pass outlines combinational logic between registers into state transfer
    arc definitions. The the original combinational logic and register is
    replaced with an arc invocation, where the register is now represented as a
    latency.
  }];
  let constructor = "circt::createConvertToArcsPass()";
  let dependentDialects = ["circt::arc::ArcDialect", "circt::hw::HWDialect"];
  let options = [
    Option<"tapRegisters", "tap-registers", "bool", "true",
      "Make registers observable">,
  ];
}

//===----------------------------------------------------------------------===//
// ConvertCombToArith
//===----------------------------------------------------------------------===//

def ConvertCombToArith : Pass<"convert-comb-to-arith"> {
  let summary = "Convert combinational ops and constants into arith ops";
  let constructor = "circt::createConvertCombToArithPass()";
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

//===----------------------------------------------------------------------===//
// ConvertCombToSMT
//===----------------------------------------------------------------------===//

def ConvertCombToSMT : Pass<"convert-comb-to-smt"> {
  let summary = "Convert combinational ops and constants to SMT ops";
  // Need to depend on HWDialect because some 'comb' canonicalization patterns
  // build 'hw.constant' operations.
  let dependentDialects = [
    "mlir::smt::SMTDialect", "hw::HWDialect", "mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ConvertHWToSMT
//===----------------------------------------------------------------------===//

def ConvertHWToSMT : Pass<"convert-hw-to-smt", "mlir::ModuleOp"> {
  let summary = "Convert HW ops and constants to SMT ops";
  let dependentDialects = ["mlir::smt::SMTDialect", "mlir::func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// ConvertVerifToSMT
//===----------------------------------------------------------------------===//

def ConvertVerifToSMT : Pass<"convert-verif-to-smt", "mlir::ModuleOp"> {
  let summary = "Convert Verif ops to SMT ops";
  let dependentDialects = [
    "mlir::smt::SMTDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect"
  ];
  let options = [
    Option<"risingClocksOnly", "rising-clocks-only", "bool",
           /*default=*/"false",
           "When lowering verif.bmc ops, only consider the circuit and property"
           "on rising clock edges.">,
  ];
}

//===----------------------------------------------------------------------===//
// ConvertArcToLLVM
//===----------------------------------------------------------------------===//

def LowerArcToLLVM : Pass<"lower-arc-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower state transfer arc representation to LLVM";
  let constructor = "circt::createLowerArcToLLVMPass()";
  let dependentDialects = [
    "arc::ArcDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ConvertSMTToZ3LLVM
//===----------------------------------------------------------------------===//

def LowerSMTToZ3LLVM : Pass<"lower-smt-to-z3-llvm", "mlir::ModuleOp"> {
  let summary = "Lower the SMT dialect to LLVM IR calling the Z3 API";
  let dependentDialects = [
    "mlir::smt::SMTDialect", "mlir::LLVM::LLVMDialect", "mlir::scf::SCFDialect",
    "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect"
  ];
  let options = [
    Option<"debug", "debug", "bool", "false",
           "Insert additional printf calls printing the solver's state to "
           "stdout (e.g. at check-sat operations) for debugging purposes">,
  ];
}

//===----------------------------------------------------------------------===//
// ConvertSeqToSV
//===----------------------------------------------------------------------===//

def LowerSeqToSV: Pass<"lower-seq-to-sv",  "mlir::ModuleOp"> {
  let summary = "Lower sequential firrtl ops to SV.";
  let constructor = "circt::createLowerSeqToSVPass()";
  let dependentDialects = [
    "circt::emit::EmitDialect",
    "circt::hw::HWDialect",
    "circt::sv::SVDialect",
  ];
  let options = [
    Option<"disableRegRandomization", "disable-reg-randomization", "bool", "false",
           "Disable emission of register randomization code">,
    Option<"disableMemRandomization", "disable-mem-randomization", "bool", "false",
           "Disable emission of memory randomization code">,
    Option<"emitSeparateAlwaysBlocks", "emit-separate-always-blocks", "bool", "false",
           "Emit assigments to registers in separate always blocks">,
    Option<"lowerToAlwaysFF", "lower-to-always-ff", "bool", "true",
           "Place assignments to registers into `always_ff` blocks if possible">
  ];
  let statistics = [
    Statistic<"numSubaccessRestored", "num-subaccess-restored",
      "Number of lhs subaccess operations restored ">
  ];
}

def LowerFirMem : Pass<"lower-seq-firmem", "mlir::ModuleOp"> {
  let summary = "Lower seq.firmem ops to instances of hw.module.generated ops";
  let constructor = "circt::createLowerFirMemPass()";
  let dependentDialects = ["circt::hw::HWDialect"];
}

//===----------------------------------------------------------------------===//
// ConvertSimToSV
//===----------------------------------------------------------------------===//

def LowerSimToSV: Pass<"lower-sim-to-sv",  "mlir::ModuleOp"> {
  let summary = "Lower simulator-specific `sim` ops to SV.";
  let constructor = "circt::createLowerSimToSVPass()";
  let dependentDialects = [
    "circt::comb::CombDialect",
    "circt::emit::EmitDialect",
    "circt::seq::SeqDialect",
    "circt::sv::SVDialect",
    "circt::hw::HWDialect"
  ];
  let options = [
    Option<"keepBitwiseLogicalOps", "keep-bitwise-logical-ops", "bool", "false",
           "Keep bitwise logic in the output">
  ];
}

//===----------------------------------------------------------------------===//
// ConvertCombToAIG
//===----------------------------------------------------------------------===//

def ConvertAIGToComb: Pass<"convert-aig-to-comb",  "hw::HWModuleOp"> {
  let summary = "Lower AIG ops to Comb ops.";
  let description = [{
    This pass converts AIG operations to Comb operations. This is mostly
    used for verifying post-synthesis results.
  }];
  let dependentDialects = [
    "circt::comb::CombDialect",
    "circt::aig::AIGDialect",
  ];
  let options = [
    Option<"keepBitwiseLogicalOps", "keep-bitwise-logical-ops", "bool", "false",
           "Keep bitwise logic in the output">
  ];
}

//===----------------------------------------------------------------------===//
// ConvertCombToAIG
//===----------------------------------------------------------------------===//

def ConvertCombToAIG: Pass<"convert-comb-to-aig",  "hw::HWModuleOp"> {
  let summary = "Lower Comb ops to AIG ops.";
  let dependentDialects = [
    "circt::comb::CombDialect",
    "circt::aig::AIGDialect",
  ];

  let options = [
    ListOption<"additionalLegalOps", "additional-legal-ops", "std::string",
               "Specify additional legal ops to partially legalize Comb to AIG">,
    Option<"maxEmulationUnknownBits", "max-emulation-unknown-bits", "uint32_t", "10",
           "Maximum number of unknown bits to emulate in a table lookup">
  ];
}

#endif // CIRCT_CONVERSION_PASSES_TD
