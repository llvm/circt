; Ensure firtool can dump MLIR from various points of the pipeline.
; Check passes from different components and registration categories.
; RUN: firtool %s -mlir-print-ir-before=cse |& FileCheck %s -Dpass=CSE
; RUN: firtool %s -mlir-print-ir-before=strip-debuginfo -strip-debug-info |& FileCheck %s -Dpass=StripDebugInfo
; RUN: firtool %s -mlir-print-ir-before=firrtl-inliner -inline |& FileCheck %s -Dpass=Inliner
; RUN: firtool %s -mlir-print-ir-before=firrtl-lower-annotations |& FileCheck %s -Dpass=LowerFIRRTLAnnotations
; RUN: firtool %s -mlir-print-ir-before=firrtl-grand-central -firrtl-grand-central |& FileCheck %s -Dpass=GrandCentral
; RUN: firtool %s -mlir-print-ir-before=export-verilog |& FileCheck %s -Dpass=ExportVerilog
; RUN: firtool %s -mlir-print-ir-before=export-split-verilog -split-verilog -o %t |& FileCheck %s -Dpass=ExportSplitVerilog
; RUN: firtool %s -mlir-print-ir-before=hw-cleanup |& FileCheck %s -Dpass=HWCleanup
; CHECK: IR Dump Before [[pass]]
circuit Empty:
  module Empty:

