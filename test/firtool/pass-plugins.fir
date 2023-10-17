; RUN: firtool -mlir-timing -initial-firrtl-pass-plugin='strip-debuginfo' \ 
; RUN:   -high-firrtl-pass-plugin='strip-debuginfo' \
; RUN:   -low-firrtl-pass-plugin='strip-debuginfo'  \
; RUN:   -hw-pass-plugin='strip-debuginfo'  \
; RUN:   -sv-pass-plugin='strip-debuginfo'  %s 2>&1 | FileCheck %s

; CHECK: FIR Parser
; CHECK: StripDebugInfo
; CHECK: LowerFIRRTLAnnotations
; CHECK: StripDebugInfo
; CHECK: EmitOMIR
; CHECK: StripDebugInfo
; CHECK: LowerFIRRTLToHW
; CHECK: StripDebugInfo
; CHECK: LowerSeqToSV
; CHECK: StripDebugInfo
; CHECK: ExportVerilog
circuit Example: 
  module Example:
