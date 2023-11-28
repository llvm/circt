; RUN: firtool %s
; RUN: firtool %s -preserve-aggregate=all -scalarize-public-modules=false
circuit NoSubAccessesWithProbes :
  extmodule Ext :
    output x : {a : Probe<UInt<2>[2]>, b : UInt<2>}[3]

  ; XXX: Modified to not use input probes, get probe from ext, widths.
  ; SPEC EXAMPLE BEGIN:
  module NoSubAccessesWithProbes :
    input i : UInt<5>
    input c : const UInt<5>
    output p : Probe<UInt>

    inst e of Ext

    ; Illegal: e.x[i], e.x[c]
    ; Illegal: e.x[0].a[i], e.x[0].a[c]

    ; Legal:
    define p = e.x[0].a[1]
   ; SPEC EXAMPLE END
