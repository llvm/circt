// RUN: firtool --split-verilog --emit-omir -o=%t%{fs-sep} %s
// RUN: cat %t%{fs-sep}some-filelist.f  | FileCheck %s --check-prefix=FILELIST
// RUN: cat %t%{fs-sep}some-file.sv  | FileCheck %s --check-prefix=FILE

sv.macro.decl @Macro

hw.module private @SomeModule() {
}

// FILE: SimpleVerbatim
// FILE-NEXT: WithNewlines
// FILE-NEXT: A
// FILE-NEXT: B
// FILE-NEXT: WithEmptyNewlines
// FILE-EMPTY:
// FILE-EMPTY:
// FILE-NEXT: X
// FILE-NEXT: `ifdef Macro
// FILE-NEXT:   `define Macro 1
// FILE-NEXT: `else
// FILE-NEXT:   `define Macro 2
// FILE-NEXT: `endif
// FILE-NEXT: module SomeModule();
// FILE-NEXT: endmodule
// FILE-EMPTY:
// FILE-NEXT: Some Reference: Macro

emit.file "some-file.sv" sym @SomeFile.sv {
  emit.verbatim "SimpleVerbatim"
  emit.verbatim "WithNewlines\nA\nB\n"
  emit.verbatim "WithEmptyNewlines\n\n\nX"

  sv.ifdef @Macro {
    sv.macro.def @Macro "1"
  } else {
    sv.macro.def @Macro "2"
  }

  emit.ref @SomeModule

  sv.verbatim "Some Reference: {{0}}" {symbols = [@Macro]}
}

// FILELIST: some-file.sv
emit.file_list "some-filelist.f", [@SomeFile.sv] sym @SomeFileList
