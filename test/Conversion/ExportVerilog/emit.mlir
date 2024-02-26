// RUN: firtool --split-verilog --emit-omir -o=%t%{fs-sep} %s
// RUN: cat %t%{fs-sep}some-filelist.f  | FileCheck %s --check-prefix=FILELIST
// RUN: cat %t%{fs-sep}some-file.sv  | FileCheck %s --check-prefix=FILE

// FILE: SimpleVerbatim
// FILE-NEXT: WithNewlines
// FILE-NEXT: A
// FILE-NEXT: B
// FILE-NEXT: WithEmptyNewlines
// FILE: X
emit.file "some-file.sv" sym @SomeFile.sv {
  emit.verbatim "SimpleVerbatim"
  emit.verbatim "WithNewlines\nA\nB\n"
  emit.verbatim "WithEmptyNewlines\n\n\nX"
}

// FILELIST: some-file.sv
emit.file_list "some-filelist.f", [@SomeFile.sv] sym @SomeFileList
