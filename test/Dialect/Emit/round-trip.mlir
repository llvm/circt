// RUN: circt-opt %s | circt-opt | FileCheck %s

sv.macro.decl @SomeMacro

// CHECK: emit.file "some-file.sv" sym @SomeFile.sv
emit.file "some-file.sv" sym @SomeFile.sv {
  // CHECK: emit.verbatim "SimpleVerbatim"
  emit.verbatim "SimpleVerbatim"

}

// CHECK: emit.file_list "filelist.f", [@SomeFile.sv] sym @SomeFileList
emit.file_list "filelist.f", [@SomeFile.sv] sym @SomeFileList
