// RUN: circt-opt %s -verify-diagnostics --split-input-file

sv.macro.decl @SomeMacro

// expected-error @below {{referenced operation is not a file: @SomeMacro}}
emit.file_list "filelist.f", [@SomeMacro]

// -----

// expected-error @below {{invalid symbol reference: @InvalidRef}}
emit.file_list "filelist.f", [@InvalidRef]
