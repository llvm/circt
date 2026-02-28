set(LLVM_LINK_COMPONENTS
  Support
  TableGen
)

add_tablegen(circt-tblgen CIRCT
  circt-tblgen.cpp
  FIRRTLAnnotationsGen.cpp
  FIRRTLIntrinsicsGen.cpp
)

target_link_libraries(circt-tblgen
  PRIVATE
  MLIRTableGen
  MLIRTblgenLib
)

llvm_update_compile_flags(circt-tblgen)
mlir_check_all_link_libraries(circt-tblgen)
