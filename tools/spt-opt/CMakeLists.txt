set(LIB_LIBS
  MLIRAnalysis
  MLIRLLVMIR
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
)
add_llvm_library(SPTSptOptMain
  spt-opt.cpp
)
target_link_libraries(SPTSptOptMain
  ${LIB_LIBS}
)

#get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
#get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
#  ${dialect_libs}
#  ${conversion_libs}
  MLIRDialect
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIROptLib
  LLVMSupport
  LLVMCore
  LLVMAsmParser
)

add_llvm_tool(spt-opt
 spt-opt.cpp
)
# Manually expand the target library, since our SPT libraries
# aren't plugged into the LLVM dependency tracking. If we don't
# do this then we can't insert the CodeGen library after ourselves
llvm_expand_pseudo_components(TARGET_LIBS AllTargetsCodeGens)

# Prepend LLVM in front of every target, this is how the library
# are named with CMake
SET(targets_to_link)
FOREACH(t ${TARGET_LIBS})
  LIST(APPEND targets_to_link "LLVM${t}")
ENDFOREACH(t)

llvm_update_compile_flags(spt-opt)
target_link_libraries(spt-opt PRIVATE ${LIBS} ${targets_to_link})

