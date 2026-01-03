add_circt_library(CIRCTArcilator
  pipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/circt/Tools/arcilator

  LINK_LIBS PUBLIC
  CIRCTArc
  CIRCTArcToLLVM
  CIRCTArcTransforms
  CIRCTCombToArith
  CIRCTConvertToArcs
  CIRCTEmitTransforms
  CIRCTExportArc
  CIRCTHWTransforms
  CIRCTLLHD
  CIRCTOMTransforms
  CIRCTSeqToSV
  CIRCTSeqTransforms
  CIRCTSimTransforms
  CIRCTSupport
  CIRCTTransforms
  CIRCTVerif
  MLIRLLVMIRTransforms
)

# Arcilator Runtime Library
set(ARC_RUNTIME_INCLUDE ${CIRCT_MAIN_INCLUDE_DIR}/circt/Tools/arcilator/)
add_subdirectory(ArcRuntime)

install(TARGETS ArcRuntime
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  COMPONENT ArcRuntime
)

install(FILES
  ${ARC_RUNTIME_INCLUDE}/ArcRuntime/ArcRuntime.h
  ${ARC_RUNTIME_INCLUDE}/ArcRuntime/Common.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ArcRuntime
  COMPONENT ArcRuntime
)
