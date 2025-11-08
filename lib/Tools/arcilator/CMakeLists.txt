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
