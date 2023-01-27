if(CIRCT_LEC_ENABLED)
  add_circt_verification_library(CIRCTSMT
    Solver.cpp
    Circuit.cpp
    LogicExporter.cpp

    LINK_COMPONENTS
    Core

    LINK_LIBS
    ${CIRCT_Z3_LIBS}
    
    LINK_LIBS PUBLIC
    CIRCTCalyx
    CIRCTESI
    CIRCTFIRRTL
    CIRCTFSM
    CIRCTHandshake
    CIRCTHWArith
    CIRCTInteropDialect
    CIRCTLLHD
    CIRCTMoore
    CIRCTPipelineOps
    CIRCTSSP
    CIRCTSystemC

  )
  target_include_directories(CIRCTSMT PRIVATE ${CIRCT_Z3_INCLUDES})
endif()
