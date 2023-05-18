if(CIRCT_LEC_ENABLED)
  add_circt_verification_library(CIRCTSMT
    Solver.cpp
    Circuit.cpp
    LogicExporter.cpp

    LINK_COMPONENTS
    Core

    LINK_LIBS
    ${Z3_LIBRARIES}
    
    LINK_LIBS PUBLIC
    MLIRTransforms
    MLIRTranslateLib
    CIRCTComb
    CIRCTHW
    CIRCTSupport
  )
  target_include_directories(CIRCTSMT PRIVATE ${CIRCT_Z3_INCLUDES})
endif()
