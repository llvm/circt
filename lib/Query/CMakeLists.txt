add_circt_library(CIRCTQuery
  Query.cpp
  LINK_LIBS PUBLIC
  # need to figure out what to put here
  CIRCTFIRRTL
  CIRCTHW
  )
