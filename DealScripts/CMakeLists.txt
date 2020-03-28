# Set the name of the project and target:
SET(TARGET "FEMatrixGenerator")

FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
INCLUDE_DIRECTORIES(include)
SET(TARGET_SRC 
  ${TARGET_SRC}  
  main.cc
  )

# Usually, you will not need to modify anything beyond this point...
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

set (DEAL_II_DIR  /home/julian/deal.ii-candi/deal.II-v9.1.1)

FIND_PACKAGE(deal.II 9.1.1 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

# === Dependecies : === 
IF(NOT DEAL_II_WITH_TRILINOS) # keep in one line
  MESSAGE(FATAL_ERROR "
Error! This programm requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_TRILINOS = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options
    DEAL_II_WITH_TRILINOS = ${DEAL_II_WITH_TRILINOS}
which conflict with the requirements."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
