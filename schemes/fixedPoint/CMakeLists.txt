cmake_minimum_required (VERSION 2.6)
project (FixedPoint)

# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/FixedPointConfig.h.in"
  "${PROJECT_BINARY_DIR}/FixedPointConfig.h"
  )

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")


include_directories ("${PROJECT_SOURCE_DIR}/lib")
add_subdirectory (lib)
set (EXTRA_LIBS ${EXTRA_LIBS} lib)


# add the executable
add_executable(fixedPoint fixedPointCompressionExample.cpp)
target_link_libraries (fixedPoint  ${EXTRA_LIBS})

