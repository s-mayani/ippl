# -----------------------------------------------------------------------------
# alpine/CMakeLists.txt
#
# This module provides example physics applications using IPPL.
# -----------------------------------------------------------------------------

function(add_example name)
  add_executable(${name} ${name}.cpp)

  target_link_libraries(${name} PRIVATE IPPL::ippl)

  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

add_alpine_example(UniformField)
