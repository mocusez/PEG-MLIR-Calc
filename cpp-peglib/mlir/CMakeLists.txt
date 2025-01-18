cmake_minimum_required(VERSION 3.14)
project(example-mlir)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(mlir-hello LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(LLVM_DIR /usr/lib/llvm-18/lib/cmake/llvm/)
find_package(LLVM REQUIRED CONFIG)
set(MLIR_DIR /usr/lib/llvm-18/lib/cmake/mlir/)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LINK_COMPONENTS
  Support
  )
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  message(STATUS "Building for x86_64 architecture")
  add_executable(mlir-calc calc.cc arith.h arith.cpp LowerToLLVM.cpp Passes.h)
else()
  add_executable(mlir-calc calc.cc arith.h arith_general_simd.cpp LowerToLLVM.cpp Passes.h)
endif()
target_include_directories(mlir-calc PRIVATE ..)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(mlir-calc
  PRIVATE
  ${add_link_deps}
  ${dialect_libs}
  ${conversion_libs}
    MLIRSupport
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRVectorDialect
    MLIRPass
    MLIRTransforms

    MLIRBuiltinToLLVMIRTranslation
    MLIRExecutionEngine
    )
