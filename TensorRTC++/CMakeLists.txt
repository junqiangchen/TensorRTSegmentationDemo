cmake_minimum_required(VERSION 3.3...3.12 FATAL_ERROR)
foreach(p
    CMP0071 # 3.10: Let AUTOMOC and AUTOUIC process GENERATED files
    )
  if(POLICY ${p})
    cmake_policy(SET ${p} NEW)
  endif()
endforeach()

project(OnnxTensorRTSeg)

set(OpenCV_INCLUDE_DIRS "C:/vs2019/opencv3.4.5/include")
set(OpenCV_LIBS_DIRS "C:/vs2019/opencv3.4.5/lib")
file(GLOB OpenCV_LIBS
	${OpenCV_LIBS_DIRS}/*.lib
)

set(CUDA_INCLUDE_DIRS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/include")
set(TENSORRT_LIBS_DIRS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/lib")
set(CUDA_LIBS_DIRS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/lib/x64")
file(GLOB Cuda_LIBS
	${CUDA_LIBS_DIRS}/*.lib
)
file(GLOB Tensorrt_LIBS
	${TENSORRT_LIBS_DIRS}/*.lib
)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

# Set your files and resources here
file(GLOB_RECURSE files 
"*.cpp"
"*.c"
"*.hpp"
"*.h"
)


# CMAKE_AUTOMOC in ON so the MocHdrs will be automatically wrapped.
add_executable(OnnxTensorRTSeg ${files})
target_link_libraries(OnnxTensorRTSeg ${OpenCV_LIBS} ${Cuda_LIBS} ${Tensorrt_LIBS})
