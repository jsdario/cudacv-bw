
# ORIGINAL CODE FROM UDACITY #
############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

cmake_minimum_required(VERSION 2.8)
find_package(CUDA)
find_package(OpenCV REQUIRED)
file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (to_bw_files main.cpp)

CUDA_ADD_EXECUTABLE(to_bw ${to_bw_files} ${hdr} ${cu})
target_link_libraries(to_bw ${OpenCV_LIBS})