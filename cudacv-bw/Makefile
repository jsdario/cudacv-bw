NVCC=nvcc

###################################
# These are the default install   #
# locations on most linux distros #
###################################

OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui

CUDA_INCLUDEPATH=/usr/local/cuda-6.5/include


NVCC_OPTS=-O3 -arch=sm_11 -Xcompiler -Wall -Xcompiler -Wextra -m64

GCC_OPTS=-O3 -Wall -Wextra -m64

student: main.o rgba_to_grey.o Makefile
	$(NVCC) -o image_process main.o rgba_to_grey.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cpp timer.h utils.h image_process.cpp
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH)

rgba_to_grey.o: rgba_to_grey.cu utils.h
	nvcc -c rgba_to_grey.cu $(NVCC_OPTS)

clean:
	rm -f *.o *.png hw
