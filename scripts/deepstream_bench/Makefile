CUDA_VER:=12.6
DS_SRC:=/opt/nvidia/deepstream/deepstream-7.1/sources

CC:= g++
CFLAGS:= -Wall -std=c++11 -shared -fPIC
CFLAGS+= -I$(DS_SRC)/includes -I/usr/local/cuda-$(CUDA_VER)/include
LIBS:= -lnvinfer
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvdsinfer_yolov8_parser.cpp
TARGET_LIB:= libnvdsinfer_yolov8_parser.so

all: $(TARGET_LIB)

$(TARGET_LIB): $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
