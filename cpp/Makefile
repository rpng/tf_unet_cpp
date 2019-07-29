SHELL := /bin/bash
CPP := g++
CFLAGS := -O3 -march=native -std=c++11 -fPIC -Wall -g
LDFLAGS := -ltensorflow `pkg-config --libs --cflags opencv4` -I/usr/local/include/opencv4 -I./build
ROSFLAGS := `pkg-config --libs --cflags rosbag` `pkg-config --libs --cflags sensor_msgs` `pkg-config --libs --cflags geometry_msgs` `pkg-config --libs --cflags cv_bridge`

all: obj main bag
main: obj
	mkdir -p build
	$(CPP) $(CFLAGS) src/main.cpp build/unet.o $(LDFLAGS) -Isrc -o main
	mv main build
bag: obj
	mkdir -p build
	$(CPP) $(CFLAGS) -Isrc build/unet.o src/add_to_bag.cpp \
	   	$(LDFLAGS) $(ROSFLAGS) -o add_to_bag
	mv add_to_bag build
obj:
	mkdir -p build
	$(CPP) $(CFLAGS) -c src/unet.cpp $(LDFLAGS) -Isrc
	mv *.o build
clean:
	rm -rf build
