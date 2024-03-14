
CC = g++ 
FLAGS = -Wall -O0 -g -lpthread -fpermissive
PKG_CONFIG = `pkg-config --libs --cflags opencv4`


lab5: sobel_threaded.o lab5.o
	$(CXX) sobel_threaded.o lab5.o $(FLAGS) -o lab5 $(PKG_CONFIG)

sobel_threaded.o: sobel_threaded.cpp sobel_threaded.hpp
	$(CXX) sobel_threaded.cpp sobel_threaded.hpp $(FLAGS) -c $(PKG_CONFIG)

lab5.o: lab5.cpp sobel_threaded.hpp
	$(CXX) lab5.cpp sobel_threaded.hpp $(FLAGS) -c $(PKG_CONFIG)
clean:
	rm -f *.o lab5 *.gch
