
CC = g++ 
FLAGS = -Wall -O0 -g
PKG_CONFIG = `pkg-config --libs --cflags opencv4`


lab3: sobel_filter.o lab3.o
	$(CXX) sobel_filter.o lab3.o $(FLAGS) -o lab3 $(PKG_CONFIG)

sobel_filter.o: sobel_filter.cpp sobel_filter.hpp
	$(CXX) sobel_filter.cpp sobel_filter.hpp $(FLAGS) -c $(PKG_CONFIG)

lab3.o: lab3.cpp sobel_filter.hpp
	$(CXX) lab3.cpp sobel_filter.hpp $(FLAGS) -c $(PKG_CONFIG)
clean:
	rm -f *.o lab3 *.gch

