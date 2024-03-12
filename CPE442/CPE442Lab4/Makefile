
CC = g++ 
FLAGS = -Wall -O0 -g -lpthread
PKG_CONFIG = `pkg-config --libs --cflags opencv4`


lab4: sobel_threaded.o lab4.o
	$(CXX) sobel_threaded.o lab4.o $(FLAGS) -o lab4 $(PKG_CONFIG)

sobel_threaded.o: sobel_threaded.cpp sobel_threaded.hpp
	$(CXX) sobel_threaded.cpp sobel_threaded.hpp $(FLAGS) -c $(PKG_CONFIG)

lab4.o: lab4.cpp sobel_threaded.hpp
	$(CXX) lab4.cpp sobel_threaded.hpp $(FLAGS) -c $(PKG_CONFIG)
clean:
	rm -f *.o lab4 *.gch
