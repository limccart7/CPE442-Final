
CC = g++ 
FLAGS = -Wall -O0 -g -lpthread
FLAGS_OPT = -Wall -O3 -g -lpthread
PKG_CONFIG = `pkg-config --libs --cflags opencv4`


sobel_threaded: sobel_threaded.o 
	$(CXX) sobel_threaded.o $(FLAGS) -o sobel_threaded $(PKG_CONFIG)

sobel_threaded.o: sobel_threaded.cpp sobel_threaded.hpp
	$(CXX) sobel_threaded.cpp sobel_threaded.hpp $(FLAGS) -c $(PKG_CONFIG)

sobel_threaded_opt: sobel_threaded_opt.o 
	$(CXX) sobel_threaded_opt.o $(FLAGS) -o sobel_threaded_opt $(PKG_CONFIG)

sobel_threaded_opt.o: sobel_threaded.cpp sobel_threaded.hpp
	$(CXX) sobel_threaded_opt.cpp sobel_threaded.hpp $(FLAGS) -c $(PKG_CONFIG)

sobel_threaded_O3: sobel_threaded_O3.o 
	$(CXX) sobel_threaded_O3.o $(FLAGS) -o sobel_threaded_O3 $(PKG_CONFIG)

sobel_threaded_O3.o: sobel_threaded.cpp sobel_threaded.hpp
	$(CXX) sobel_threaded_O3.cpp sobel_threaded.hpp $(FLAGS_OPT) -c $(PKG_CONFIG)

clean:
	rm -f *.o lab6 *.gch
