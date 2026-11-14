CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Iinclude
COMMON = src/common.cpp src/models.cpp

all: train predict

train: src/main.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) src/main.cpp $(COMMON) -o train

predict: src/predict.cpp $(COMMON)
	$(CXX) $(CXXFLAGS) src/predict.cpp $(COMMON) -o predict

clean:
	rm -f train predict
