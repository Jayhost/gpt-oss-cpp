CXX = g++
CXXFLAGS = -O3 -std=c++17 -fsanitize=address -fopenmp -DNDEBUG -Wall
LDFLAGS = -fopenmp

OBJS = main.o gguf.o model.o tokenizer.o

TARGET = gptoss

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean