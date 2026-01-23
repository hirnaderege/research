CC = nvcc
CFLAGS = -std=c++14 -arch=sm_60
TARGET = allocator
SOURCES = allocator.h

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)