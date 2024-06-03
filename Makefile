COMPILER = gcc
COMPILER_FLAGS = -Wall -Werror -Wextra -g -std=c++17
LIBRARIES = -lgtest, -lgtest_main -lstdc++

CC = $(COMPILER) $(COMPILER_FLAGS)
LIB = $(LIBRARIES)
OS := $(shell uname)

ifeq ($(OS), Linux)
	LIB += -lrt -lm -lsubunit -lpthread
endif


all: build

build: clean
	-mkdir build

rebuild:

test: clean
	$(CC) tests/*.cc -o tests/res_test $(LIB)
	./tests/res_test

clean:

clear: clean