SCRIPTS_DIR=.
BIN_DIR=.

CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops -Wno-unused-result -std=c99

all: road2vec

road2vec: main.c
	$(CC) main.c -o ${BIN_DIR}/road2vec $(CFLAGS)

clean:
	pushd ${BIN_DIR} && rm -rf hin2vec; popd
