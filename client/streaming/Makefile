CC=gcc
FLAGS=`pkg-config --cflags --libs gstreamer-1.0`

SOURCES=main.c
TARGETS=$(foreach n, $(SOURCES), $(basename $(n)))

all: ${TARGETS}

.PHONY: all clean

${TARGETS}: %:%.c
	$(CC) -Wall $< -o $@ $(FLAGS)

clean:
	rm -f ${TARGETS}
