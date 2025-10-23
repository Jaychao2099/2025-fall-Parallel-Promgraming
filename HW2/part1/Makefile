CC = gcc
# CC = clang
CFLAGS = -Wall -pthread -mavx2 -O3 #-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -ftree-vectorize -fverbose-asm

OBJ = pi.c
TARGET = pi.out

$(TARGET): $(OBJ)
	$(CC) -o $@ $(OBJ) $(CFLAGS)

# %.o: %.c
# 	$(CC) $(CFLAGS) -c $<

clean:
# 	rm -f $(OBJ) $(TARGET)
	rm -f $(TARGET)

