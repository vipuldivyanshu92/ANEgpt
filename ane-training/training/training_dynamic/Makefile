CC = xcrun clang
CFLAGS = -O2 -framework Foundation -framework IOSurface -framework Accelerate \
         -isysroot $(shell xcrun --show-sdk-path) -fobjc-arc

train: train.m config.h io.h cpu_ops.h mil_dynamic.h
	$(CC) $(CFLAGS) -o train train.m

clean:
	rm -f train
