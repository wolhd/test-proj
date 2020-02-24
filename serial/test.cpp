#include <stdio.h>
#include <string.h>

// Linux headers
#include <fcntl.h> // Contains file controls like O_RDWR
#include <errno.h> // Error integer and strerror() function
#include <termios.h> // Contains POSIX terminal control definitions
#include <unistd.h> // write(), read(), close()

// $ sudo adduser $USER dialout
void test() {
    int serial_port = open("/dev/ttyACM0", O_RDONLY);

    if (serial_port < 0) {
        printf("Error %i from open: %s\n", errno, strerror(errno));
        return;
    }
    char data[128];
    if (read(serial_port, data, 128) < 0) {
        printf("Error reading\n");
    }
    else {
        printf("read data: %s\n", data);
    }

	struct termios tty;
	memset(&tty, 0, sizeof tty);

	// Read in existing settings, and handle any error
	if(tcgetattr(serial_port, &tty) != 0) {
		printf("Error %i from tcgetattr: %s\n", errno, strerror(errno));
		return;
	}
}

int main(int argc, char* argv[]) {
    test();
}

