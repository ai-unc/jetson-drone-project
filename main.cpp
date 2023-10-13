#include "serial_port.h"

int main() {
    // Create an instance of the Serial_Port class
    Serial_Port serialPort("/dev/ttyUSB0", 57600);

    try {
        // Open the serial port
        serialPort.start();

        mavlink_message_t message;
        

        // Send the message
        int bytesWritten = serialPort.write_message(message);

        if (bytesWritten > 0) {
            printf("Sent %d bytes\n", bytesWritten);
        } else {
            printf("Failed to send the message\n");
        }
    } catch (int error) {
        printf("Error: %d\n", error);
    }

    // Close the serial port
    serialPort.stop();

    return 0;
}