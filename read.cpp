#include "serial_port.h"

int main() {
    Serial_Port serialPort("/dev/ttyTHS1", 57600);

    try {
        serialPort.start();

        while (true) {
            mavlink_message_t message;

            if (serialPort.read_message(message)) {
                // Handle the received MAVLink message heartbeat message for testing case
                switch (message.msgid) {
                    case MAVLINK_MSG_ID_HEARTBEAT:
                        mavlink_heartbeat_t heartbeat;
                        mavlink_msg_heartbeat_decode(&message, &heartbeat);
                        printf("Received Heartbeat from system type: %d\n", heartbeat.type);
                        break;

                    default:
                        // Handle images
			printf("Default case\n");
                        break;
                }

            } else {
		    printf("didn't get message\n");
	    }
           sleep(1);
        }
    } catch (int error) {
        printf("Error: %d\n", error);
    }

    serialPort.stop();

    return 0;
}
