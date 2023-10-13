#include "serial_port.h"

int main() {
    // Change to specific serial port and baud rate
    Serial_Port serialPort("/dev/ttyUSB0", 57600);

    try {
        // Open the serial port
        serialPort.start();

        mavlink_message_t message;
        message = create_heartbeat_message();

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

// Heatbeat message initialization
mavlink_message_t create_heartbeat_message() {
    mavlink_message_t message;
    mavlink_heartbeat_t heartbeat;
    memset(&heartbeat, 0, sizeof(heartbeat));

    // Set the values for the heartbeat message
    heartbeat.type = MAV_TYPE_GCS; // Type of the system sending the message
    heartbeat.autopilot = MAV_AUTOPILOT_GENERIC; // Autopilot type
    heartbeat.base_mode = MAV_MODE_GUIDED_ARMED; // System mode
    heartbeat.custom_mode = 0; // Custom mode (if applicable)
    heartbeat.system_status = MAV_STATE_ACTIVE; // System status

    // Pack the heartbeat message into the MAVLink message
    mavlink_msg_heartbeat_encode(1, 200, &message, &heartbeat);

    return message;
}