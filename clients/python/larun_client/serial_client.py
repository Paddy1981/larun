"""
Serial Communication Backend for LARUN Devices

Implements the binary protocol with COBS framing and CRC16 checksums
for communicating with LARUN TinyML microcontroller devices.
"""

import struct
import time
from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    import serial
except ImportError:
    raise ImportError(
        "pyserial is required for serial communication. "
        "Install with: pip install pyserial"
    )


class SerialClient:
    """
    Low-level serial communication client for LARUN devices

    Implements the binary protocol:
    [0x00][Version][Type][Length][Payload][CRC16][0x00]

    Message types:
    - 0x01: DATA_LIGHTCURVE
    - 0x10: CMD_INFER
    - 0x11: CMD_SELECT_MODEL
    - 0x12: CMD_GET_INFO
    - 0x20: RESULT_SUCCESS
    - 0x21: RESULT_ERROR
    - 0xF0: PING / 0xF1: PONG
    """

    # Message types
    DATA_LIGHTCURVE = 0x01
    DATA_SPECTRUM = 0x02
    DATA_IMAGE = 0x03
    DATA_FITS = 0x04
    DATA_CSV = 0x05

    CMD_INFER = 0x10
    CMD_SELECT_MODEL = 0x11
    CMD_GET_INFO = 0x12
    CMD_GET_MODELS = 0x13
    CMD_RESET = 0x14

    RESULT_SUCCESS = 0x20
    RESULT_ERROR = 0x21
    INFO_DEVICE = 0x22
    INFO_MODELS = 0x23

    PING = 0xF0
    PONG = 0xF1
    ACK = 0xF2
    NACK = 0xF3

    FRAME_DELIMITER = 0x00
    PROTOCOL_VERSION = 0x01

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 5.0):
        """
        Initialize serial client

        Args:
            port: Serial port (e.g., '/dev/ttyUSB0', 'COM3')
            baudrate: Baud rate (default: 115200)
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None

    def connect(self) -> bool:
        """
        Open serial connection

        Returns:
            True on success

        Raises:
            serial.SerialException: If connection fails
        """
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            # Wait for device to initialize
            time.sleep(0.5)
            # Flush any pending data
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            return True
        except serial.SerialException as e:
            raise ConnectionError(f"Failed to connect to {self.port}: {e}")

    def close(self):
        """Close serial connection"""
        if self.serial and self.serial.is_open:
            self.serial.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ========================================================================
    # Low-Level Protocol
    # ========================================================================

    @staticmethod
    def calculate_crc16(data: bytes) -> int:
        """
        Calculate CRC16-CCITT checksum

        Args:
            data: Data bytes

        Returns:
            CRC16 checksum
        """
        crc = 0xFFFF
        polynomial = 0x1021

        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ polynomial
                else:
                    crc = crc << 1
                crc &= 0xFFFF

        return crc

    @staticmethod
    def cobs_encode(data: bytes) -> bytes:
        """
        COBS (Consistent Overhead Byte Stuffing) encoding
        Removes 0x00 bytes from data stream

        Args:
            data: Input data

        Returns:
            COBS-encoded data
        """
        if len(data) == 0:
            return b''

        output = bytearray()
        code_index = 0
        code = 1
        output.append(0)  # Placeholder for first code

        for i, byte in enumerate(data):
            if byte == 0:
                output[code_index] = code
                code = 1
                code_index = len(output)
                output.append(0)
            else:
                output.append(byte)
                code += 1
                if code == 0xFF:
                    output[code_index] = code
                    code = 1
                    code_index = len(output)
                    output.append(0)

        output[code_index] = code
        return bytes(output)

    @staticmethod
    def cobs_decode(data: bytes) -> bytes:
        """
        COBS decoding

        Args:
            data: COBS-encoded data

        Returns:
            Decoded data
        """
        if len(data) == 0:
            return b''

        output = bytearray()
        i = 0

        while i < len(data):
            code = data[i]
            i += 1

            for j in range(1, code):
                if i >= len(data):
                    break
                output.append(data[i])
                i += 1

            if code < 0xFF and i < len(data):
                output.append(0)

        return bytes(output)

    def encode_message(self, msg_type: int, payload: bytes = b'') -> bytes:
        """
        Encode message with COBS framing and CRC16

        Args:
            msg_type: Message type
            payload: Message payload

        Returns:
            Encoded message ready to send
        """
        # Build message: [Version][Type][Length][Payload]
        length = len(payload)
        header = struct.pack('<BBH', self.PROTOCOL_VERSION, msg_type, length)
        message = header + payload

        # Calculate CRC
        crc = self.calculate_crc16(message)
        message += struct.pack('>H', crc)  # Big-endian CRC

        # COBS encode
        encoded = self.cobs_encode(message)

        # Add frame delimiters
        return bytes([self.FRAME_DELIMITER]) + encoded + bytes([self.FRAME_DELIMITER])

    def decode_message(self, data: bytes) -> Tuple[int, bytes]:
        """
        Decode received message

        Args:
            data: Received data with frame delimiters

        Returns:
            (msg_type, payload) tuple

        Raises:
            ValueError: If message is invalid
        """
        # Check frame delimiters
        if len(data) < 3:
            raise ValueError("Message too short")
        if data[0] != self.FRAME_DELIMITER or data[-1] != self.FRAME_DELIMITER:
            raise ValueError("Invalid frame delimiters")

        # COBS decode
        decoded = self.cobs_decode(data[1:-1])

        if len(decoded) < 6:  # Header (4) + CRC (2)
            raise ValueError("Decoded message too short")

        # Extract CRC
        received_crc = struct.unpack('>H', decoded[-2:])[0]
        message = decoded[:-2]

        # Verify CRC
        calculated_crc = self.calculate_crc16(message)
        if received_crc != calculated_crc:
            raise ValueError(f"CRC mismatch: expected {calculated_crc:04X}, got {received_crc:04X}")

        # Parse header
        version, msg_type, length = struct.unpack('<BBH', message[:4])

        if version != self.PROTOCOL_VERSION:
            raise ValueError(f"Protocol version mismatch: {version}")

        # Extract payload
        payload = message[4:4+length]

        return msg_type, payload

    def send_message(self, msg_type: int, payload: bytes = b''):
        """Send message to device"""
        encoded = self.encode_message(msg_type, payload)
        self.serial.write(encoded)
        self.serial.flush()

    def receive_message(self, timeout: Optional[float] = None) -> Tuple[int, bytes]:
        """
        Receive message from device

        Args:
            timeout: Read timeout (uses default if None)

        Returns:
            (msg_type, payload) tuple
        """
        if timeout is not None:
            old_timeout = self.serial.timeout
            self.serial.timeout = timeout

        try:
            # Read until first frame delimiter
            while True:
                byte = self.serial.read(1)
                if len(byte) == 0:
                    raise TimeoutError("No response from device")
                if byte[0] == self.FRAME_DELIMITER:
                    break

            # Read message body until next delimiter
            message = bytes([self.FRAME_DELIMITER])
            while True:
                byte = self.serial.read(1)
                if len(byte) == 0:
                    raise TimeoutError("Incomplete message")
                message += byte
                if byte[0] == self.FRAME_DELIMITER:
                    break

            return self.decode_message(message)

        finally:
            if timeout is not None:
                self.serial.timeout = old_timeout

    # ========================================================================
    # High-Level API
    # ========================================================================

    def ping(self, timeout: float = 1.0) -> bool:
        """
        Ping device

        Args:
            timeout: Ping timeout

        Returns:
            True if device responds
        """
        try:
            self.send_message(self.PING)
            msg_type, _ = self.receive_message(timeout=timeout)
            return msg_type == self.PONG
        except (TimeoutError, ValueError):
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        Get device information

        Returns:
            Device info dictionary
        """
        self.send_message(self.CMD_GET_INFO)
        msg_type, payload = self.receive_message()

        if msg_type != self.INFO_DEVICE:
            raise RuntimeError(f"Unexpected response: {msg_type:02X}")

        # Parse JSON payload
        import json
        return json.loads(payload.decode('utf-8'))

    def select_model(self, model_id: int) -> bool:
        """
        Select active model

        Args:
            model_id: Model ID (0x01-0x08)

        Returns:
            True on success
        """
        payload = struct.pack('<Bxxx', model_id)
        self.send_message(self.CMD_SELECT_MODEL, payload)

        msg_type, _ = self.receive_message()
        return msg_type == self.ACK

    def upload_lightcurve(self, flux: np.ndarray) -> bool:
        """
        Upload light curve data

        Args:
            flux: Flux array (1D, float32)

        Returns:
            True on success
        """
        if flux.dtype != np.float32:
            flux = flux.astype(np.float32)

        # Build payload: [num_points][data_format][reserved][data...]
        num_points = len(flux)
        header = struct.pack('<HBB', num_points, 0, 0)  # format=0 (float32)
        payload = header + flux.tobytes()

        self.send_message(self.DATA_LIGHTCURVE, payload)

        msg_type, _ = self.receive_message()
        return msg_type == self.ACK

    def infer(self) -> Dict[str, Any]:
        """
        Run inference on uploaded data

        Returns:
            Inference result dictionary
        """
        self.send_message(self.CMD_INFER)
        msg_type, payload = self.receive_message(timeout=10.0)  # Longer timeout

        if msg_type == self.RESULT_ERROR:
            error_msg = payload.decode('utf-8')
            raise RuntimeError(f"Inference failed: {error_msg}")

        if msg_type != self.RESULT_SUCCESS:
            raise RuntimeError(f"Unexpected response: {msg_type:02X}")

        # Parse JSON result
        import json
        return json.loads(payload.decode('utf-8'))

    def reset(self):
        """Reset device"""
        self.send_message(self.CMD_RESET)
        time.sleep(2.0)  # Wait for reboot
