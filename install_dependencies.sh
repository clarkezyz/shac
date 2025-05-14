#!/bin/bash
# Script to install dependencies for SHAC

echo "Installing dependencies for SHAC - Spherical Harmonic Audio Codec"
echo ""

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "Python pip not found. Installing pip..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# Install required Python packages
echo "Installing required Python packages..."
pip install numpy matplotlib pygame sounddevice

# Check for controller support
echo "Checking for joystick support..."
if ! ls /dev/input/js* &> /dev/null; then
    echo "No joystick devices found. Make sure your Xbox controller is connected."
    echo "Installing additional packages for Xbox controller support..."
    sudo apt-get install -y joystick xboxdrv
    
    echo ""
    echo "To pair an Xbox controller via Bluetooth:"
    echo "1. Press the Xbox button to turn on the controller"
    echo "2. Press and hold the sync button on top of the controller until the Xbox button flashes"
    echo "3. Go to Settings > Bluetooth in Ubuntu and select the Xbox controller"
    echo ""
    echo "For a wired connection, simply connect your controller via USB."
fi

# Check if PortAudio is installed (required by sounddevice)
echo "Checking for PortAudio (required for audio output)..."
if ! pkg-config --exists portaudio-2.0; then
    echo "PortAudio not found. Installing..."
    sudo apt-get install -y portaudio19-dev
fi

echo ""
echo "Installation complete! You can now run:"
echo "python realtime_demo.py"
echo ""
echo "If you experience permission issues with audio devices, you may need to add your user to the audio group:"
echo "sudo usermod -a -G audio $USER"
echo "Then log out and log back in for the changes to take effect."