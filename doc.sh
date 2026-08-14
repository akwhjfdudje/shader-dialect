#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Verify if doxygen is available in PATH
if ! command -v doxygen &> /dev/null; then
    echo "Error: doxygen command not found. Please install Doxygen (and make sure it is in your PATH) and try again."
    exit 1
fi

# Run doxygen
doxygen Doxyfile

echo "Doxygen documentation generated successfully."
echo "You can open the documentation at: docs/doxygen/html/index.html"
