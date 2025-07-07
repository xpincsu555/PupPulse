#!/usr/bin/env bash

# PupPulse Data Preparation Script
# Extracts the Dog_E.zip dataset into the correct directory structure

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if zip file path is provided
if [ $# -eq 0 ]; then
    print_error "No zip file path provided!"
    echo "Usage: $0 <path_to_Dog_E.zip>"
    echo "Example: $0 Dog_E.zip"
    exit 1
fi

ZIP_FILE="$1"
DATA_DIR="data/raw/images"

# Check if zip file exists
if [ ! -f "$ZIP_FILE" ]; then
    print_error "Zip file not found: $ZIP_FILE"
    exit 1
fi

print_status "Starting PupPulse dataset preparation..."

# Create data directory structure
print_status "Creating directory structure..."
mkdir -p "$DATA_DIR"

# Extract zip file
print_status "Extracting $ZIP_FILE..."
if command -v unzip >/dev/null 2>&1; then
    unzip -q "$ZIP_FILE" -d "$DATA_DIR"
else
    print_error "unzip command not found! Please install unzip."
    exit 1
fi

# Check if extraction was successful and expected directories exist
EXPECTED_DIRS=("angry" "happy" "relaxed" "sad")
MISSING_DIRS=()

for dir in "${EXPECTED_DIRS[@]}"; do
    if [ ! -d "$DATA_DIR/$dir" ]; then
        MISSING_DIRS+=("$dir")
    fi
done

if [ ${#MISSING_DIRS[@]} -gt 0 ]; then
    print_warning "Some expected directories are missing: ${MISSING_DIRS[*]}"
    print_status "Checking if images directory exists within extracted content..."
    
    # Sometimes the zip contains an extra 'images' folder
    if [ -d "$DATA_DIR/images" ]; then
        print_status "Found nested images directory, moving contents up one level..."
        mv "$DATA_DIR/images"/* "$DATA_DIR/"
        rmdir "$DATA_DIR/images"
    fi
fi

# Verify the final structure and count images
print_status "Verifying dataset structure..."
TOTAL_IMAGES=0

for dir in "${EXPECTED_DIRS[@]}"; do
    if [ -d "$DATA_DIR/$dir" ]; then
        IMAGE_COUNT=$(find "$DATA_DIR/$dir" -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l)
        print_status "  $dir: $IMAGE_COUNT images"
        TOTAL_IMAGES=$((TOTAL_IMAGES + IMAGE_COUNT))
    else
        print_error "  $dir: Directory not found!"
    fi
done

if [ $TOTAL_IMAGES -eq 0 ]; then
    print_error "No images found! Please check the zip file structure."
    exit 1
fi

print_status "Dataset preparation completed successfully!"
print_status "Total images extracted: $TOTAL_IMAGES"
print_status "Dataset location: $DATA_DIR"

# Display directory tree if tree command is available
if command -v tree >/dev/null 2>&1; then
    print_status "Directory structure:"
    tree "$DATA_DIR" -L 2
else
    print_status "Directory structure:"
    ls -la "$DATA_DIR"
fi

print_status "You can now run training with:"
echo "python src/models/train.py --config configs/emotion_classification.yaml"
