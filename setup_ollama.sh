#!/usr/bin/env bash
# 
# Ollama Setup Script for PDF Data Extractor
# Run this script to install and configure Ollama on macOS
#
set -e

echo "🚀 Ollama Setup for PDF Data Extractor"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS only"
    exit 1
fi

print_status "Step 1: Checking if Ollama is already installed..."

# Check if Ollama app exists
if [ -d "/Applications/Ollama.app" ]; then
    print_success "Ollama app found in Applications"
else
    print_status "Downloading Ollama for macOS..."
    
    # Download Ollama
    curl -L https://ollama.com/download/Ollama-darwin.zip -o /tmp/Ollama-darwin.zip
    
    # Extract and install
    cd /tmp
    unzip -q Ollama-darwin.zip
    sudo mv Ollama.app /Applications/
    
    print_success "Ollama installed to /Applications/Ollama.app"
fi

print_status "Step 2: Setting up ollama command..."

# Create symlink for ollama command
if [ -f "/Applications/Ollama.app/Contents/Resources/ollama" ]; then
    sudo ln -sf /Applications/Ollama.app/Contents/Resources/ollama /usr/local/bin/ollama
    print_success "ollama command available in PATH"
else
    print_error "Ollama binary not found in expected location"
    exit 1
fi

print_status "Step 3: Starting Ollama server..."

# Start Ollama server in background
nohup ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

# Wait for server to start
sleep 5

# Check if server is running
if curl -s http://localhost:11434/api/tags > /dev/null; then
    print_success "Ollama server is running (PID: $OLLAMA_PID)"
else
    print_error "Failed to start Ollama server"
    exit 1
fi

print_status "Step 4: Installing AI model..."

# Install tinyllama (small and fast)
print_status "Installing tinyllama model (recommended for fast extraction)..."
ollama pull tinyllama

# Check if model is installed
if ollama list | grep -q "tinyllama"; then
    print_success "tinyllama model installed successfully"
else
    print_error "Failed to install tinyllama model"
    exit 1
fi

print_status "Step 5: Testing LLM integration..."

# Test model
TEST_RESPONSE=$(ollama run tinyllama "Hello, can you extract data?" | head -n 1)
if [ -n "$TEST_RESPONSE" ]; then
    print_success "LLM is responding correctly"
else
    print_warning "LLM test had issues, but installation completed"
fi

echo ""
echo "🎉 Ollama Setup Complete!"
echo "========================"
echo ""
echo "✅ Ollama installed and running"
echo "✅ tinyllama model ready for use"  
echo "✅ Server running on http://localhost:11434"
echo ""
echo "Next steps:"
echo "1. Run: python mainApp.py \"input/Data Input.pdf\""
echo "2. Your PDF extractor will now use intelligent LLM extraction!"
echo ""
echo "Optional: Install larger models for better accuracy:"
echo "  ollama pull llama3.2:1b    # Better accuracy, slower"
echo "  ollama pull llama3.2:3b    # Best accuracy, requires more RAM"
echo ""
print_warning "Keep the terminal open or run 'ollama serve' to maintain the server"