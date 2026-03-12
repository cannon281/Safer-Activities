#!/usr/bin/env python3
import cv2
import numpy as np
import torch
import os
import sys
import platform

def test_display_and_blackwell_gpu():
    """
    Test that display and RTX 5070 Ti (Blackwell architecture) GPU are working properly.
    """
    print("=" * 60)
    print("SYSTEM INFORMATION:")
    print("-" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    
    # Check CUDA availability and print info
    print("\n" + "=" * 60)
    print("PYTORCH & CUDA INFORMATION:")
    print("-" * 60)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
        
        # Verify CUDA compute capability
        device_properties = torch.cuda.get_device_properties(0)
        major, minor = device_properties.major, device_properties.minor
        print(f"Compute capability: {major}.{minor}")
        
        # Check if this is a Blackwell GPU (should report 12.0)
        if major == 12:
            print("✅ Blackwell architecture GPU detected and properly supported!")
        else:
            print("⚠️ This is not a Blackwell GPU or compute capability is not correctly detected.")
        
        # Test CUDA with tensor operations
        print("\nRunning tensor operations on GPU...")
        
        # Simple tensor creation and operations
        x = torch.rand(5, 3).cuda()
        y = torch.rand(5, 3).cuda()
        z = x + y
        print("Test successful, GPU is operational!")
        
        # Slightly more complex test using a small neural network
        print("\nTesting with a small neural network...")
        try:
            model = torch.nn.Sequential(
                torch.nn.Linear(3, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            ).cuda()
            
            test_input = torch.rand(32, 3).cuda()
            output = model(test_input)
            print("Neural network test successful!")
        except Exception as e:
            print(f"Neural network test failed: {e}")
    else:
        print("❌ CUDA not available. Please check your installation.")
    
    # Test display with OpenCV
    print("\n" + "=" * 60)
    print("TESTING OPENCV DISPLAY:")
    print("-" * 60)
    window_name = 'Blackwell GPU Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Create a simple colored image
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:133, :] = [0, 0, 255]  # Red
    img[133:266, :] = [0, 255, 0]  # Green
    img[266:, :] = [255, 0, 0]  # Blue
    
    # Add text about GPU status
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        compute_capability = f"{major}.{minor}"
        status_text = f"GPU: {device_name} (SM_{major}{minor})"
        status_color = (0, 255, 0) if major == 12 else (0, 165, 255)  # Green if Blackwell, orange otherwise
    else:
        status_text = "GPU: Not detected"
        status_color = (0, 0, 255)  # Red
    
    # Add text to the image
    cv2.putText(img, 'GPU & Display Test', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img, status_text, (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(img, 'Press ESC to exit', (50, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    print("Showing test image. Press ESC to close.")
    cv2.imshow(window_name, img)
    
    # Wait for ESC key press
    while True:
        key = cv2.waitKey(100)
        if key == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_display_and_blackwell_gpu()