#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DNG Image Viewer / Reader Utility
This script reads a DNG (Digital Negative) raw image file using the rawpy library
and displays it using matplotlib and opencv-python.
"""

import os
import sys
import argparse

def check_dependencies():
    """
    Check if required libraries are installed, print installation guide if missing.
    """
    missing_libs = []
    try:
        import rawpy
    except ImportError:
        missing_libs.append("rawpy")
    try:
        import matplotlib
    except ImportError:
        missing_libs.append("matplotlib")
    try:
        import cv2
    except ImportError:
        missing_libs.append("opencv-python")
    try:
        import numpy
    except ImportError:
        missing_libs.append("numpy")

    if missing_libs:
        print("Missing required libraries. Please install them using the following command:")
        print(f"pip install {' '.join(missing_libs)}")
        sys.exit(1)

def read_and_display_dng(file_path, use_camera_wb=True, use_auto_wb=False, half_size=False, output_gamma=None):
    """
    Read DNG file and display it.
    
    Parameters:
    - file_path: str, path to the DNG file.
    - use_camera_wb: bool, whether to use camera white balance.
    - use_auto_wb: bool, whether to use automatic white balance.
    - half_size: bool, whether to output at half size (faster processing).
    - output_gamma: tuple or None, custom gamma curves (power, toe_slope).
    """
    import rawpy
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    print(f"Loading DNG file: {file_path} ...")
    try:
        with rawpy.imread(file_path) as raw:
            print("Successfully opened raw image.")
            print(f"Raw image size: {raw.sizes.height}x{raw.sizes.width}")
            print(f"Color filter array (CFA) pattern: {raw.raw_pattern.tolist()}")
            
            # Postprocess parameter configuration
            # - half_size: Reduce resolution by half (fast demosaicing)
            # - use_camera_wb: Use the white balance shot by the camera
            # - use_auto_wb: Let LibRaw calculate automatic white balance
            # - no_auto_bright: Disable automatic brightening (keep original exposure curve)
            print("Processing raw image (demosaicing)...")
            
            postprocess_opts = {
                'half_size': half_size,
                'use_camera_wb': use_camera_wb,
                'use_auto_wb': use_auto_wb,
                'no_auto_bright': True
            }
            
            if output_gamma is not None:
                postprocess_opts['output_gamma'] = output_gamma

            rgb_image = raw.postprocess(**postprocess_opts)
            print(f"Processed image shape: {rgb_image.shape}")

            # Plotting with Matplotlib (RGB format)
            print("Displaying image via matplotlib...")
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(rgb_image)
            ax.set_title(f"DNG Preview - {os.path.basename(file_path)}")
            ax.axis('off')
            plt.tight_layout()
            
            # Save a copy as preview if desired
            preview_save_path = os.path.splitext(file_path)[0] + "_preview.jpg"
            
            # Using OpenCV to display and allow window interaction
            # Note: OpenCV expects BGR color format
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # Show interactive preview using CV2
            window_name = f"DNG Viewer - Press ESC/Q to exit, S to save preview"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
            cv2.imshow(window_name, bgr_image)
            
            # Show Matplotlib plot as well in the background
            plt.show(block=False)
            
            print("\nControls for OpenCV window:")
            print(" - Press 'ESC' or 'q' to close the preview window.")
            print(" - Press 's' to save the processed image as JPEG.")
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'): # ESC or q
                    break
                elif key == ord('s'):
                    cv2.imwrite(preview_save_path, bgr_image)
                    print(f"Saved preview to: {preview_save_path}")
            
            cv2.destroyAllWindows()
            plt.close(fig)

    except Exception as e:
        print(f"An error occurred during DNG processing: {e}")

if __name__ == "__main__":
    check_dependencies()
    
    parser = argparse.ArgumentParser(description="Read and display DNG raw images.")
    parser.add_argument("file_path", type=str, help="Path to the input DNG file")
    parser.add_argument("--auto-wb", action="store_true", help="Use automatic white balance instead of camera white balance")
    parser.add_argument("--half-size", action="store_true", help="Load half size for faster loading")
    
    args = parser.parse_args()
    
    # Run reader and display
    read_and_display_dng(
        file_path=args.file_path,
        use_camera_wb=not args.auto_wb,
        use_auto_wb=args.auto_wb,
        half_size=args.half_size
    )
