#!/usr/bin/env python
"""
SHAC Demonstration Script

This script runs demonstrations of the SHAC codec functionality.
"""

import sys
import argparse
from shac.codec.examples import (
    create_example_sound_scene,
    demonstrate_interactive_navigation,
    demonstrate_streaming_processor,
    main as demo_main
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SHAC Demonstration Script')
    parser.add_argument('demo', nargs='?', choices=['scene', 'navigation', 'streaming', 'all'],
                      default='all', help='Which demo to run (default: all)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("Spherical Harmonic Audio Codec (SHAC) Demonstrations")
    print("==================================================")
    
    if args.demo == 'scene' or args.demo == 'all':
        print("\nRunning Sound Scene Demo:")
        print("--------------------------")
        create_example_sound_scene()
    
    if args.demo == 'navigation' or args.demo == 'all':
        print("\nRunning Navigation Demo:")
        print("------------------------")
        demonstrate_interactive_navigation()
    
    if args.demo == 'streaming' or args.demo == 'all':
        print("\nRunning Streaming Demo:")
        print("-----------------------")
        demonstrate_streaming_processor()
    
    if args.demo == 'all':
        print("\nAll demonstrations complete!")