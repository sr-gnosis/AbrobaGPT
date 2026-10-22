import os
import sys
from pathlib import Path

# Add current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

def main():
    try:
        import tui
        tui.main()
    except ImportError as e:
        print(f"Error: tui.py not found or import failed. {e}")
    except Exception as e:
        print(f"Error launching TUI: {e}")

if __name__ == "__main__":
    main()
