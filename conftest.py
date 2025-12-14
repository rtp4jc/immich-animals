import sys
import os
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.absolute()

# Add project root to Python path
sys.path.insert(0, str(project_root))
