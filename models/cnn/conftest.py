import sys
from pathlib import Path

# this file is to sys.path so pytest can find
# mnist_loader, experiment_logger, cnn_shallow, etc. without package prefixes.
sys.path.insert(0, str(Path(__file__).parent))
