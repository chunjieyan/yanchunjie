#!/usr/bin/env python3
"""Training-side interface to the complete 42-model source and runtime plan."""
import sys
from replay_pipeline import main
if __name__=='__main__':sys.exit(main(['--phase','training']+sys.argv[1:]))
