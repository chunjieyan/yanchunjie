#!/usr/bin/env python3
"""Evaluation-side interface to all 54 configurations and the saved-result contract."""
import sys
from replay_pipeline import main
if __name__=='__main__':sys.exit(main(['--phase','evaluation']+sys.argv[1:]))
