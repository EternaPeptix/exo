#!/usr/bin/env python3
"""Patch api/main.py to include ExpertParallel in sharding combinations."""
import sys

filepath = sys.argv[1]
with open(filepath) as f:
    content = f.read()

# Add ExpertParallel to the sharding iteration loop
old = '        for sharding in (Sharding.Pipeline, Sharding.Tensor):'
new = '        for sharding in (Sharding.Pipeline, Sharding.Tensor, Sharding.ExpertParallel):'

if old in content:
    content = content.replace(old, new, 1)
    print("  Added ExpertParallel to sharding combinations")
else:
    print("  WARNING: Could not find sharding iteration loop")

with open(filepath, 'w') as f:
    f.write(content)
print(f"Patched {filepath} successfully")
