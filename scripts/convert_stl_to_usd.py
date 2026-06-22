#!/usr/bin/env python3
"""Convert STL to USD using Isaac Sim's Asset Converter (headless)."""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input STL file")
parser.add_argument("--output", required=True, help="Output USD file")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.asset_converter

converter = omni.kit.asset_converter.AssetConverter()
task = converter.create_converter_task(args_cli.input, args_cli.output, None)

while not task.done():
    import time
    time.sleep(0.1)

if task.status() == omni.kit.asset_converter.AssetConverterStatus.SUCCESS:
    print(f"Done: {args_cli.output}")
else:
    print(f"Error: {task.status()}", file=sys.stderr)

simulation_app.close()
