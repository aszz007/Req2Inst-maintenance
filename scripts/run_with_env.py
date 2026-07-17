# scripts/run_with_env.py
"""Run a command after configuring the project environment."""
import subprocess
import sys
import argparse
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

if sys.platform == 'win32':
    try:
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
    except:
        pass

ENV_MAP = {
    'text': 'qwen_text',
    'image_qwen2.5': 'qwen_vision25',
    'image_qwen3': 'qwen_vision3',
    'uml_qwen2.5': 'qwen_vision25',
    'uml_qwen3': 'qwen_vision3',
}


def run_in_env(env_name: str, script_path: str, args: list = None):
    """Run in env."""

    qwen_version = None
    if 'qwen3' in env_name or env_name == 'qwen_vision3':
        qwen_version = 'qwen3'
    elif 'qwen2.5' in env_name or 'qwen25' in env_name or env_name == 'qwen_vision25':
        qwen_version = 'qwen2.5'

    if args is None:
        args = []

    if qwen_version and '--version' not in args:
        args = args + ['--version', qwen_version]

    cmd = [
        'conda', 'run',
        '-n', env_name,
        '--no-capture-output',
        'python', script_path
    ]

    if args:
        cmd.extend(args)

    print(f"Environment: {env_name}")
    if qwen_version:
        print(f"Qwen version: {qwen_version}")
    print(f"Script: {script_path}")
    print(f"Arguments: {' '.join(args) if args else 'none'}")
    print("-" * 60)

    env = os.environ.copy()
    if qwen_version:
        env['QWEN_VISION_VERSION'] = qwen_version

    result = subprocess.run(cmd, env=env)

    return result.returncode


def main():
    """Run the command-line entry point."""
    args = sys.argv[1:]

    env_name = None
    script_path = None
    script_args = []

    i = 0
    while i < len(args):
        if args[i] == '--env' and i + 1 < len(args):
            env_key = args[i + 1]
            if env_key not in ENV_MAP:
                print(f"Error: Invalid environment type '{env_key}'")
                print(f"Available options: {', '.join(ENV_MAP.keys())}")
                sys.exit(1)
            env_name = ENV_MAP[env_key]
            i += 2
        elif args[i] == '--script' and i + 1 < len(args):
            script_path = args[i + 1]
            i += 2
            script_args = args[i:]
            break
        else:
            i += 1

    if not env_name or not script_path:
        print("Usage: python scripts/run_with_env.py --env <environment_type> --script <script_path> [script_args...]")
        print(f"\nAvailable environment types: {', '.join(ENV_MAP.keys())}")
        print("\nExample:")
        print(
            "  python scripts/run_with_env.py --env uml_qwen3 --script scripts/preprocessing/uml/recognize_uml.py --single data/test.png --streaming")
        sys.exit(1)

    exit_code = run_in_env(env_name, script_path, script_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
