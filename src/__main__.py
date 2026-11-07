import sys
from fire import Fire
from fire.core import FireError
from src.cli import CLI


if __name__ == "__main__":
    try:
        Fire(CLI)
    except TypeError as error:
        if "not supported between instances of" in str(error):
            print(
                "Error: Invalid command arguments. "
                "Use '--option value' for named arguments.",
                file=sys.stderr
            )
        else:
            print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    except (FireError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
