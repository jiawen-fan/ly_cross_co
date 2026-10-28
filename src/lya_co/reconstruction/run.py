"""CLI entry point; importing this module never launches reconstruction.

Run python -m lya_co.reconstruction.run --help for the ASTRID comparison.
The solver remains the external Dachshund executable.
"""
from lya_co.pipeline import main

if __name__ == "__main__":
    main()
