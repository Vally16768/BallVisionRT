"""
main.py

Entry point for the DotLumen 2D/3D ball tracking and top-view mapping pipeline.
The heavy lifting is implemented in the DotLumenPipeline class
inside pipeline/pipeline.py.
"""

from pipeline import DotLumenPipeline


def main():
    pipeline = DotLumenPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
