import os
import sys


def train():
    import .run_glue
    run_glue.main()


def main():
    train()


if __name__ == "__main__":
    main()
