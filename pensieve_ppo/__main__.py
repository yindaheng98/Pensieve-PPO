"""Package command-line entry point for Pensieve PPO."""

import argparse
from typing import Callable, Optional, Sequence

from . import generate_exp_pool, imitate, imitate_exp_pool, test, train


AddArguments = Callable[[argparse.ArgumentParser], None]
RunCommand = Callable[[argparse.Namespace], None]


def add_subcommand(
    subparsers: argparse._SubParsersAction,
    name: str,
    description: str,
    add_arguments: AddArguments,
    run_command: RunCommand,
    aliases: Sequence[str] = (),
) -> None:
    """Register a package-level subcommand."""
    subparser = subparsers.add_parser(
        name,
        aliases=aliases,
        help=description,
        description=description,
    )
    add_arguments(subparser)
    subparser.set_defaults(func=run_command)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run a Pensieve PPO subcommand."""
    parser = argparse.ArgumentParser(prog='pensieve_ppo', description='Pensieve PPO command line tools')
    subparsers = parser.add_subparsers(dest='command', metavar='command', required=True)

    add_subcommand(
        subparsers,
        name='generate-exp-pool',
        aliases=('generate_exp_pool',),
        description=generate_exp_pool.DESCRIPTION,
        add_arguments=generate_exp_pool.add_arguments,
        run_command=generate_exp_pool.main,
    )
    add_subcommand(
        subparsers,
        name='imitate-exp-pool',
        aliases=('imitate_exp_pool',),
        description=imitate_exp_pool.DESCRIPTION,
        add_arguments=imitate_exp_pool.add_arguments,
        run_command=imitate_exp_pool.main,
    )
    add_subcommand(
        subparsers,
        name='imitate',
        description=imitate.DESCRIPTION,
        add_arguments=imitate.add_arguments,
        run_command=imitate.main,
    )
    add_subcommand(
        subparsers,
        name='test',
        description=test.DESCRIPTION,
        add_arguments=test.add_arguments,
        run_command=test.main,
    )
    add_subcommand(
        subparsers,
        name='train',
        description=train.DESCRIPTION,
        add_arguments=train.add_arguments,
        run_command=train.main,
    )

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
