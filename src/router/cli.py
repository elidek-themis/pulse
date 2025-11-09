import subprocess

from pathlib import Path

import click


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.argument("config_path", type=click.STRING, required=True, metavar="CONFIG_PATH")
@click.pass_context
def serve(ctx, config_path: str):
    """Start a vLLM server with the specified model configuration.

    \b
    All additional arguments are passed directly to 'vllm serve':
        serve model_1.yaml
        serve model_2.yaml --port 8001 --gpu-memory-utilization 0.5
        serve model_3.yaml --port 8002 --max-logprobs 128000 --enable-prefix-caching

    \b
    For all vLLM options:
        vllm serve --help
    """
    config_path = Path(config_path)

    if not config_path.exists():
        click.echo(f"Error: Model config '{config_path}' not found.", err=True)
        return 1

    cmd = ["vllm", "serve", "--config", str(config_path)]
    cmd.extend(ctx.args)  # add any extra passed arguments

    click.echo(f"Running: {' '.join(cmd)}")

    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    serve()
