import numpy as np
import typer
from rich.console import Console

from boiling_learning.model.layers import ImageNormalization

app = typer.Typer()
console = Console()


@app.command()
def main() -> None:
    image = np.random.randint(0, 255, (3, 3, 1), dtype=np.uint8)

    image_standardizer = ImageNormalization()
    standardized_image = image_standardizer(image).numpy()

    console.print('Original:')
    console.print(f'Mean={image.mean()}; Std={image.std()}')
    console.print(image)

    console.print('Standardized:')
    console.print(f'Mean={standardized_image.mean()}; Std={standardized_image.std()}')
    console.print(standardized_image)
