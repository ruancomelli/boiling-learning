from __future__ import annotations

import itertools
import textwrap
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TypeVar

import more_itertools as mit
import tensorflow as tf
import typer
from autokeras.keras_layers import CastToFloat32

from boiling_learning.app.configuration import configure
from boiling_learning.app.constants import figures_path
from boiling_learning.app.displaying.latex import NEW_LINE_TOKEN
from boiling_learning.io.dataclasses import dataclass
from boiling_learning.model.definitions import hoboldnet2, kramernet
from boiling_learning.model.layers import ImageNormalization
from boiling_learning.model.model import ModelArchitecture

_T = TypeVar("_T")

SKIPPED_LAYERS = (
    tf.keras.layers.Flatten,
    tf.keras.layers.Activation,
    CastToFloat32,
)

app = typer.Typer()


@app.command()
def main() -> None:
    configure(
        force_gpu_allow_growth=True,
        use_xla=True,
        require_gpu=True,
    )

    for model_name, model in (
        (
            "hoboldnet",
            hoboldnet2(input_shape=(120, 196), dropout=0.5, normalize_images=False),
        ),
        (
            "kramernet",
            kramernet(input_shape=(128, 96), dropout=0.5, normalize_images=False),
        ),
    ):
        for max_rows_per_column in (None, 7):
            filename = (
                f"{model_name}_{max_rows_per_column}_rows.text"
                if max_rows_per_column is not None
                else f"{model_name}_single_column.text"
            )
            Path(diagrams_path() / filename).write_text(
                "\n".join(
                    model_to_tikz(
                        model,
                        max_rows_per_column=max_rows_per_column,
                        standalone=True,
                    )
                )
            )


def model_to_tikz(  # noqa: PLR0915
    architecture: ModelArchitecture,
    /,
    *,
    max_rows_per_column: int | None,
    standalone: bool,
) -> Iterator[str]:
    preamble = (
        textwrap.dedent(
            """
                \\documentclass[tikz]{standalone}
                \\usetikzlibrary{shapes}
                \\usetikzlibrary{shapes.multipart}
                \\usetikzlibrary{fit}
                \\usetikzlibrary{arrows.meta}
                \\usetikzlibrary{matrix, positioning}
                \\usetikzlibrary{decorations.pathreplacing}
                \\usetikzlibrary{fadings}

                \\usepackage{siunitx}
                \\usepackage{caption}
                \\usepackage{subcaption}
                \\usepackage{multirow}

                \\colorlet{mlred}{red!80!black}
                \\colorlet{mlblue}{blue!80!black}
                \\colorlet{mlgreen}{green!60!black}
                \\colorlet{mlorange}{orange!70!red!60!black}
                \\colorlet{mldarkred}{red!30!black}
                \\colorlet{mldarkblue}{blue!40!black}
                \\colorlet{mldarkgreen}{green!30!black}
                \\colorlet{mldarkyellow}{yellow!30!black}
                \\tikzstyle{node}=[thick,circle,draw=mldarkyellow,minimum size=30,inner sep=0.5,outer sep=0.6,fill=yellow!20]
                \\tikzstyle{node in}=[node,green!20!black,draw=mlgreen!30!black,fill=mlgreen!25]
                \\tikzstyle{node hidden}=[node,blue!20!black,draw=mlblue!30!black,fill=mlblue!20]
                \\tikzstyle{node convol}=[node,orange!20!black,draw=mlorange!30!black,fill=mlorange!20]
                \\tikzstyle{node out}=[node,red!20!black,draw=mlred!30!black,fill=mlred!20]
                \\tikzstyle{connect}=[mldarkblue] %,line cap=round
                \\tikzstyle{connect arrow}=[-{Latex[length=4,width=3.5]},thick,mldarkblue,shorten <=0.5,shorten >=1]
                \\tikzset{
                    input node/.style={node in},
                    hidden node/.style={node hidden},
                    output node/.style={node out},
                }

                \\begin{document}

                \\begin{figure}[ht]
                    \\begin{tikzpicture}
            """
        )
        if standalone
        else "\\begin{tikzpicture}"
    )
    ending = (
        textwrap.dedent(
            """
                    \\end{tikzpicture}
                \\end{figure}
                \\end{document}
            """
        )
        if standalone
        else "\\end{tikzpicture}"
    )
    yield from preamble.splitlines()

    layers = {
        (column_index, row_index): layer
        for column_index, column in enumerate(
            _allocate_layers(
                tuple(_group_layers(architecture.model.layers)),
                max_rows_per_column=max_rows_per_column,
            )
        )
        for row_index, layer in enumerate(column)
    }

    for (column_index, row_index), layer in layers.items():
        match layer:
            case tf.keras.layers.InputLayer():
                color = "gray!20!white"
                layer_name = "Input layer"
                notes = ()
            case NormalizedInputBlock():
                color = "gray!20!white"
                layer_name = "Input layer"
                notes = ("Standardized",)
            case tf.keras.layers.Conv2D():
                color = "blue!20!white"
                layer_name = "Convolutional layer"

                kernel_size = "\\num{" + "x".join(map(str, layer.kernel_size)) + "}"
                strides_size = "\\num{" + "x".join(map(str, layer.strides)) + "}"

                notes = (
                    ", ".join(
                        (
                            f"\\num{{{layer.filters}}} filters",
                            f"kernel size: {kernel_size}",
                            f"stride: {strides_size}",
                        )
                    ),
                )
            case tf.keras.layers.ReLU():
                color = "green!20!white"
                layer_name = "Activation layer"
                notes = ("ReLU",)
            case tf.keras.layers.MaxPool2D():
                color = "red!20!white"
                layer_name = "Pooling layer"

                size = "\\num{" + "x".join(map(str, layer.pool_size)) + "}"
                strides_size = "\\num{" + "x".join(map(str, layer.strides)) + "}"

                notes = (
                    ", ".join(
                        ("Max-pooling", f"size: {size}", f"stride: {strides_size}")
                    ),
                )
            case tf.keras.layers.Dropout():
                color = "gray!20!white"
                layer_name = "Regularization layer"
                notes = (f"Dropout: \\SI{{{layer.rate * 100:.0f}}}{{\\percent}}",)
            case tf.keras.layers.Dense():
                color = "yellow!20!white"
                layer_name = "Dense layer"
                notes = (f"{layer.units} units" if layer.units > 1 else "1 unit",)
            case ConvolutionalBlock(
                name=layer_name,
                convolutional_layer=conv_layer,
                activation_layer=tf.keras.layers.ReLU(),
                pooling_layer=pooling_layer,
            ):
                color = "blue!20!white"
                layer_name = "Convolutional block"

                conv_layer_kernel_size = (
                    "\\num{" + "x".join(map(str, conv_layer.kernel_size)) + "}"
                )
                conv_layer_strides_size = (
                    "\\num{" + "x".join(map(str, conv_layer.strides)) + "}"
                )

                maxpooling_size = (
                    "\\num{" + "x".join(map(str, pooling_layer.pool_size)) + "}"
                )
                maxpooling_strides_size = (
                    "\\num{" + "x".join(map(str, pooling_layer.strides)) + "}"
                )

                notes = (
                    ", ".join(
                        (
                            f"\\num{{{conv_layer.filters}}} filters",
                            f"kernel size: {conv_layer_kernel_size}",
                            f"stride: {conv_layer_strides_size}",
                        )
                    ),
                    ", ".join(
                        (
                            "Max-pooling",
                            f"size: {maxpooling_size}",
                            f"stride: {maxpooling_strides_size}",
                        )
                    ),
                    "Activation: ReLU",
                )
            case DenseBlock(
                name=layer_name,
                dense_layer=dense_layer,
                activation_layer=tf.keras.layers.ReLU(),
            ):
                color = "yellow!20!white"
                layer_name = "Dense block"

                notes = (
                    f"{dense_layer.units} units" if dense_layer.units > 1 else "1 unit",
                    "Activation: ReLU",
                )
            case _ if isinstance(layer, SKIPPED_LAYERS):
                continue
            case _:
                raise TypeError(f"Unknown layer type: {type(layer)}")

        emph_name = f"\\textbf{{{layer_name}}}"

        shape = f"Shape: \\({tuple(_clean_shape(layer.output_shape))}\\)"
        trainable_weights_count = sum(
            tf.keras.backend.count_params(p) for p in layer.trainable_weights
        )
        trainable_weights = f"Weights: {trainable_weights_count}"

        match column_index, row_index:
            case 0, 0:
                location_text = None
            case _, 0:
                previous_layer = layers[(column_index - 1, 0)]
                location_text = f"right=0.3 of {previous_layer.name}"
            case _, _:
                previous_layer = layers[(column_index, row_index - 1)]
                location_text = f"below=0.3 of {previous_layer.name}"

        node_options = ", ".join(
            item
            for item in (
                "rectangle",
                "thick",
                location_text,
                f"fill={color}",
                "minimum width=8cm",
            )
            if item is not None
        )

        yield f"\\node[{node_options}] ({layer.name}) {{%"
        yield "\\begin{tabular}{cc}%"

        yield f"\\multicolumn{{2}}{{c}}{{{emph_name}}} {NEW_LINE_TOKEN}"
        yield f"{shape} & {trainable_weights} {NEW_LINE_TOKEN}"

        for note in notes:
            yield f"\\multicolumn{{2}}{{c}}{{\\textit{{{note}}}}}" + NEW_LINE_TOKEN

        yield "\\end{tabular}%"
        yield "};"  # finalize node

        match column_index, row_index:
            case 0, 0:
                pass
            case _, 0:
                # get last element from the previous column
                _, previous_layer = max(
                    (
                        (row, layer)
                        for (col, row), layer in layers.items()
                        if col == column_index - 1
                    ),
                    key=lambda row_layer_pair: row_layer_pair[0],
                )

                yield f"\\node[below=0.1 of {previous_layer.name}] ({previous_layer.name}-below) {{}};"
                yield f"\\node[right=0 of {previous_layer.name}] ({previous_layer.name}-between) {{}};"
                yield f"\\node[above=0.1 of {layer.name}] ({layer.name}-above) {{}};"

                yield (
                    "\\draw[->]"
                    f" ({previous_layer.name})"
                    f" -- ({previous_layer.name}-below.center)"
                    f" -- ({previous_layer.name}-between.center |- 10, 10 |- {previous_layer.name}-below.center)"
                    f" -- ({previous_layer.name}-between.center |- 10, 10 |- {layer.name}-above.center)"
                    f" -- ({layer.name}-above.center)"
                    f" -- ({layer.name});"
                )
            case _, _:
                previous_layer = layers[(column_index, row_index - 1)]
                yield f"\\draw[->] ({previous_layer.name}) -- ({layer.name});"

    yield from ending.splitlines()


def _clean_shape(shape: tuple) -> Iterator[int]:
    for item in shape:
        match item:
            case None:
                continue
            case int():
                yield item
            case tuple():
                yield from _clean_shape(item)
            case _:
                raise ValueError(f"unsupported nested value: {item}")


def _allocate_layers(
    layers: Sequence[_T], /, *, max_rows_per_column: int | None
) -> Iterator[Sequence[_T]]:
    """Allocate layers to columns."""
    if max_rows_per_column is None:
        # allocate all layers to a single column
        yield layers
        return

    if max_rows_per_column <= 0:
        raise ValueError("max_rows_per_column must be positive")

    layer_placeholders = "_" * len(layers)
    chunked_placeholders = mit.chunked_even(layer_placeholders, max_rows_per_column)
    layers_per_column = reversed(tuple(len(chunk) for chunk in chunked_placeholders))

    for column_size in layers_per_column:
        yield layers[:column_size]
        layers = layers[column_size:]


convolutional_names = (f"convolutional_block_{index}" for index in itertools.count())
dense_names = (f"dense_block_{index}" for index in itertools.count())
input_names = (f"input_block_{index}" for index in itertools.count())


@dataclass
class ConvolutionalBlock:
    name: str
    convolutional_layer: tf.keras.layers.Conv2D
    activation_layer: tf.keras.layers.ReLU
    pooling_layer: tf.keras.layers.MaxPool2D

    @classmethod
    def new(
        cls,
        convolutional_layer: tf.keras.layers.Conv2D,
        activation_layer: tf.keras.layers.ReLU,
        pooling_layer: tf.keras.layers.MaxPool2D,
    ) -> ConvolutionalBlock:
        return ConvolutionalBlock(
            name=next(convolutional_names),
            convolutional_layer=convolutional_layer,
            activation_layer=activation_layer,
            pooling_layer=pooling_layer,
        )

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.convolutional_layer.input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.pooling_layer.output_shape

    @property
    def trainable_weights(self) -> list[tf.Variable]:
        return self.convolutional_layer.trainable_weights


@dataclass
class DenseBlock:
    name: str
    dense_layer: tf.keras.layers.Dense
    activation_layer: tf.keras.layers.ReLU

    @classmethod
    def new(
        cls,
        dense_layer: tf.keras.layers.Dense,
        activation_layer: tf.keras.layers.ReLU,
    ) -> DenseBlock:
        return DenseBlock(
            name=next(dense_names),
            dense_layer=dense_layer,
            activation_layer=activation_layer,
        )

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.dense_layer.input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.dense_layer.output_shape

    @property
    def trainable_weights(self) -> list[tf.Variable]:
        return self.dense_layer.trainable_weights


@dataclass
class NormalizedInputBlock:
    name: str
    input_layer: tf.keras.layers.InputLayer
    normalization_layer: ImageNormalization

    @classmethod
    def new(
        cls,
        input_layer: tf.keras.layers.InputLayer,
        normalization_layer: ImageNormalization,
    ) -> NormalizedInputBlock:
        return NormalizedInputBlock(
            name=next(input_names),
            input_layer=input_layer,
            normalization_layer=normalization_layer,
        )

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.input_layer.input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.normalization_layer.output_shape

    @property
    def trainable_weights(self) -> list[tf.Variable]:
        return self.normalization_layer.trainable_weights


def _group_layers(
    layers: Iterable[tf.keras.layers.Layer],
) -> Iterator[
    tf.keras.layers.Layer | ConvolutionalBlock | DenseBlock | NormalizedInputBlock
]:
    layer_list = [layer for layer in layers if not isinstance(layer, SKIPPED_LAYERS)]
    must_skip = set[int]()

    for index, layer in enumerate(layer_list):
        if index in must_skip:
            continue

        if (
            index < len(layer_list) - 2
            and isinstance(layer, tf.keras.layers.Conv2D)
            and isinstance(layer_list[index + 1], tf.keras.layers.ReLU)
            and isinstance(layer_list[index + 2], tf.keras.layers.MaxPool2D)
        ):
            must_skip |= {index + 1, index + 2}

            yield ConvolutionalBlock.new(
                convolutional_layer=layer,
                activation_layer=layer_list[index + 1],
                pooling_layer=layer_list[index + 2],
            )
        elif (
            index < len(layer_list) - 1
            and isinstance(layer, tf.keras.layers.Dense)
            and isinstance(layer_list[index + 1], tf.keras.layers.ReLU)
        ):
            must_skip |= {index + 1}

            yield DenseBlock.new(
                dense_layer=layer,
                activation_layer=layer_list[index + 1],
            )
        elif (
            index < len(layer_list) - 1
            and isinstance(layer, tf.keras.layers.InputLayer)
            and isinstance(layer_list[index + 1], ImageNormalization)
        ):
            must_skip |= {index + 1}

            yield NormalizedInputBlock.new(
                input_layer=layer,
                normalization_layer=layer_list[index + 1],
            )
        else:
            yield layer


def diagrams_path() -> Path:
    return figures_path().parent / "diagrams"
