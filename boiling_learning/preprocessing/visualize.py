from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import bokeh.models
import bokeh.plotting
import funcy
import matplotlib as mpl
import matplotlib.pyplot as plt
import more_itertools as mit
import numpy as np
import tensorflow as tf
from frozendict import frozendict
from matplotlib import gridspec
from matplotlib.colors import NoNorm

import boiling_learning.utils.utils as bl_utils
from boiling_learning.management import Manager
from boiling_learning.preprocessing.Case import Case
from boiling_learning.preprocessing.experiment_video import ExperimentVideo
from boiling_learning.preprocessing.ImageDataset import ImageDataset
from boiling_learning.preprocessing.transformers import DictFeatureTransformer, Transformer
from boiling_learning.utils.functional import Kwargs, P, Pack, nth_arg
from boiling_learning.utils.Parameters import Parameters

_T = TypeVar('_T')
ImageType = Any

DEFAULT_ANNOTATORS = frozendict(
    {
        'grayscaler': None,
        'normalizer': None,
        'downscaler': None,
        'region_cropper': lambda transformer, image, fig: fig.rect(
            x=(transformer.pack.kwargs['left'] + transformer.pack.kwargs['right']) / 2,
            y=-(transformer.pack.kwargs['bottom'] + transformer.pack.kwargs['top']) / 2,
            width=transformer.pack.kwargs['right'] - transformer.pack.kwargs['left'],
            height=transformer.pack.kwargs['bottom'] - transformer.pack.kwargs['top'],
            fill_alpha=0.3,
            fill_color='royalblue',
        ),
        'visualization_shrinker': lambda transformer, image, fig: fig.rect(
            x=image.shape[1] / 2,
            y=-image.shape[0] * (1 - transformer.pack.kwargs['bottom']) / 2,
            width=image.shape[1],
            height=image.shape[0] * (1 - transformer.pack.kwargs['bottom']),
            fill_alpha=0.3,
            fill_color='yellow',
        ),
        'final_height_shrinker': lambda transformer, image, fig: fig.rect(
            x=(
                transformer.pack.kwargs['left']
                + (
                    image.shape[1]
                    - (transformer.pack.kwargs['right'] + transformer.pack.kwargs['left'])
                )
                / 2
            ),
            y=-image.shape[0]
            + transformer.pack.kwargs['bottom']
            + transformer.pack.kwargs['height'] / 2,
            width=image.shape[1]
            - (transformer.pack.kwargs['right'] + transformer.pack.kwargs['left']),
            height=transformer.pack.kwargs['height'],
            fill_alpha=0.3,
            fill_color='red',
        ),
        'random_cropper': lambda transformer, image, fig: fig.rect(
            x=image.shape[1] / 2,
            y=-image.shape[0] / 2,
            width=transformer.pack.args[0][1],
            height=image.shape[0],
            fill_alpha=0.3,
            fill_color='green',
        ),
        'random_left_right_flipper': None,
        'random_brightness': None,
        'random_contrast': None,
        'random_quality': None,
    }
)

_first_arg_getter = Transformer('first_argument', nth_arg(0))

DEFAULT_VISUALIZERS = frozendict(
    {
        'grayscaler': lambda transformer, image: (
            (transformer.transform_feature, image, transformer.pack),
        ),
        'normalizer': lambda transformer, image: (
            (transformer.transform_feature, image, transformer.pack),
        ),
        'downscaler': lambda transformer, image: (
            (transformer.transform_feature, image, transformer.pack),
        ),
        'region_cropper': lambda transformer, image: (
            (transformer.transform_feature, image, transformer.pack),
        ),
        'visualization_shrinker': lambda transformer, image: (
            (transformer.transform_feature, image, transformer.pack),
        ),
        'final_height_shrinker': lambda transformer, image: (
            (transformer.transform_feature, image, transformer.pack),
        ),
        'random_cropper': lambda transformer, image: (
            (transformer.transform_feature, image, transformer.pack),
        ),
        'random_left_right_flipper': lambda transformer, image: (
            (_first_arg_getter, image, P()),
            (
                Transformer('left_right_flipper', tf.image.flip_left_right),
                image,
                P(),
            ),
        ),
        'random_brightness': lambda transformer, image: (
            (_first_arg_getter, image, P()),
            (
                Transformer(
                    'brightness_adjuster',
                    tf.image.adjust_brightness,
                    pack=P(transformer.pack.args[0]),
                ),
                image,
                P(transformer.pack.args[0]),
            ),
            (
                Transformer(
                    'brightness_adjuster',
                    tf.image.adjust_brightness,
                    pack=P(transformer.pack.args[1]),
                ),
                image,
                P(transformer.pack.args[1]),
            ),
        ),
        'random_contrast': lambda transformer, image: (
            (_first_arg_getter, image, P()),
            (
                Transformer(
                    'contrast_adjuster',
                    tf.image.adjust_contrast,
                    pack=P(transformer.pack.args[0]),
                ),
                image,
                P(transformer.pack.args[0]),
            ),
            (
                Transformer(
                    'contrast_adjuster',
                    tf.image.adjust_contrast,
                    pack=P(transformer.pack.args[1]),
                ),
                image,
                P(transformer.pack.args[1]),
            ),
        ),
        'random_quality': lambda transformer, image: (
            (_first_arg_getter, image, P()),
            (
                Transformer(
                    'jpeg_quality_adjuster',
                    tf.image.adjust_jpeg_quality,
                    pack=P(transformer.pack.args[0]),
                ),
                image,
                P(transformer.pack.args[0]),
            ),
            (
                Transformer(
                    'jpeg_quality_adjuster',
                    tf.image.adjust_jpeg_quality,
                    pack=P(transformer.pack.args[1]),
                ),
                image,
                P(transformer.pack.args[1]),
            ),
        ),
    }
)


def visualize_transformations(
    ev: ExperimentVideo,
    idx: int,
    transformers: Sequence[Transformer],
    method: str,
    visualizers: Mapping[
        str,
        Callable[
            [Transformer, ImageType],
            Sequence[Tuple[Transformer, ImageType, Pack]],
        ],
    ] = DEFAULT_VISUALIZERS,
    annotators: Optional[
        Mapping[
            str,
            Optional[
                Callable[
                    [Transformer, ImageType, bokeh.plotting.Figure],
                    bokeh.plotting.Figure,
                ]
            ],
        ]
    ] = DEFAULT_ANNOTATORS,
    plot_original: bool = True,
    normalize: bool = False,
) -> Union[List[mpl.figure.Figure], List[bokeh.plotting.Figure]]:
    METHODS = frozendict(
        {
            'plt': _visualize_transformations_plt,
            'bokeh': _visualize_transformations_bokeh,
        }
    )

    try:
        return METHODS[method](
            ev=ev,
            idx=idx,
            transformers=transformers,
            visualizers=visualizers,
            annotators=annotators,
            plot_original=plot_original,
            normalize=normalize,
        )
    except KeyError as e:
        raise ValueError(f'method must be one of {set(METHODS.keys())}') from e


def _visualize_transformations_plt(
    ev: ExperimentVideo,
    idx: int,
    transformers: Sequence[Transformer],
    visualizers: Mapping[
        str,
        Callable[
            [Transformer, ImageType],
            Sequence[Tuple[Transformer, ImageType, Pack]],
        ],
    ] = DEFAULT_VISUALIZERS,
    annotators=None,
    plot_original: bool = True,
    normalize: bool = False,
) -> List[mpl.figure.Figure]:
    print('Transformers:', transformers)

    visualization_title = f'{ev.name}[{idx}]'
    image = ev[idx]

    if normalize:
        image = image / 255

    print('Original image shape:', image.shape)

    figs = []
    if plot_original:
        bl_utils.prepare_fig(n_rows=1, n_cols=1, subfig_size='small')
        fig = plt.figure()
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0])
        ax.imshow(image)
        ax.set_title(visualization_title)
        figs.append(fig)

    for transformer in transformers:
        transformer_name = transformer.name
        print(transformer_name)
        if isinstance(transformer, DictFeatureTransformer):
            transformer = transformer[ev.name]

        visualizer = visualizers[transformer_name]
        f_image_pack_pairs = visualizer(transformer, image)
        n_subfigs = len(f_image_pack_pairs)

        bl_utils.prepare_fig(n_rows=1, n_cols=n_subfigs, subfig_size='small')
        fig = plt.figure()
        gs = fig.add_gridspec(1, n_subfigs)

        for sgs, (f, image, pack_) in zip(gs, f_image_pack_pairs):
            ax = fig.add_subplot(sgs)
            img_to_show = np.squeeze(f(image))
            xlabel = f'shape={img_to_show.shape} minmax=({img_to_show.min()}, {img_to_show.max()})'
            print(xlabel)

            cmap = None if len(img_to_show.shape) == 3 else 'gray'

            ax.imshow(img_to_show, cmap=cmap, norm=NoNorm())
            ax.set_title(f'{transformer_name}({pack_})')
            ax.set_xlabel(xlabel)

        figs.append(fig)
        image = transformer.transform_feature(image)
    return figs


def _make_fig(image, *args, **kwargs):
    hover_tool = bokeh.models.tools.HoverTool(
        tooltips=[
            ('x', '$x{(0.)}'),
            ('y', '$y{(0.)}'),
        ]
    )

    height = 256
    img_height = image.shape[0]
    img_width = image.shape[1]
    p = bokeh.plotting.figure(
        *args,
        **kwargs,
        tools=[hover_tool, bokeh.models.tools.CrosshairTool()],
        toolbar_location=None,
        x_range=(0, img_width),
        y_range=(-img_height, 0),
        match_aspect=True,
        plot_height=height,
        plot_width=height * img_width // img_height,
    )
    image = np.flipud(image)
    p.image(image=[image], x=0, y=-img_height, dw=img_width, dh=img_height)

    return p


def _tensor_to_image(tensor: tf.Tensor) -> np.ndarray:
    return np.squeeze(tensor.numpy())


def _make_figs(f_img_packs, return_single_image: bool = False):

    figs = [
        _make_fig(_tensor_to_image(f(img)), y_axis_label=str(pack)) for f, img, pack in f_img_packs
    ]

    if len(figs) == 1 and return_single_image:
        return figs[0]
    else:
        return bokeh.layouts.column(*figs)


def _visualize_transformations_bokeh(
    ev: ExperimentVideo,
    idx: int,
    transformers: Sequence[Transformer],
    visualizers: Mapping[
        str,
        Callable[
            [Transformer, ImageType],
            Sequence[Tuple[Transformer, ImageType, Pack]],
        ],
    ] = DEFAULT_VISUALIZERS,
    annotators: Mapping[
        str,
        Optional[
            Callable[
                [Transformer, ImageType, bokeh.plotting.Figure],
                bokeh.plotting.Figure,
            ]
        ],
    ] = DEFAULT_ANNOTATORS,
    plot_original: bool = True,
    normalize: bool = False,
) -> List[bokeh.plotting.Figure]:
    print('Transformers:', transformers)

    visualization_title = f'{ev.name}[{idx}]'
    image = ev[idx]
    if normalize:
        image = image / 255

    print('Original image shape:', image.shape)

    ps = []
    if plot_original:
        p = _make_fig(image, title=visualization_title)
        ps.append(p)

    for first, _, transformer in mit.mark_ends(transformers):
        transformer_name = transformer.name
        print(transformer_name)
        if isinstance(transformer, DictFeatureTransformer):
            transformer = transformer[ev.name]

        annotator = annotators[transformer_name]
        if annotator is None:
            annotator = funcy.constantly(None)

        if not first or plot_original:
            annotator(transformer, image, p)

        visualizer = visualizers[transformer_name]
        p = _make_figs(visualizer(transformer, image), return_single_image=True)
        p_canvas = bokeh.layouts.column(bokeh.models.Div(text=transformer_name), p)
        if first and not plot_original:
            p_canvas = bokeh.layouts.row(bokeh.models.Div(text=visualization_title), p_canvas)

        image = transformer.transform_feature(image)
        ps.append(p_canvas)

    return ps


def visualize_dataset(
    manager: Manager[tf.data.Dataset, tf.data.Dataset],
    case_list: Iterable[Case],
    params: Parameters,
    load: Union[bool, Callable[[], Tuple[bool, _T]]] = False,
    save: Union[bool, Callable[[Union[_T, Any]], Any]] = False,
    n_samples_per_ev: int = 0,
) -> None:
    # See <https://stackoverflow.com/a/34934631/5811400> for plotting
    def _make_ds(params: Parameters) -> tf.data.Dataset:
        return manager.provide_elem(
            creator_description=Kwargs(params[['creator', 'desc']]),
            creator_params=Kwargs(params[['creator', 'value']]),
            post_processor_description=Kwargs(params[['post_processor', 'desc']]),
            post_processor_params=Kwargs(params[['post_processor', 'value']]),
            load=load,
            save=save,
        )

    pad = 2

    if n_samples_per_ev == 0:
        img_ds_list = case_list
        n_samples = 1
    else:
        img_ds_list = []
        for case in case_list:
            for ev in case.values():
                img_ds = ImageDataset(
                    ev.name,
                    column_names=ev.column_names,
                    column_types=ev.column_types,
                )
                img_ds.add(ev)
                img_ds_list.append(img_ds)
        n_samples = n_samples_per_ev

    params[['creator', {'desc', 'value'}, 'dataset_size']] = n_samples
    fig_spec = bl_utils.prepare_fig(n_cols=3, n_rows=len(img_ds_list), subfig_size='small')
    fig = plt.figure(figsize=fig_spec['fig_size'])
    outer = gridspec.GridSpec(fig_spec['n_rows'], fig_spec['n_cols'])
    for row, img_ds in enumerate(img_ds_list):
        params[['creator', 'desc', 'image_dataset']] = sorted(img_ds.keys())
        params[['creator', 'value', 'image_dataset']] = img_ds

        ds = _make_ds(params)
        for col, (split_name, ds_split) in enumerate(zip(('train', 'val', 'test'), ds)):
            elem = row * fig_spec['n_cols'] + col
            inner = gridspec.GridSpecFromSubplotSpec(
                n_samples,
                1,
                subplot_spec=outer[elem],
            )
            for sample, (img, _) in enumerate(ds_split.take(n_samples)):
                img = tf.image.convert_image_dtype(img, tf.uint8)

                ax = plt.Subplot(fig, inner[sample])
                img = np.squeeze(img)
                ax.imshow(img, cmap='gray', norm=NoNorm())

                ax.set_xlabel(f'[{img.min()}, {img.max()}]')
                ax.set_xticks([])
                ax.set_yticks([])

                if row == 0 and sample == 0:
                    ax.annotate(
                        split_name,
                        xy=(0.5, 1),
                        xytext=(0, pad),
                        xycoords='axes fraction',
                        textcoords='offset points',
                        size='large',
                        ha='center',
                        va='baseline',
                    )

                if col == 0:
                    ax.annotate(
                        img_ds.name,
                        xy=(0, 0.5),
                        xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label,
                        textcoords='offset points',
                        size='large',
                        ha='right',
                        va='center',
                    )

                fig.add_subplot(ax)
                fig.show()
