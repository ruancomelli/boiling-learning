in_path = python_project_home_path / 'testing_extract'
df = pd.DataFrame(
    {
        'path': list(in_path.glob('**/*.png'))
    }
)

img_ds = ImageDataset(
    df_path=python_project_home_path / 'my_csv.csv',
    path_column='path',
    target_column='target',
    set_column=None,
    train_key='train',
    val_key='val',
    test_key='test',
    df=df
)
img_ds.transform_images(
    image_dataset_transformer(img_ds)
)