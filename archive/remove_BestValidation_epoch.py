import parse

pattern = 'BestValidation_epoch{epoch}'
parser = parse.compile(pattern).parse
glob_pattern = 'BestValidation_epoch*'
for path in manager.models_path.iterdir():
    if path.is_dir():
        best_validation_folders = list(path.glob(glob_pattern))
        if best_validation_folders:
            to_erase, to_keep = (
                best_validation_folders[:-1],
                best_validation_folders[-1],
            )
            to_erase_epochs = [parser(p.name) for p in to_erase]
            to_erase_epochs = [
                int(p['epoch'])
                for p in to_erase_epochs
                if (p and 'epoch' in p)
            ]
            to_keep_epochs = parser(to_keep.name)
            to_keep_epochs = int(to_keep_epochs['epoch'])
            if not all(x < to_keep_epochs for x in to_erase_epochs):
                raise ValueError('order was not preserved')
            for p in to_erase:
                print(f'Erasing {p}')
                bl.utils.rmdir(p, recursive=True, keep=False)
            renamed = to_keep.with_name('BestValidation')
            print(f'Renaming {to_keep} to {renamed}')
            to_keep.rename(renamed)
            print()
