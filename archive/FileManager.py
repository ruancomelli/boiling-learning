# class FileList(collections.abc.MutableMapping):
#     def __init__(
#         self,
#         reference_path,
#         file_name_fmt: str = None
#     ):
#         self.reference_path = Path(reference_path).absolute().resolve()
#         self.file_name_fmt = file_name_fmt if file_name_fmt is not None else '{index}.data'
#         self.files = {}

#     def __len__(self):
#         return len(self.files)

#     def __iter__(self):
#         return iter(self.files)

#     def __contains__(self, key):
#         return key in self.files

#     def __delitem__(self, key):
#         self.pop_file_path(self, key)

#     def __getitem__(self, key):
#         return self.get_file_path(key)

#     def __setitem__(self, key, item):
#         self.new_file_path(file_key=key, file_path=item)

#     def new_file_path(self, file_key, file_path=None):
#         if file_path is None:
#             int_key_list = [
#                 int(parse.parse(self.file_name_fmt, file_name))
#                 for file_name in self.files.values()
#             ]

#             missing_elems = boiling_learning.utils.missing_elements(int_key_list)
#             if missing_elems:
#                 index = missing_elems[0]
#             else:
#                 index = str(max(int_key_list, default=-1) + 1)

#             file_name = self.file_name_fmt.format(index=index)
#             file_path = self.reference_path / file_name

#         self.files[file_key] = file_path

#         return file_path

#     def get_file_path(self, file_key):
#         return self.files[file_key]

#     def pop_file_path(self, file_key, default=None):
#         return self.files.pop(file_key, default)

#     def provide_file_path(self, file_key):
#         if file_key not in self.files:
#             return self.new_file_path(file_key)
#         else:
#             return self.get_file_path(file_key)

# class FileManager:
#     def __init__(
#         self,
#         reference_path: Path = None
#     ):
#         self.reference_path = Path(
#             reference_path
#             if reference_path is not None
#             else Path('.')
#         ).absolute().resolve()

#         self.files = {}
#         self.file_lists = {}

#     def new_file_path(self, file_key, file_path=None):
#         file_path = (
#             file_path
#             if file_path is not None
#             else self.reference_path / file_key
#         ).absolute().resolve()
#         self.files[file_key] = file_path

#         return file_path

#     def get_file_path(self, file_key):
#         return self.files[file_key]

#     def pop_file_path(self, file_key, default=None):
#         return self.files.pop(file_key, default)

#     def provide_file_path(self, file_key):
#         if file_key not in self.files:
#             return self.new_file_path(file_key)
#         else:
#             return self.get_file_path(file_key)


#     def new_file_list(
#         self,
#         file_list_name,
#         reference_path=None,
#         file_name_fmt: str = None
#     ):
#         reference_path = Path(
#             reference_path
#             if reference_path is not None
#             else self.reference_path / file_list_name
#         ).absolute().resolve()
#         self.file_lists[file_list_name] = FileList(reference_path, file_name_fmt)

#         return reference_path

#     def get_file_list(self, file_list_name):
#         return self.file_lists[file_list_name]

#     def pop_file_list(self, file_list_name, default=None):
#         return self.file_lists.pop(file_list_name, default)

#     def provide_file_list(
#         self,
#         file_list_name,
#         reference_path,
#         file_name_fmt: str = None
#     ):
#         if file_list_name not in self.file_lists:
#             return self.new_file_list(file_list_name, reference_path, file_name_fmt)
#         else:
#             return self.get_file_list(file_list_name)

# # def print_exists(x):
# #     print(f'{x.exists()}: {x}')

# # file_manager = FileManager(Path('.'))
# # file_manager.new_file_path('calibration', Path('..') / 'Experimental Set Calibration' / 'Processing' / 'coefficients.csv')
# # file_manager.new_file_list('experiments', Path('.') / 'experiments')
# # print_exists(file_manager.files['calibration'])
# # print_exists(file_manager.file_lists['experiments'].reference_path)


# # self.calibration_file_path = (
# #     calibration_file_path
# #     if calibration_file_path is not None
# #     else self.reference_path.parent.parent / 'Experimental Set Calibration' / 'Processing' / 'coefficients.csv'
# # )

# # self.experiment_dir_path = (
# #     experiment_dir_path
# #     if experiment_dir_path is not None
# #     else self.reference_path / 'experiments'
# # )

# # self.experiment_file_list = FileList(
# #     self.experiment_dir_path,
# #     file_name_fmt='experiment_{index:03d}.csv'
# # )

# # self.models_dir_path = (
# #     models_dir_path
# #     if models_dir_path is not None
# #     else self.reference_path / 'models'
# # )

# # self.models_file_list = FileList(
# #     self.models_dir_path,
# #     file_name_fmt='model_{index:03d}.data'
# # )
