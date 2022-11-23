# from boiling_learning.model.training import FitModelReturn


# def get_pretrained_baseline_boiling_model(
#     direct: bool = True,
#     normalize_images: bool = True,
# ) -> FitModelReturn:
#     compiled_model = compile_model(
#         get_baseline_boiling_model(direct=direct, normalize_images=normalize_images),
#         get_baseline_compile_params(),
#     )

#     return fit_boiling_model(
#         compiled_model,
#         baseline_boiling_dataset_direct if direct else baseline_boiling_dataset_indirect,
#         get_baseline_fit_params(),
#         target='Flux [W/cm**2]',
#     )
