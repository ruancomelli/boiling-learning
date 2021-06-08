import time

worker_config_file_path = boiling_learning_path / '_tmp' / 'dynamic_user_pool'

user_pool = bl.utils.worker.DynamicUserPool(
    worker_config_file_path,
    [
        'ruan.comelli@lepten.ufsc.br',
        'ruancomelli@gmail.com',
        'rugortal@gmail.com',
        'pitycomelli@gmail.com',
        'pucacomelli@gmail.com',
        'jmcardoso1944@gmail.com',
        'AZULA',
        'LEPTEN'
    ],
    reset=reset_user_pool,
    overwrite=overwrite_user_pool_state
)
user_pool.stamp_ticket()

if wait_for_others:
    time.sleep(60)

gpu_distribution = {
    'ruan.comelli@lepten.ufsc.br': True,
    'ruancomelli@gmail.com': True,
    'rugortal@gmail.com': True,
    'pitycomelli@gmail.com': True,
    'pucacomelli@gmail.com': True,
}
has_gpu = gpu_distribution.__getitem__
