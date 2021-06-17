from pathlib import Path

import boiling_learning as bl

boiling_learning_path = Path()
drive_user: str = 'ruan.comelli@lepten.ufsc.br'
nb_user: str = 'pucacomelli@gmail.com'
work_manager: str = 'ruancomelli@gmail.com'

worker_config_file_path = boiling_learning_path / '_tmp' / 'user_pool.json'

if nb_user == work_manager:
    user_pool = bl.utils.worker.UserPool(
        [
            'ruan.comelli@lepten.ufsc.br',
            'ruancomelli@gmail.com',
            'rugortal@gmail.com',
            'pitycomelli@gmail.com',
            'pucacomelli@gmail.com',
            # 'brunghah@gmail.com',
        ],
        manager=work_manager,
        current=nb_user,
        server=drive_user,
    )
    user_pool.to_json(worker_config_file_path)

user_pool = bl.utils.worker.UserPool.from_json(worker_config_file_path)
user_pool.current = nb_user
