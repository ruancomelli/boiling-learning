from management import Mirror

x = Mirror({'propagate': True},
    a='a',
    b='b',
    z=Mirror({'propagate': True},
        x='x',
        y=Mirror.Split(
            first='first',
            alpha='alpha',
            beta='beta'
        )
    ),
    w=Mirror.Split(
        first=1,
        second=2
    )
)

import pprint
pprint.pprint(x.mirror)