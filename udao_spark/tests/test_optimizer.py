import numpy as np

from ..optimizer.utils import weighted_utopia_nearest


def test_utopia_nearest() -> None:
    po_objs = np.array(
        [
            [0.571963906288147, 0.0045116416296772655],
            [0.5702601671218872, 0.0045670890688858925],
            [0.5673424601554871, 0.004591371871823445],
            [0.5711890459060669, 0.004533636013269011],
            [0.5703619122505188, 0.004567058871627475],
            [0.5719089508056641, 0.004527660221639607],
            [0.567581295967102, 0.004579991659531577],
            [0.571953296661377, 0.0045260947925133835],
            [0.5711076259613037, 0.004545867563271688],
            [0.5731040835380554, 0.0045050495237172064],
            [0.5673501491546631, 0.004582551325974365],
            [0.5734590888023376, 0.004499184697285502],
            [0.5705934166908264, 0.004564498913269697],
        ],
        dtype=object,
    )
    po_confs = np.array(
        [
            ["0.12", "1536KB"],
            ["0.09", "512KB"],
            ["0.13", "512KB"],
            ["0.13", "2560KB"],
            ["0.12", "1024KB"],
            ["0.12", "2048KB"],
            ["0.11", "512KB"],
            ["0.13", "1536KB"],
            ["0.12", "2560KB"],
            ["0.09", "2048KB"],
            ["0.12", "512KB"],
            ["0.1", "1536KB"],
            ["0.11", "1024KB"],
        ]
    )
    ret = weighted_utopia_nearest(po_objs, po_confs, weights=np.array([1, 1]))
    assert ret is not None
