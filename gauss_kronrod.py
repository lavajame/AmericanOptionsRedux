import numpy as np
from enum import Enum
from numpy.polynomial.legendre import leggauss


class GKRule(Enum):
    FAST = "7/15"
    BALANCED = "10/21"
    ACCURATE = "15/31"
    ULTRA = "20/41"


class GaussKronrodRule:
    """
    Fixed Gauss–Kronrod-style rule on [-1, 1].

    Options:
    - Provide custom nodes/weights.
    - Provide gk_n to generate a (gk_n, 2*gk_n+1) Gauss–Legendre pair on the fly.
      Note: this is a Gauss–Legendre rule with 2*gk_n+1 points, not a true
      Kronrod extension. It is used here as a cheap fixed-node alternative.
    - Built-in rule "7/15" uses the classic 15-point Kronrod rule extending 7-point Gauss.
    """

    _RULES = {
        "7/15": {
            "pos_x": [
                0.9914553711208126,
                0.9491079123427585,
                0.8648644233597691,
                0.7415311855993945,
                0.5860872354676911,
                0.4058451513773972,
                0.2077849550078985,
                0.0,
            ],
            "pos_w": [
                0.02293532201052922,
                0.06309209262997855,
                0.1047900103222502,
                0.1406532597155259,
                0.1690047266392679,
                0.1903505780647854,
                0.2044329400752989,
                0.2094821410847278,
            ],
        }
        ,
        "10/21": {
            "pos_x": [
                0.9956571630258081,
                0.9739065285171717,
                0.9301574913557082,
                0.8650633666889845,
                0.7808177265864169,
                0.6794095682990244,
                0.5627571346686047,
                0.4333953941292472,
                0.2943928627014602,
                0.1488743389816312,
                0.0,
            ],
            "pos_w": [
                0.01169463886737187,
                0.03255816230796473,
                0.054755896574352,
                0.07503967481091996,
                0.09312545458369761,
                0.1093871588022976,
                0.1234919762620659,
                0.1347092173114733,
                0.1427759385770601,
                0.1477391049013385,
                0.1494455540029169,
            ],
        },
        "15/31": {
            "pos_x": [
                0.9980022986933971,
                0.9879925180204854,
                0.9677390756791391,
                0.937273392400706,
                0.8972645323440819,
                0.8482065834104272,
                0.790418501442466,
                0.72441773136017,
                0.650996741297417,
                0.5709721726085388,
                0.4850818636402397,
                0.3941513470775634,
                0.2991800071531688,
                0.2011940939974345,
                0.1011420669187175,
                0.0,
            ],
            "pos_w": [
                0.005377479872923349,
                0.01500794732931612,
                0.02546084732671532,
                0.03534636079137585,
                0.04458975132476488,
                0.05348152469092809,
                0.06200956780067064,
                0.06985412131872826,
                0.07684968075772038,
                0.08308050282313302,
                0.08856444305621176,
                0.09312659817082532,
                0.09664272698362368,
                0.09917359872179196,
                0.1007698455238756,
                0.1013300070147915,
            ],
        },
        "20/41": {
            "pos_x": [
                0.9988590315882777,
                0.9931285991850949,
                0.9815078774502503,
                0.9639719272779138,
                0.9408226338317548,
                0.912234428251326,
                0.878276811252282,
                0.8391169718222188,
                0.7950414288375512,
                0.7463319064601508,
                0.6932376563347514,
                0.636053680726515,
                0.5751404468197103,
                0.5108670019508271,
                0.4435931752387251,
                0.3737060887154196,
                0.301627868114913,
                0.2277858511416451,
                0.1526054652409227,
                0.07652652113349732,
                0.0,
            ],
            "pos_w": [
                0.003073583718520532,
                0.008600269855642943,
                0.01462616925697125,
                0.02038837346126652,
                0.02588213360495116,
                0.0312873067770328,
                0.0366001697582008,
                0.04166887332797369,
                0.04643482186749767,
                0.05094457392372869,
                0.05519510534828599,
                0.05911140088063957,
                0.06265323755478117,
                0.06583459713361842,
                0.06864867292852161,
                0.07105442355344407,
                0.07303069033278667,
                0.0745828754004992,
                0.07570449768455667,
                0.07637786767208074,
                0.07660071191799966,
            ],
        },
    }

    def __init__(self, rule=GKRule.FAST, gk_n=None, nodes=None, weights=None):
        if nodes is not None or weights is not None:
            if nodes is None or weights is None:
                raise ValueError("Both nodes and weights must be provided for a custom rule")
            self._x = np.asarray(nodes, dtype=float)
            self._w = np.asarray(weights, dtype=float)
            if self._x.shape != self._w.shape:
                raise ValueError("nodes and weights must have the same shape")
        elif gk_n is not None:
            n = int(gk_n)
            if n < 1:
                raise ValueError("gk_n must be >= 1")
            # Use Gauss–Legendre with (2n+1) points as a fixed-node rule.
            self._x, self._w = leggauss(2 * n + 1)
        else:
            if isinstance(rule, GKRule):
                rule = rule.value
            if rule not in self._RULES:
                raise ValueError(f"Unknown rule '{rule}'. Provide gk_n or custom nodes/weights.")
            pos_x = np.array(self._RULES[rule]["pos_x"], dtype=float)
            pos_w = np.array(self._RULES[rule]["pos_w"], dtype=float)
            neg_x = -pos_x[:-1][::-1]
            neg_w = pos_w[:-1][::-1]
            self._x = np.concatenate([neg_x, pos_x])
            self._w = np.concatenate([neg_w, pos_w])

    def integrate(self, func, a, b):
        if b <= a:
            return 0.0
        t = 0.5 * (b - a) * self._x + 0.5 * (b + a)
        ft = np.array([func(v) for v in t], dtype=float)
        return 0.5 * (b - a) * np.sum(self._w * ft)
