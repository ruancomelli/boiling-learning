from typing import Dict, List, Tuple

from pint import Quantity
from typing_extensions import Literal

from boiling_learning.utils.dataclasses import dataclass
from boiling_learning.utils.units import unit_registry as ureg

Q_ = ureg.Quantity


@dataclass(frozen=True)
class NukiyamaBoilingCurve:
    """Experimental data published by Nukiyama.

    NUKIYAMA, Shiro. The maximum and minimum values of the heat Q transmitted from metal to boiling
    water under atmospheric pressure. International Journal of Heat and Mass Transfer, v. 9, p.
    1419-1433, Dec. 1966. Translated from Journal of the Society of Mechanical Engineers, 37,
    367-374, Japan, 1934. Available at http://www.htsj.or.jp/wp/media/IJHMT1984-3.pdf. Table 3.
    """

    material: str = 'Nichrome'
    diameter: Quantity[float] = 0.575 * ureg.mm
    length: Quantity[int] = 200 * ureg.mm

    @staticmethod
    def fetch() -> Tuple[Quantity[List[float]], Quantity[List[float]]]:
        temperature_excess = Q_(
            [
                3,
                8,
                13.5,
                18.8,
                25.7,
                31.0,
                35.5,
                38.0,
                44.0,
                46.5,
            ],
            ureg.delta_degC,
        )

        heat_flux = Q_(
            [
                0.0527,
                1.385,
                5.44,
                12.66,
                22.42,
                27.15,
                32.43,
                35.29,
                38.22,
                40.48,
            ],
            ureg.cal / (ureg.cm ** 2 * ureg.s),
        )

        return temperature_excess.to(ureg.delta_degC), heat_flux.to(ureg.W / ureg.cm ** 2)


class IncroperaBoilingCurveImposedHeat:
    @staticmethod
    def fetch() -> Tuple[Quantity[List[float]], Quantity[List[float]]]:
        temperature_excess = Q_(
            [
                1.0212414255144324,
                1.1018964979744297,
                1.1889215046644752,
                1.2828195269266038,
                1.384133378198594,
                1.4934487419546942,
                1.6113975574729946,
                1.7386616730021682,
                1.8759767874461115,
                2.0241367033528923,
                2.183997915794044,
                2.3564845636619753,
                2.542593772008357,
                2.743401416306932,
                2.9600683419632574,
                3.1938470750256442,
                3.446089062891131,
                3.7182524868641442,
                4.011910691731348,
                4.328761281083059,
                4.670635930960206,
                5.03951097855838,
                5.4375188472010265,
                5.866960373628218,
                6.330318108863595,
                6.830270669550249,
                7.344288543409449,
                7.8751910401332985,
                8.403721992661401,
                8.820222784287909,
                9.225435356897236,
                9.615981577046774,
                10.023061038628551,
                10.447373653863334,
                10.889648964800902,
                11.35064739765846,
                11.831161570259699,
                12.332017654822565,
                12.854076798438758,
                13.360413811729222,
                13.954301804602558,
                14.515547283758346,
                15.120541265442219,
                15.760648751905565,
                16.427854315549713,
                17.123305116503897,
                17.84819687835642,
                18.60377594402246,
                19.379163345041942,
                20.085669376500974,
                20.922820663553836,
                21.794863607154348,
                22.575252277201308,
                23.368898342071958,
                24.190445493874016,
                25.040874611473264,
                25.841712302827947,
                26.815624663167114,
                27.77578711449342,
                28.85177686130595,
                30.003985167957016,
                31.782519390120363,
                34.17434012946951,
                36.1792385571704,
                38.05677194042588,
                41.4697721662008,
                43.52507491787161,
                45.972533391256846,
                46.71980945978444,
                49.10467120234631,
                50.471865584313555,
                52.99947917909825,
                54.67347350061054,
                57.203209480434374,
                58.99144656423765,
                61.701590787218365,
                63.65044225153255,
                66.49799534354108,
                68.11024691023756,
                71.33784649903893,
                73.59106014939877,
                76.97192971851062,
                79.40309650090876,
                83.05097862284293,
                85.67415282689251,
                89.61013547972898,
                92.12163463417896,
                96.02147836062997,
                99.05432721138436,
                103.6050125783019,
                107.30025671164387,
                114.58504567779985,
                120.61616809495135,
                128.8001054489834,
                138.01537596173083,
                144.85562048610808,
                154.15069231610494,
                166.32512526397642,
                179.46106422505665,
                193.63444651958005,
                208.9272068058376,
                225.42775073481255,
                243.23146601286413,
                262.44127382685554,
                283.1682238194456,
                305.5321360540294,
                329.6622936804517,
                355.69819030571045,
                382.46656250677074,
                409.8309254728908,
                439.1531285064763,
                470.573249333236,
                504.2413878301097,
                538.4547109234384,
                574.9894449626746,
                614.0031001892266,
                653.4023509807894,
                695.3297664712895,
                739.947573520795,
                787.4284087360724,
                837.9559864415148,
                891.7258095123201,
                945.6728110120328,
                1002.1906724663049,
                1068.8764645047154,
                1128.6771000346314,
                1196.13215977998,
                1267.6186516196867,
                1345.2354373804974,
                1417.684429838068,
                1504.504342578914,
                1587.9079004257346,
                1676.8261153000299,
                1767.4321617454282,
                1857.9892584653082,
                1940.2511375462348,
                2095.296401133923,
                2227.3690811254255,
                2346.328932095881,
                2456.0799216334617,
                2575.752256913151,
                2701.255618986861,
                2809.900789876056,
                2925.176526833793,
                3029.915636955091,
                3082.714818080651,
            ],
            ureg.delta_degC,
        )

        heat_flux_ratio = Q_(
            [
                -0.4224776170798896,
                -0.42028236914600536,
                -0.4176997245179064,
                -0.4136966253443526,
                -0.4090478650137741,
                -0.40491563360881533,
                -0.40013774104683164,
                -0.39548898071625316,
                -0.3907110881542699,
                -0.3854166666666665,
                -0.37986398071625316,
                -0.3740530303030303,
                -0.36759641873278226,
                -0.3608815426997245,
                -0.3530044765840221,
                -0.34499827823691454,
                -0.33595902203856753,
                -0.32601584022038566,
                -0.3143939393939392,
                -0.30186811294765836,
                -0.2877926997245177,
                -0.2721676997245177,
                -0.25434745179063345,
                -0.23368629476584024,
                -0.21070075757575757,
                -0.18397038567493107,
                -0.1543087121212119,
                -0.12249053030303036,
                -0.09214743589743568,
                -0.06104572510822526,
                -0.028764204545454586,
                0.002722537878787623,
                0.03409090909090917,
                0.06593276515151536,
                0.09789299242424243,
                0.12973484848484862,
                0.16193181818181834,
                0.19412878787878807,
                0.22703598484848486,
                0.2576618457300277,
                0.2909382284382287,
                0.32313188705234186,
                0.3553503787878789,
                0.38944128787878807,
                0.423532196969697,
                0.4576231060606062,
                0.49171401515151536,
                0.5259232954545456,
                0.5600895316804408,
                0.5930182506887054,
                0.6281960227272729,
                0.6636535812672177,
                0.6966856060606061,
                0.7289299242424244,
                0.7610321969696972,
                0.7938446969696971,
                0.8246527777777779,
                0.8579975895316806,
                0.8915719696969698,
                0.9224668560606062,
                0.9517045454545455,
                0.9856060606060607,
                1.0077909779614327,
                1.015625,
                1.007981601731602,
                0.978607093663912,
                0.9543087121212123,
                0.9108664772727274,
                0.8905066287878789,
                0.8453426308539946,
                0.8231872294372296,
                0.7732007575757577,
                0.7514204545454547,
                0.7021349862258954,
                0.6808712121212122,
                0.6316287878787882,
                0.6098484848484849,
                0.5611979166666667,
                0.5412878787878789,
                0.49325284090909105,
                0.47159090909090917,
                0.4224668560606062,
                0.40127840909090917,
                0.3521543560606062,
                0.33001893939393945,
                0.2812500000000002,
                0.2608901515151516,
                0.21366003787878807,
                0.19223484848484862,
                0.14322916666666674,
                0.12290313852813872,
                0.07634032634032639,
                0.06096117424242431,
                0.02690254820936655,
                0.016177398989899006,
                0.01822916666666652,
                0.021780303030302983,
                0.02793560606060619,
                0.03684573002754821,
                0.04962982093663926,
                0.0675792011019285,
                0.08914428374655659,
                0.1131628787878789,
                0.13950585399449045,
                0.16701101928374662,
                0.1958075068870524,
                0.22679924242424243,
                0.25804924242424243,
                0.28901515151515156,
                0.3189867424242425,
                0.3498106060606061,
                0.38120265151515165,
                0.4130208333333334,
                0.4439709595959598,
                0.47537878787878807,
                0.5072601010101012,
                0.5377209595959598,
                0.5684974747474748,
                0.5997474747474748,
                0.631628787878788,
                0.6636679292929294,
                0.6957070707070708,
                0.7262665719696971,
                0.7557765151515152,
                0.7902462121212122,
                0.8185606060606062,
                0.8496685606060608,
                0.8824810606060608,
                0.9138257575757577,
                0.9445161845730029,
                0.977017773892774,
                1.0089962121212124,
                1.0388621794871797,
                1.0697375541125542,
                1.1016244172494174,
                1.127722537878788,
                1.1799775094696972,
                1.2123440285204992,
                1.2481060606060608,
                1.2800116550116551,
                1.3131425865800868,
                1.3473193473193474,
                1.3796918044077136,
                1.4124913911845731,
                1.445707070707071,
                1.4727746212121213,
            ]
        )

        return temperature_excess, heat_flux_ratio

    @staticmethod
    def extra() -> Dict[Literal['MAX', 'MIN', 'BURNOUT'], Tuple[Quantity[float], Quantity[float]]]:
        return {
            'MAX': (
                Q_(36.1792385571704, ureg.delta_degC),
                Q_(1),
            ),
            'MIN': (
                Q_(144.85562048610808, ureg.delta_degC),
                Q_(0.01822916666666652),
            ),
            'BURNOUT': (
                Q_(1994.6304671307805, ureg.delta_degC),
                Q_(1.1536458333333335),
            ),
        }


class IncroperaBoilingCurveImposedTemperatureExcess:
    @staticmethod
    def fetch() -> Tuple[Quantity[List[float]], Quantity[List[float]]]:
        temperature_excess = Q_(
            [
                1.2746231374307015,
                1.41943769421438,
                1.5807051579323592,
                1.7602948030042893,
                1.9602882788951492,
                2.1830037388142776,
                2.4310227097633987,
                2.707220015388242,
                3.014797098473843,
                3.3573191293293716,
                3.73875633019113,
                4.163529994640821,
                4.636563735456814,
                5.1005854499813505,
                5.666206728840396,
                6.157514827792382,
                6.63660805422927,
                6.9606549155014115,
                7.300524071452148,
                7.57238849329594,
                7.852598263994612,
                8.097782436011032,
                8.338959930173612,
                8.587320450572099,
                8.843077930376689,
                9.070887845218989,
                9.364575071689458,
                9.562957066988556,
                9.57075624690227,
                10.190795072339247,
                10.443102819379213,
                10.769171138675775,
                11.219820567102131,
                11.584162808000693,
                12.034669556183417,
                12.637964792282855,
                13.533723538632659,
                14.142818717126543,
                14.783329496381906,
                15.596312079772037,
                16.26223094126634,
                17.28351838818343,
                17.983017458139614,
                19.43637575992814,
                21.036881784318833,
                22.180498804037953,
                24.700505705860184,
                27.50681973004864,
                30.557144581112986,
                34.11217837552191,
                37.80242785594117,
                41.08021842280123,
                43.77726309353136,
                46.19722348381221,
                48.513079745481235,
                50.696446568551934,
                52.719574517855406,
                54.823438829888275,
                57.01126142656075,
                59.28639280605802,
                61.652317173895746,
                64.11265777873807,
                66.6711824611515,
                69.33180942379025,
                72.4521384829543,
                76.08414763683463,
                79.89822858003798,
                83.9035085297774,
                88.54160495014624,
                93.8942409138628,
                100.05869114785735,
                107.67608937030309,
                118.16283966470398,
                131.58774837053338,
                145.2001065180031,
                163.18661293822313,
                181.726835794458,
                202.37348060142597,
                224.26620827829402,
                247.31461551011628,
                270.0767074886286,
                293.494644757325,
                318.9431154660208,
                344.9069819384943,
                372.9844615586147,
                403.34761500704536,
                436.1825097271645,
                471.6903601588539,
                510.08876079409873,
                551.6130196106476,
                596.5176000551024,
                641.9300680195572,
                690.7997554297284,
                743.3898579856427,
                799.9836082920134,
                860.885801253787,
                921.9040109699844,
                992.0879189730508,
                1067.6148788383164,
                1157.0376829268007,
                1237.700587143832,
                1323.9869072726663,
                1417.8289832504615,
                1525.7673127733294,
                1649.9738463397575,
                1775.5852020351651,
                1864.594882664584,
            ],
            ureg.delta_degC,
        )

        heat_flux = Q_(
            [
                927.3859728835369,
                1083.6175026957746,
                1254.1553683184845,
                1448.5472067092805,
                1678.8194641540683,
                1943.0291944975213,
                2242.6558392351335,
                2600.9498611926983,
                3010.282893204471,
                3481.646184007933,
                4029.5816940906548,
                4666.951648306068,
                5398.252539966192,
                6270.594682135501,
                7054.445161641491,
                8180.1230731732285,
                9853.947035711928,
                11756.756947565125,
                13620.25703693745,
                15962.579216617805,
                19194.712026984093,
                23289.630537135912,
                28339.513153598567,
                34833.16441265502,
                43247.821152876924,
                53130.785773066455,
                65691.14004151053,
                75984.7948523468,
                92297.8530060746,
                107689.52694097382,
                137248.85922717495,
                164802.1543878676,
                201017.25013854934,
                240777.32908545755,
                285777.315338408,
                340384.4227293002,
                407471.2587615908,
                480774.38381223136,
                539050.4252323667,
                603787.8370538187,
                662348.049883574,
                728632.0505946386,
                816321.8407052283,
                882170.7987503833,
                971002.7218261592,
                1064527.098763494,
                1211936.6015993357,
                1340040.578384019,
                1404390.1676286734,
                1333222.9159596898,
                1190098.0299054477,
                1011822.328969292,
                855934.24632503,
                723699.0607196501,
                609281.5716119152,
                512953.5930815473,
                435291.395096861,
                369387.4089215119,
                311691.91849194403,
                263504.7895658826,
                222767.32248980235,
                188683.52169775806,
                160418.88614631273,
                137162.556472513,
                115607.84745431472,
                96452.53939044278,
                81448.84447758179,
                69300.16316434079,
                58711.88373944923,
                49728.85998562809,
                42401.26507997616,
                35991.329870519425,
                30694.49164446315,
                27117.75682187971,
                25685.425221246758,
                26069.635480597484,
                28171.48006408555,
                32249.047708619666,
                38231.59735093478,
                45035.66027602651,
                53040.62995340463,
                62812.56348954219,
                74634.7843174265,
                88116.8154876267,
                104230.74023051874,
                123524.29042756082,
                146665.6565577074,
                174306.76878690426,
                207352.76565073628,
                247129.6929518292,
                294259.34793302935,
                346432.40108628443,
                409011.967337166,
                482375.48822044925,
                569588.5880849218,
                672388.4642579944,
                787030.2065409685,
                930327.25637485,
                1097050.0052832037,
                1313347.779891056,
                1471848.0329926808,
                1795715.2892876896,
                2101883.435160469,
                2481902.26781383,
                2971998.3644059645,
                3515487.7961905403,
                3887703.8736272226,
            ],
            (ureg.W / ureg.m ** 2),
        )

        return temperature_excess.to(ureg.delta_degC), heat_flux.to(ureg.W / ureg.cm ** 2)

    @staticmethod
    def extra() -> Dict[Literal['A', 'B', 'C', 'D', 'E'], Tuple[Quantity[float], Quantity[float]]]:
        return {
            'A': (
                Q_(5.329765171785223, ureg.delta_degC),
                Q_(6526.742602759848, ureg.W / ureg.m ** 2),
            ),
            'B': (
                Q_(10, ureg.delta_degC),
                Q_(100738.36274301627, ureg.W / ureg.m ** 2),
            ),
            'C': (
                Q_(31.099515499401985, ureg.delta_degC),
                Q_(1382213.3392412085, ureg.W / ureg.m ** 2),
            ),
            'D': (
                Q_(141.62460655153197, ureg.delta_degC),
                Q_(26021.687873296054, ureg.W / ureg.m ** 2),
            ),
            'E': (
                Q_(1187.228084422286, ureg.delta_degC),
                Q_(1402700.1913364157, ureg.W / ureg.m ** 2),
            ),
        }