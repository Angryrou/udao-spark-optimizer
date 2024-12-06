from typing import List, Optional

from ..utils import BenchmarkType


class Benchmark:
    def __init__(
        self,
        benchmark_type: BenchmarkType,
        scale_factor: int = 100,
        ext: Optional[str] = None,
    ):
        self.benchmark_type = benchmark_type
        self.scale_factor = scale_factor
        self.templates = self._get_templates(benchmark_type, ext)
        self.template2id = {t: i for i, t in enumerate(self.templates)}

    def get_name(self) -> str:
        return self.benchmark_type.value

    def get_prefix(self) -> str:
        if self.benchmark_type in [
            BenchmarkType.JOB_TRAIN,
            BenchmarkType.JOB_SYNTHETIC,
            BenchmarkType.JOB_LIGHT,
            BenchmarkType.JOB_EXT,
        ]:
            return self.get_name()
        else:
            return f"{self.get_name()}{self.scale_factor}"

    def get_template_id(self, template: str) -> int:
        return self.template2id[template]

    def _get_templates(
        self, benchmark_type: BenchmarkType, ext: Optional[str]
    ) -> List[str]:
        if ext and benchmark_type not in (BenchmarkType.JOB, BenchmarkType.TPCDS):
            raise ValueError(f"{benchmark_type} does not support extension")
        if benchmark_type == BenchmarkType.TPCH:
            return [str(i) for i in range(1, 23)]
        elif benchmark_type == BenchmarkType.TPCDS:
            return [
                t
                for t in "1 2 3 4 5 6 7 8 9 10 11 12 13 14a 14b 15 16 17 18 "
                "19 20 21 22 23a 23b 24a 24b 25 26 27 28 29 30 31 "
                "32 33 34 35 36 37 38 39a 39b 40 41 42 43 44 45 46 "
                "47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 "
                "64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 "
                "81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 "
                "98 99".split()
                if t not in ["61"]  # to be dropped due to parsing issue
            ]
        elif benchmark_type == BenchmarkType.TPCDS_EXT:
            return [str(i) for i in range(101, 1101)]
        elif benchmark_type == BenchmarkType.TPCDS_EXT_SELECTED:
            return [
                t
                for t in (
                    "341 650 454 698 406 1071 729 1011 419 466 274 165 "
                    "927 486 1069 1089 429 799 239 489 1024 1009 501 897 "
                    "1021 982 597 356 310 763 987 104 101 600 721 340 1049 "
                    "335 1084 725 992 1031 1079 989 912 269 952 330 174 326 "
                    "607 816 400 665 1019 167 868 118 302 114 261 1099 142 "
                    "626 757 199 266 1059 153 789 194 352 1091 534 1054 678 "
                    "112 999 107 937 436 301 857 877 778 669 195 977 1094 "
                    "979 617 577 206 932 687 967 560 229 470 661 592 1029 "
                    "384 1044 1034 809 398 553 639 612 132 411 887 334 695 "
                    "892 283 1074 1039 947 962 797 540 1041 1001 1051 922 "
                    "508 1014 949 565 734 259 1061 847 785 155 869 703 629 "
                    "441 819 102 717 376 959 806 972 837 957 185 369 969 296 "
                    "360 939 904 929 1064 1004 917 942 919 901 744 529 145 "
                    "997 133 1081 205 827 546 277"
                ).split()
            ]
        elif benchmark_type == BenchmarkType.TPCDS_EXT_STAR_JOINS:
            templates = []
            for sign, num_per_sign in [
                ("ss", 511),
                ("sr", 1000),
                ("cs", 1000),
                ("cr", 1000),
                ("ws", 1000),
                ("wr", 1000),
                ("inv", 7),
            ]:
                templates += [f"{sign}{i}" for i in range(num_per_sign)]
            return templates
        elif benchmark_type == BenchmarkType.TPCXBB:
            return [str(i) for i in range(1, 31)]
        elif benchmark_type == BenchmarkType.JOB_TRAIN:
            return [str(i) for i in range(100000)]
        elif benchmark_type == BenchmarkType.JOB_SYNTHETIC:
            return [str(i) for i in range(5000)]
        elif benchmark_type == BenchmarkType.JOB_LIGHT:
            return [str(i) for i in range(70)]
        elif benchmark_type == BenchmarkType.JOB_EXT:
            return [str(i) for i in range(40000)]
        elif benchmark_type == BenchmarkType.JOB:
            res = (
                ["TRAIN" + str(i) for i in range(100000)]
                + ["SYNTHETIC" + str(i) for i in range(5000)]
                + ["LIGHT" + str(i) for i in range(70)]
            )
            if ext:
                res += ["EXT" + str(i) for i in range(40000)]
            return res
        else:
            raise ValueError(f"{benchmark_type} is not supported")
