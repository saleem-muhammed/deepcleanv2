from deepclean.couplings import Coupling, SubtractionProblem


class Sub60Hz_O3_MDC(SubtractionProblem):
    description = "Subtraction of 60Hz main noise as applied in the O3 DeepClean MDC study"

    H1 = Coupling(
        55,
        65,
        [
            "PEM-CS_MAINSMON_EBAY_1_DQ",
            "ASC-INP1_P_INMON",
            "ASC-INP1_Y_INMON",
            "ASC-MICH_P_INMON",
            "ASC-MICH_Y_INMON",
            "ASC-PRC1_P_INMON",
            "ASC-PRC1_Y_INMON",
            "ASC-PRC2_P_INMON",
            "ASC-PRC2_Y_INMON",
            "ASC-SRC1_P_INMON",
            "ASC-SRC1_Y_INMON",
            "ASC-SRC2_P_INMON",
            "ASC-SRC2_Y_INMON",
            "ASC-DHARD_P_INMON",
            "ASC-DHARD_Y_INMON",
            "ASC-CHARD_P_INMON",
            "ASC-CHARD_Y_INMON",
            "ASC-DSOFT_P_INMON",
            "ASC-DSOFT_Y_INMON",
            "ASC-CSOFT_P_INMON",
            "ASC-CSOFT_Y_INMON"
        ],
    )
    L1 = Coupling(
        58,
        62,
        [
            "PEM-CS_MAINSMON_EBAY_1_DQ",
            "ASC-INP1_P_INMON",
            "ASC-MICH_P_INMON",
            "ASC-MICH_Y_INMON",
            "ASC-PRC1_P_INMON",
            "ASC-PRC1_Y_INMON",
            "ASC-PRC2_P_INMON",
            "ASC-PRC2_Y_INMON",
            "ASC-SRC1_P_INMON",
            "ASC-SRC1_Y_INMON",
            "ASC-SRC2_P_INMON",
            "ASC-SRC2_Y_INMON",
            "ASC-DHARD_P_INMON",
            "ASC-DHARD_Y_INMON",
            "ASC-CHARD_P_INMON",
            "ASC-CHARD_Y_INMON",
            "ASC-DSOFT_P_INMON",
            "ASC-CSOFT_P_INMON",
        ],
    )