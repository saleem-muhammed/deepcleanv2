from deepclean.couplings import Coupling, SubtractionProblem


class Sub104Hz(SubtractionProblem):
    description = "Anomalous noise in O4"

    L1 = Coupling(
        101,
        107,
        [
            "PEM-EX_ACC_BEAMTUBE_3950X_X_DQ",
            "PEM-EX_ACC_BEAMTUBE_MAN_X_DQ",
            "PEM-EX_ACC_BEAMTUBE_MAN_Z_DQ",
            "PEM-EX_ACC_BSC4_ETMX_X_DQ",
            "PEM-EX_ACC_BSC4_ETMX_Y_DQ",
            "PEM-EX_ACC_BSC4_ETMX_Z_DQ",
            "PEM-EX_ACC_EBAY_FLOOR_Z_DQ",
            "PEM-EX_ACC_HVAC_FLOOR_Z_DQ",
            "PEM-EX_ACC_ISCTEX_TRANS_Y_DQ",
            "PEM-EX_ACC_OPLEV_ETMX_Y_DQ",
            "PEM-EX_ACC_VEA_FLOOR_Z_DQ",
            "PEM-EY_ACC_BEAMTUBE_3950Y_Y_DQ",
            "PEM-EY_ACC_BEAMTUBE_MAN_Y_DQ",
            "PEM-EY_ACC_BEAMTUBE_MAN_Z_DQ",
            "PEM-EY_ACC_BSC5_ETMY_X_DQ",
            "PEM-EY_ACC_BSC5_ETMY_Y_DQ",
            "PEM-EY_ACC_BSC5_ETMY_Z_DQ",
            "PEM-EY_ACC_EBAY_FLOOR_Z_DQ",
            "PEM-EY_ACC_HVAC_FLOOR_Z_DQ",
            "PEM-EY_ACC_ISCTEY_TRANS_X_DQ",
            "PEM-EY_ACC_OPLEV_ETMY_X_DQ",
            "PEM-EY_ACC_VEA_FLOOR_Z_DQ",
            "PEM-MX_ACC_BEAMTUBE_1900X_X_DQ",
            "PEM-MX_ACC_BEAMTUBE_2100X_X_DQ",
            "PEM-MX_ACC_BEAMTUBE_VEA_X_DQ",
            "PEM-MX_ACC_BEAMTUBE_VEA_Z_DQ",
            "PEM-MY_ACC_BEAMTUBE_1900Y_Y_DQ",
            "PEM-MY_ACC_BEAMTUBE_2100Y_Y_DQ",
            "PEM-MY_ACC_BEAMTUBE_VEA_Y_DQ",
            "PEM-MY_ACC_BEAMTUBE_VEA_Z_DQ",
        ],
    )
