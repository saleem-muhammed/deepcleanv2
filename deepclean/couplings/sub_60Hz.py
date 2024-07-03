from deepclean.couplings import Coupling, SubtractionProblem


class Sub60HzRestricted(SubtractionProblem):
    description = "Subtraction of 60Hz main noise"

    H1 = Coupling(
        55,
        65,
        [
            "PEM-CS_MAINSMON_EBAY_1_DQ",
            #            "ASC-INP1_P_OUT_DQ",
            #            "ASC-INP1_Y_OUT_DQ",
            #            "ASC-MICH_P_OUT_DQ",
            #            "ASC-MICH_Y_OUT_DQ",
            #            "ASC-PRC1_P_OUT_DQ",
            #            "ASC-PRC1_Y_OUT_DQ",
            #            "ASC-PRC2_P_OUT_DQ",
            #            "ASC-PRC2_Y_OUT_DQ",
            #            "ASC-SRC1_P_OUT_DQ",
            #            "ASC-SRC1_Y_OUT_DQ",
            #            "ASC-SRC2_P_OUT_DQ",
            #            "ASC-SRC2_Y_OUT_DQ",
            #            "ASC-DHARD_P_OUT_DQ",
            #            "ASC-DHARD_Y_OUT_DQ",
            #            "ASC-CHARD_P_OUT_DQ",
            #            "ASC-CHARD_Y_OUT_DQ",
            #            "ASC-DSOFT_P_OUT_DQ",
            #            "ASC-DSOFT_Y_OUT_DQ",
            #            "ASC-CSOFT_P_OUT_DQ",
            #            "ASC-CSOFT_Y_OUT_DQ",
        ],
    )
    L1 = Coupling(
        58,
        62,
        [
            "PEM-CS_MAINSMON_EBAY_1_DQ",
            #            "ASC-CHARD_P_OUT_DQ",
            #            "ASC-CHARD_Y_OUT_DQ",
            #            "ASC-CSOFT_P_OUT_DQ",
            #            "ASC-DHARD_P_OUT_DQ",
            #            "ASC-DHARD_Y_OUT_DQ",
            #            "ASC-DSOFT_P_OUT_DQ",
            #            "ASC-INP1_P_OUT_DQ",
            #            "ASC-MICH_P_OUT_DQ",
            #            "ASC-MICH_Y_OUT_DQ",
            #            "ASC-PRC1_P_OUT_DQ",
            #            "ASC-PRC1_Y_OUT_DQ",
            #            "ASC-PRC2_P_OUT_DQ",
            #            "ASC-PRC2_Y_OUT_DQ",
            #            "ASC-SRC1_P_OUT_DQ",
            #            "ASC-SRC1_Y_OUT_DQ",
            #            "ASC-SRC2_P_OUT_DQ",
            #            "ASC-SRC2_Y_OUT_DQ",
        ],
    )
    K1 = Coupling(
        55,
        65,
        [
            "PEM-MIC_BS_BOOTH_BS_Z_OUT_DQ",
            "PEM-MIC_BS_FIELD_BS_Z_OUT_DQ",
            "PEM-MIC_BS_TABLE_POP_Z_OUT_DQ",
            "PEM-MIC_IXC_BOOTH_IXC_Z_OUT_DQ",
            "PEM-MIC_IXC_FIELD_IXC_Z_OUT_DQ",
            "PEM-MIC_IYC_BOOTH_IYC_Z_OUT_DQ",
            "PEM-MIC_OMC_BOOTH_OMC_Z_OUT_DQ",
            "PEM-MIC_OMC_TABLE_AS_Z_OUT_DQ",
            "PEM-MIC_TMSX_TABLE_TMS_Z_OUT_DQ",
            #            "PEM-VOLT_AS_TABLE_GND_OUT_DQ",
            #            "PEM-VOLT_IMCREFL_TABLE_GND_OUT_DQ",
            #            "PEM-VOLT_ISS_TABLE_GND_OUT_DQ",
            #            "PEM-VOLT_OMC_CHAMBER_GND_OUT_DQ",
            #            "PEM-VOLT_PSL_TABLE_GND_OUT_DQ",
            #            "PEM-VOLT_REFL_TABLE_GND_OUT_DQ",
        ],
    )
