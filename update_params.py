import numpy as np


def update_params(neta, __weight_hidden, __weight_output, __dC_dw_hidden, __dC_dw_output, __dC_db_hidden, __dC_db_output, __bias_hidden, __bias_output):
    __weight_hidden = __weight_hidden - neta * __dC_dw_hidden
    __weight_output = __weight_output - neta * __dC_dw_output
    # print(weight_list)
    print("updated weight_hidden")
    print(__weight_hidden)
    print("updated weight_output")
    print(__weight_output)

    __bias_hidden = __bias_hidden - neta * __dC_db_hidden
    __bias_output = __bias_output - neta * __dC_db_output
    return [__weight_hidden, __weight_output, __bias_hidden, __bias_output]
