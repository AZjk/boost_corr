import numpy as np

nexus_xpcs_schema = {
    "twotime": {
        "type": "NXdata",
        "required": False,
        "description": "The data (results) in this section are based on the two-time intensity correlation function derived from a time series of scattering images",
        "g2_err_from_two_time_corr_func": {
            "type": "NX_NUMBER",
            "units": "NX_DIMENSIONLESS",
            "required": False,
            "description": "error values for the g2 values",
            "data": np.zeros((2, 3)),
            "attributes": {
                "storage_mode": {
                    "type": "NX_CHAR", 
                    "required": True,
                    "data": "one_array_q_first",
                    "description": "Any of these values: one_array_q_first one_array_q_last data_exchange_keys other"
                }
            }
        },
        "g2_from_two_time_corr_func": {
            "type": "NX_NUMBER",
            "units": "NX_DIMENSIONLESS",
            "required": False,
            "data": np.zeros((2, 3)),
            "description": "frame weighted average along the diagonal direction in two_time_corr_func",
            "attributes": {
                "storage_mode": {
                    "type": "NX_CHAR", 
                    "required": True,
                    "data": "one_array_q_first",
                    "description": "Any of these values: one_array_q_first one_array_q_last data_exchange_keys other"
                },
                "baseline_reference": {
                    "type": "NX_INT", 
                    "required": True,
                    "data": 1.0,
                    "description": "Any of these values: 0 | 1"
                },
                "first_point_for_fit": {
                    "type": "NX_INT", 
                    "required": True,
                    "data": 0,
                    "description": "first_point_for_fit describes if the first point should or should not be included in fitting"
                }
            }
        },
        "two_time_corr_func": {
            "type": "NX_NUMBER",
            "units": "NX_ANY",
            "required": False,
            "description": "two-time correlation of speckle intensity for a given q-bin or roi",
            "data": np.zeros((2, 3)),
            "attributes": {
                "storage_mode": {
                    "type": "NX_CHAR", 
                    "required": True,
                    "data": "one_array_q_first",
                    "description": "storage_mode describes the format of the data to be loaded"
                },
                "baseline_reference": {
                    "type": "NX_INT", 
                    "required": True,
                    "data": 1,
                    "description": "baseline is the expected value of a full decorrelation"
                },
                "populated_elements": {
                    "type": "NX_CHAR", 
                    "required": True,
                    "data": "upper_half",
                    "description": "populated_elements describe the elements of the 2D array that are populated"
                },
                "time_origin_location": {
                    "type": "NX_CHAR", 
                    "required": True,
                    "data": "upper_left",
                    "description": "time_origin_location is the location of the origin"
                }
            }
        }
    },
    "multitau": {
        "type": "NXdata",
        "required": True,
        "description": "The multitau results data captured here are most commonly required for high throughput, equilibrium dynamics experiments",
        "delay_difference": {
            "type": "NX_INT",
            "units": "NX_COUNT",
            "required": True,
            "data": 1,
            "description": "delay_difference (also known as delay or lag step)",
            "attributes": {
                "storage_mode": {
                    "type": "NX_CHAR", 
                    "required": True,
                    "data": "one_array_q_first",
                    "description": "Any of these values: one_array | data_exchange_keys | other"
                }
            }
        },
        "frame_average": {
            "type": "NX_NUMBER", 
            "units": "NX_COUNT", 
            "required": True,
            "description": "Two-dimensional average along the frames stack",
            "data": 1
        },
        "frame_sum": {
            "type": "NX_NUMBER", 
            "units": "NX_COUNT", 
            "required": True,
            "description": "Two-dimensional summation along the frames stack",
            "data": 1,
        },
        "g2": {
            "type": "NX_NUMBER",
            "units": "NX_DIMENSIONLESS",
            "required": True,
            "description": "normalized intensity auto-correlation function",
            "data": None,
            "attributes": {
                "storage_mode": {
                    "type": "NX_CHAR", 
                    "required": True,
                    "data": "one_array_q_first",
                    "description": "storage_mode describes the format of the data to be loaded"
                }
            }
        },
        "g2_derr": {
            "type": "NX_NUMBER",
            "units": "NX_DIMENSIONLESS",
            "required": True,
            "description": "error values for the g₂ values",
            "data": None,
            "attributes": {
                "storage_mode": {
                    "type": "NX_CHAR", 
                    "required": True,
                    "data": "one_array_q_first",
                    "description": "Any of these values: one_array | data_exchange_keys | other"
                }
            }
        },
        "G2_unnormalized": {
            "type": "NX_NUMBER",
            "units": "NX_ANY",
            "required": True,
            "description": "unnormalized intensity auto-correlation function",
            "data": None,
            "attributes": {
                "storage_mode": {
                    "type": "NX_CHAR", 
                    "required": True,
                    "data": "one_array_q_first",
                    "description": "Any of these values: one_array | data_exchange_keys | other"
                }
            }
        }
    },
    
}