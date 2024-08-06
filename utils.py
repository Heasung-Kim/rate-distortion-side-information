
def get_simulation_name(config):
    """
    """
    config["name"] = config["task"] + "_" + config["side_information"] + "_" + config["distortion_metric"] + "_" + \
                     config["dataset"] + "_" + config["model_type"]

    if config["task"] == "compression":
        config["name"] = config["name"] + "_" + str(config["n_codeword_bits"]) + "bit"
    elif config["task"] == "rd_estimation":
        config["name"] = config["name"] + "_" + str(config["lmbda"])

    if config["dataset"] == "CDL":
        config["name"] = config["name"] + "_maxspeed" + str(config["max_ue_speed"])

    return


if __name__ == "__main__":
    print("hello world!")

