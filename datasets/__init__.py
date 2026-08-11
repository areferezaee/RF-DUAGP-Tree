from datasets.capsulorhexis_loader import CL
#from .jigsaws import JIGSAWSDataset


def load_dataset(config):
    if config.DATASET == "capsulorhexis":
        return CapsulorhexisDataset(
            config.SPLIT_FILE,
            config.DATA_PATH
        )

    elif config.DATASET == "jigsaws":
        return JIGSAWSDataset(
            config.SPLIT_FILE,
            config.DATA_PATH
        )

    else:
        raise ValueError("Unsupported dataset")
